"""Deciding what a full redesign actually changes, before anything is drawn.

An image model asked to "redesign this room" with the photograph in front of it
mostly polishes what it sees: the same bed, a little shinier; the same desk,
a little cleaner. That is a restyle wearing the wrong name, and it is the
complaint a client raised about Full redesign. The fix is not a louder
adjective, it is a decision made first and handed over as an instruction: this
bed becomes *that* bed, this desk becomes *that* desk, this wall has nowhere to
store things so a wardrobe goes here.

So a full redesign makes one plan, in words, by looking at the photograph(s),
and the drawing prompt is built from it. The plan is:

- **Specific.** Material, colour, shape and size for every new piece. Vague
  plans produce vague pictures.
- **The same for every angle.** A room photographed from three corners is one
  design seen three times, so the three requests must not each invent their own
  bed. The plan is keyed on the photographs themselves (in a fixed order, not
  upload order) and the brief, cached for a while, and concurrent requests for
  the same plan share one call.
- **Optional.** If planning fails for any reason the drawing carries on with the
  stronger wording alone. A plan is an improvement, never a dependency.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import logging
import time

from PIL import Image

from .config import Settings, resolve_backend

log = logging.getLogger(__name__)

PLAN_EDGE = 1024           # long edge the planner sees; it needs layout, not pixels
CACHE_SECONDS = 30 * 60
CACHE_LIMIT = 64

_cache: dict[str, tuple[float, dict | None]] = {}
_inflight: dict[str, asyncio.Future] = {}


# Every layer of a space a full redesign must address. A redesign that does the
# ceiling and the TV unit and leaves the walls as they were is half a job, and
# an image model told only "finishes" will happily do whichever few it likes.
# So the plan has a slot for each, the planner is told every slot is mandatory,
# and anything it leaves blank is filled with a default instruction below —
# the drawing is never left to forget a layer.
LAYERS = (
    ("walls", "Walls"),
    ("wall_features", "Wall design"),
    ("wall_decor", "Wall decor"),
    ("ceiling", "Ceiling"),
    ("floor", "Floor"),
    ("lighting", "Lighting"),
    ("window_dressing", "Window dressing"),
    ("textiles", "Soft furnishings"),
)

DEFAULT_LAYER = {
    "walls": "a new colour or finish on every wall, clearly different from now",
    "wall_features": ("a real design on the main wall: fluted or slatted wood, "
                      "panelling, textured plaster or a wallpaper feature, not "
                      "just paint"),
    "wall_decor": "a few chosen pieces on the walls: framed art, a mirror or shelves",
    "ceiling": ("a visible ceiling treatment: a false ceiling with cove "
                "lighting, or coving and a new colour (the height is unchanged)"),
    "floor": "a new floor finish or a new large rug, clearly different from now",
    "lighting": ("layered lighting: ambient, a task light, and one accent light, "
                 "in new fittings"),
    "window_dressing": "new curtains, sheers or blinds, different from now",
    "textiles": "new cushions, throws, bedding or upholstery fabrics",
}
# The same layers for a space people work or shop in: the walls carry the
# brand, the lighting carries the mood, and "soft furnishings" is upholstery.
DEFAULT_LAYER_COMMERCIAL = {
    **DEFAULT_LAYER,
    "wall_features": ("a feature wall that carries the brand: panelling, a "
                      "textured or colour-blocked finish, or a signage wall"),
    "wall_decor": "wall graphics, framed work, mirrors or display shelving",
    "lighting": ("a lighting scheme for the space: ambient, task or display "
                 "lighting, and feature pendants"),
    "textiles": "upholstery, acoustic panels and soft seating fabrics",
}

PLANNER = """\
You are a senior interior designer briefing a renovation of the room in the \
attached photograph(s). The photographs show the room as it is now; several \
photographs are the SAME room from different corners.

Write the plan as JSON only, exactly this shape:
{{"replace": [{{"current": "", "new": ""}}],
  "add": [{{"piece": "", "where": ""}}],
  "finishes": {{"walls": "", "wall_features": "", "wall_decor": "", "ceiling": "",
               "floor": "", "lighting": "", "window_dressing": "", "textiles": ""}},
  "zones": [{{"zone": "", "changes": ""}}],
  "fixed": [""]}}

The brief:
- Direction: {style}
- The room is for: {room}
- How far to go: a FULL redesign. This is a renovation, not a tidy-up. The \
loose furniture and furnishings are being replaced, not cleaned, recoloured or \
polished.
{kept}{words}{nudge}
Rules:
1. "replace": every movable piece you can see in ANY photograph — bed, desk, \
chair, sofa, table, shelving, rug, curtains, lamps, bedside units, mirrors, \
art, laundry baskets, storage — gets an entry. "current" says what it is now, \
briefly ("the white two-drawer desk"). "new" is a DIFFERENT design, not a \
better version of the same one: a different silhouette AND a different \
material or colour. Be specific enough to draw: material, colour, shape, \
rough size ("a walnut writing desk, 120cm, tapered legs, one slim drawer"). \
Clutter, bags, clothes and bottles are simply removed: put "removed" as "new".
2. "add": one to three pieces this room genuinely lacks for what it is for \
(storage if there is nowhere to put things away, proper seating, a light at the \
right height). Say where each goes, on a wall or floor you can actually see, \
clear of every door, window and walkway. Size it for the space: a small room \
gets small pieces. Never a second bed, and never anything that fills the room.
3. "finishes": EVERY one of the eight must be filled in with a concrete, \
visible design. A layer left as it is now counts as a failure, and a colour \
alone is not a design. All together, in one palette that suits the direction:
   - walls: the colour or finish of the walls overall;
   - wall_features: a REAL DESIGN on at least one wall, not just paint: fluted \
or slatted wood, panelling, textured plaster, wallpaper, a stone or tile \
feature, built-in niches. Say which wall;
   - wall_decor: what hangs or sits on the walls (art, mirror, shelves);
   - ceiling: a real treatment (false ceiling with cove lighting, coving, a \
colour); its height does not change;
   - floor: the actual finish or a new large rug ("matte wide-plank oak");
   - lighting: new fittings at ambient, task and accent level;
   - window_dressing: curtains, sheers or blinds;
   - textiles: bedding, cushions, throws or upholstery fabrics.
   Describe each as something a builder could price ("fluted oak slats behind \
the headboard, floor to ceiling").
   SCALE: judge the size of the space. In a small or ordinary room, do all of \
the above. In a LARGE space (a big living area, open plan, an office floor, a \
shop floor, a lobby, a studio) also fill "zones": one row per distinct area \
(reception, seating cluster, workstations, display wall, dining end, entrance) \
saying what changes there, so no part of the space is left untouched. \
Commercial spaces: the walls carry the brand, so the feature wall matters.
4. "fixed": things that do not move or change: every door and window (say where \
each is), the air conditioner, ceiling fan and light switches where mounted, \
built-in storage. List them so the drawing knows what to leave alone.
5. At most one or two plants in the whole room. No cut flowers.
6. Count what is there. If there is one bed, the plan has one bed.
7. Everything must look like a real, buyable, believable room, in proportion.
"""

# One extra nudge per option, so the designs a person asks for are alternatives
# rather than the same plan drawn three times. Kept in step with
# openai_images.VARIATIONS, which does the same job in the drawing prompt.
OPTION_NUDGES = (
    "",
    "\nThis is option 2 of several. Make it a clearly different design from the "
    "obvious one: change the dominant material and the colour story.",
    "\nThis is option 3 of several. Take the quiet version: tighter palette, "
    "fewer and simpler pieces, more empty floor.",
    "\nThis is option 4 of several. Change the furniture's silhouettes boldly "
    "and lead with a statement piece.",
)


def _shrink(photo: bytes) -> bytes:
    """A planner needs layout, not pixels. Small and JPEG keeps it fast."""
    image = Image.open(io.BytesIO(photo))
    image = image.convert("RGB")
    scale = PLAN_EDGE / max(image.size)
    if scale < 1:
        image = image.resize((round(image.width * scale),
                              round(image.height * scale)), Image.LANCZOS)
    out = io.BytesIO()
    image.save(out, format="JPEG", quality=85)
    return out.getvalue()


def plan_key(photos: list[bytes], style: str, room: str, depth: str,
             extra: str, keep: str, option: int) -> str:
    """Same room, same brief, same option → same key, whichever photo is the
    one being drawn and whatever order they were uploaded in."""
    digests = sorted(hashlib.sha1(p).hexdigest() for p in photos)
    brief = "|".join([style.strip().lower(), room.strip().lower(),
                      depth.strip().lower(), extra.strip(), keep.strip(),
                      str(option)])
    return hashlib.sha1(("/".join(digests) + "#" + brief).encode()).hexdigest()


def _clean(plan: object) -> dict | None:
    """Only keep what is usable. A half-valid plan is better than none, an
    empty one is worse than none (it would tell the drawing to change nothing)."""
    if not isinstance(plan, dict):
        return None

    def rows(name, a, b):
        out = []
        for row in plan.get(name) or []:
            if isinstance(row, dict) and str(row.get(a, "")).strip() \
                    and str(row.get(b, "")).strip():
                out.append({a: str(row[a]).strip()[:160], b: str(row[b]).strip()[:200]})
        return out[:14]

    finishes = plan.get("finishes") if isinstance(plan.get("finishes"), dict) else {}
    keys = tuple(k for k, _ in LAYERS)
    cleaned = {
        "replace": rows("replace", "current", "new"),
        "add": rows("add", "piece", "where"),
        "finishes": {k: str(v).strip()[:160] for k, v in finishes.items()
                     if k in keys and str(v).strip()},
        "zones": rows("zones", "zone", "changes")[:8],
        "fixed": [str(x).strip()[:120] for x in (plan.get("fixed") or [])
                  if str(x).strip()][:12],
    }
    return cleaned if cleaned["replace"] or cleaned["add"] else None


async def ask_json(prompt: str, photos: list[bytes], settings: Settings,
                   seed: int = 0, max_tokens: int = 2600) -> dict:
    """One vision question, JSON back, on whichever engine is configured.

    Shared by the planner here and the self-check (app/checking.py), so both
    behave the same under the free and paid keys.
    """
    jpegs = [_shrink(p) for p in photos]
    backend = resolve_backend(settings)

    if backend == "free":
        from . import google_ai

        payload = await google_ai._call(
            settings.google_vision_model,
            {"contents": [{"parts": [{"text": prompt}] +
                          [google_ai._part_image(j) for j in jpegs]}],
             "generationConfig": {"response_mime_type": "application/json",
                                  "temperature": 0.2,
                                  "maxOutputTokens": max(max_tokens, 4096),
                                  "thinkingConfig": {"thinkingLevel": "low"}}},
            settings)
        text = "".join(part.get("text", "") for part in google_ai._parts(payload))
        return json.loads(text)

    from openai import AsyncOpenAI

    content = [{"type": "text", "text": prompt}]
    for j in jpegs:
        content.append({"type": "image_url", "image_url": {
            "url": "data:image/jpeg;base64," + base64.b64encode(j).decode("ascii")}})
    response = await AsyncOpenAI(
        api_key=settings.openai_api_key, timeout=settings.request_timeout_s
    ).chat.completions.create(
        model=settings.vlm_model,
        messages=[{"role": "user", "content": content}],
        response_format={"type": "json_object"},
        max_tokens=max_tokens,
        temperature=0.2,
        seed=seed,
    )
    return json.loads(response.choices[0].message.content or "{}")


async def _ask(prompt: str, photos: list[bytes], settings: Settings,
               seed: int) -> dict | None:
    return _clean(await ask_json(prompt, photos, settings, seed))


async def make_plan(photos: list[bytes], *, style_text: str, room_text: str,
                    keep: str = "", extra: str = "", option: int = 0,
                    key_parts: tuple[str, str, str, str] = ("", "", "", ""),
                    settings: Settings) -> dict | None:
    """The redesign plan for these photographs, or None if there isn't one.

    Never raises: planning failing must not stop a design being drawn.
    `key_parts` is (style id, room id, depth id, extra) — the raw inputs, kept
    separate from the prose so the cache key does not change when wording does.
    """
    if not photos:
        return None
    key = plan_key(photos, key_parts[0], key_parts[1], key_parts[2],
                   key_parts[3], keep, option)

    hit = _cache.get(key)
    if hit and time.time() - hit[0] < CACHE_SECONDS:
        return hit[1]
    pending = _inflight.get(key)
    if pending is not None:
        return await asyncio.shield(pending)

    future: asyncio.Future = asyncio.get_running_loop().create_future()
    _inflight[key] = future
    plan: dict | None = None
    try:
        prompt = PLANNER.format(
            style=style_text, room=room_text or "a room",
            kept=(f"- These pieces were deliberately kept by the client and "
                  f"must NOT be replaced: {keep.strip()}. Leave them out of "
                  f"\"replace\" and design around them.\n" if keep.strip() else ""),
            words=(f"- The client also asks: {extra.strip()[:600]}\n"
                   if extra.strip() else ""),
            nudge=OPTION_NUDGES[option % len(OPTION_NUDGES)])
        plan = await _ask(prompt, photos, settings,
                          seed=int(key[:8], 16) % (2 ** 31))
    except Exception as exc:       # noqa: BLE001 — see the docstring
        log.warning("Redesign plan unavailable (%s); drawing without one", exc)
        plan = None
    finally:
        _inflight.pop(key, None)
        if not future.done():
            future.set_result(plan)

    if plan is not None:           # a failure is not remembered: try again next time
        if len(_cache) >= CACHE_LIMIT:
            _cache.pop(min(_cache, key=lambda k: _cache[k][0]))
        _cache[key] = (time.time(), plan)
    return plan


def missing_layers(plan: dict | None) -> list[str]:
    """Layers the planner left blank (and which render_plan will fill)."""
    done = (plan or {}).get("finishes", {})
    return [k for k, _ in LAYERS if not done.get(k)]


def render_plan(plan: dict | None, commercial: bool = False) -> str:
    """The plan as an instruction the image model can follow.

    Every layer in LAYERS appears, planner's own words where it gave them and a
    default instruction where it did not, so a layer can be neglected by the
    planner but never silently dropped from what the drawing is told to do.
    """
    if not plan:
        return ""
    fallback = DEFAULT_LAYER_COMMERCIAL if commercial else DEFAULT_LAYER
    lines = ["THE REDESIGN — carry out every line of this. Each piece named "
             "below is taken out of the room and replaced; the new piece is "
             "not a cleaner version of the old one."]
    for row in plan["replace"]:
        new = row["new"]
        if new.strip().lower().rstrip(".") == "removed":
            lines.append(f"- Remove the {row['current']}.")
        else:
            lines.append(f"- Take out the {row['current']}. In its place: {new}.")
    for row in plan["add"]:
        lines.append(f"- Add {row['piece']}, {row['where']}.")
    for row in plan.get("zones", []):
        lines.append(f"- {row['zone'].capitalize()}: {row['changes']}.")
    lines.append("EVERY layer of the space changes — none is left as it was, "
                 "and the walls change by design, not only by colour:")
    done = plan["finishes"]
    for key, label in LAYERS:
        lines.append(f"- {label}: {done.get(key) or fallback[key]}.")
    if plan["fixed"]:
        lines.append("Leave exactly where they are: " + "; ".join(plan["fixed"]) + ".")
    return "\n".join(lines)


def changes_of(plan: dict | None) -> list[str]:
    """One short line per change, for showing under a design."""
    if not plan:
        return []
    lines = []
    for row in plan["replace"]:
        if row["new"].strip().lower().rstrip(".") == "removed":
            lines.append(f"{row['current']}: removed")
        else:
            lines.append(f"{row['current']} \u2192 {row['new']}")
    lines += [f"Added {row['piece']}, {row['where']}" for row in plan["add"]]
    lines += [f"{label}: {plan['finishes'][key]}" for key, label in LAYERS
              if plan["finishes"].get(key)]
    return lines
