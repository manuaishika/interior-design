"""The whole job on one OpenAI key.

    GPT-4o        -> where the doors and windows are, as boxes
    gpt-image-2.5 -> redraws the room

Why the mask is off by default
------------------------------
It used to send an inpainting mask: the doors and windows painted out, the
rest of the frame handed over to be repainted. That is exactly what happened.
Everything except the openings was inside the repaint zone, so the desk, the
television, the office chair and the wardrobe were erased and something new
was invented in their place. A cupboard came back as a doorway. A one-bed
room came back with two beds, and an air conditioner appeared on a wall it
could not physically be on.

A mask is the right tool for a weak inpainting model that needs to be told
where to look. On a strong editing model it is a demolition order. Given the
same photograph and the same account, ChatGPT keeps all of that furniture —
because nobody hands it a mask saying "replace this".

So the protection moved into words (`generation.PRESERVE`), where it can name
what a rectangle cannot: that a wardrobe stays a wardrobe, that the air
conditioner cannot move to another wall, that one bed stays one bed.

The mask is still computed and still returned, so the openings it found can be
inspected. `USE_INPAINT_MASK=true` sends it again.

Mask convention
---------------
Ours is white-repaints. OpenAI's is *transparent*-repaints. Getting that
backwards does not error — it repaints precisely the door and preserves
everything else — so the conversion is done in one place and tested.
"""

from __future__ import annotations

import base64
import io
import json
import logging

import numpy as np
from openai import AsyncOpenAI
from PIL import Image

from .config import Settings
from .imaging import Mask

log = logging.getLogger(__name__)


class OpenAIImageError(RuntimeError):
    pass


def _client(settings: Settings) -> AsyncOpenAI:
    if not settings.openai_api_key:
        raise OpenAIImageError(
            "No OPENAI_API_KEY set. Make one at platform.openai.com/api-keys."
        )
    return AsyncOpenAI(api_key=settings.openai_api_key,
                       timeout=settings.request_timeout_s)


# The image model, newest first.
#
# gpt-image-1 is two generations old. 2.5 arrived in September 2026 in two
# flavours, and "sunburst" is the one built for edits where precision matters
# — which is this whole job: repaint the room, leave the doorway alone.
#
# It is a list rather than a name because a model id is the one thing that
# goes stale without warning, and an account that cannot see the newest should
# fall back rather than fail. Override the whole chain with OPENAI_IMAGE_MODEL.
IMAGE_MODELS = (
    "gpt-image-2.5-sunburst",
    "gpt-image-2",
    "gpt-image-1",
)


def image_models(settings: Settings) -> tuple[str, ...]:
    chosen = (settings.openai_image_model or "").strip()
    return (chosen,) if chosen else IMAGE_MODELS


def _no_such_model(exc: Exception) -> bool:
    text = str(exc).lower()
    return "model_not_found" in text or "does not exist" in text or (
        "404" in text and "model" in text)


def explain(exc: Exception) -> str:
    """Turn an OpenAI error into the sentence that says what to go and do.

    "Error code: 403" is what makes a working product look broken. Each of
    these has a different fix and none of them is obvious from the raw text.
    """
    text = str(exc)
    low = text.lower()

    if "must be verified" in low or ("403" in text and "verif" in low):
        return (
            "Your OpenAI organisation has not been verified for image models. "
            "This is a one-time ID check and it is separate from billing: go to "
            "platform.openai.com/settings/organization/general, click Verify "
            "Organization, then wait about 15 minutes for it to take effect. "
            "Reading rooms works without it, which is why only the pictures fail."
        )
    if "insufficient_quota" in low or "exceeded your current quota" in low:
        return (
            "That key's organisation has no credit on it. Credits belong to an "
            "organisation, not to a person — if the money was added to your "
            "client's account, the key has to be made inside their "
            "organisation too. Check which one you are in at "
            "platform.openai.com/settings/organization/billing."
        )
    if "invalid_api_key" in low or "incorrect api key" in low:
        return ("OpenAI does not recognise that key. Make a fresh one at "
                "platform.openai.com/api-keys and paste it again.")
    if "rate limit" in low or "429" in text:
        return "OpenAI is rate-limiting this key. Wait a minute and try again."
    if "content_policy" in low or "safety system" in low:
        return ("OpenAI's safety filter refused this photograph. Try a "
                "different picture of the room.")
    return f"Could not draw: {text}"


# ---------------------------------------------------------------------------
# Finding the structure
# ---------------------------------------------------------------------------

FIND = """\
Find every part of this photograph that a renovation must not move.

Return JSON only:
{"regions": [{"kind": "door", "box": [0.11, 0.20, 0.28, 0.86]}]}

kind is one of: door, window, walkway.
  door     — any doorway, door leaf, or opening through a wall to elsewhere
  window   — any window, including its frame and sill
  walkway  — the strip of bare floor people actually walk along to cross the
             room or reach a door. Not the whole floor.

box is [left, top, right, bottom] as fractions of the image, 0 to 1, from the
top-left. It must trace the object's own visible edges — where the door frame
or window frame actually starts and stops — plus a small margin for the
architrave, typically 1-3% of the frame on each side. It is not a zone, a
wall segment, or "the general area near" the object.

Get this wrong and the renovation is blocked from touching wall that has
nothing to do with the door: a floor-to-ceiling window that fills the back
of the room is still just that window, not the whole wall either side of it.
If in doubt, draw the box tighter, not looser — a door with 2% too little
margin still gets protected; a box that swallows half the room protects
nothing precisely and blocks a normal redesign.

One box per object. Two windows side by side are two boxes, not one box
spanning both.

Return an empty list if there are none. Do not invent any.
"""
def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def boxes_to_masks(regions: list[dict], size: tuple[int, int]) -> list[Mask]:
    """Rasterise fractional boxes into masks in the image's own pixels."""
    width, height = size
    masks: list[Mask] = []

    for index, region in enumerate(regions):
        box = region.get("box") or []
        if len(box) != 4:
            continue
        try:
            left, top, right, bottom = (_clamp01(v) for v in box)
        except (TypeError, ValueError):
            continue
        if right <= left or bottom <= top:
            continue

        x0, x1 = int(left * width), int(round(right * width))
        y0, y1 = int(top * height), int(round(bottom * height))
        # A model that rounds a thin window to zero width should not silently
        # contribute nothing; give it at least one pixel to be dilated from.
        x1, y1 = max(x1, x0 + 1), max(y1, y0 + 1)

        array = np.zeros((height, width), dtype=bool)
        array[y0:y1, x0:x1] = True
        kind = str(region.get("kind") or "other").strip().lower()
        masks.append(Mask(mask_id=f"{kind}_{index:03d}", array=array))

    return masks


async def find_structure(image: Image.Image, settings: Settings) -> list[dict]:
    """Ask what must not move, in fractions of the frame."""
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=88)
    data_uri = "data:image/jpeg;base64," + base64.b64encode(
        buffer.getvalue()).decode("ascii")

    try:
        response = await _client(settings).chat.completions.create(
            model=settings.vlm_model,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": FIND},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ]}],
            response_format={"type": "json_object"},
            max_tokens=900,
            temperature=0.0,      # geometry, not imagination
        )
    except Exception as exc:
        raise OpenAIImageError(f"Could not find the structure: {exc}") from exc

    try:
        parsed = json.loads(response.choices[0].message.content or "{}")
    except json.JSONDecodeError as exc:
        raise OpenAIImageError("Structure came back unreadable") from exc

    regions = parsed.get("regions")
    return regions if isinstance(regions, list) else []


# ---------------------------------------------------------------------------
# Repainting what is left
# ---------------------------------------------------------------------------

def to_openai_mask(inpaint_mask: Image.Image, *, inverted: bool = False) -> bytes:
    """Convert our mask to the one OpenAI's edit endpoint expects.

    Ours: white repaints, black preserves.
    Theirs: *transparent* repaints, opaque preserves.

    Getting this backwards does not raise anything — it repaints exactly the
    door and carefully preserves the sofa — so it lives in one function with a
    test on it rather than inline at the call site.
    """
    grey = inpaint_mask.convert("L")
    if inverted:
        grey = Image.eval(grey, lambda v: 255 - v)

    # Alpha is the inverse of "repaint here".
    alpha = Image.eval(grey, lambda v: 255 - v)
    out = Image.new("RGBA", grey.size, (0, 0, 0, 0))
    out.putalpha(alpha)

    buffer = io.BytesIO()
    out.save(buffer, format="PNG")
    return buffer.getvalue()


def _edit_size(size: tuple[int, int]) -> str:
    """The output size to ask gpt-image-2.5 for, given the room's own.

    gpt-image-2.5 takes an arbitrary WIDTHxHEIGHT as long as each side is a
    multiple of 16, the aspect ratio is between 1:3 and 3:1, no edge is over
    3840, and the total is 0.66-8.3 MP. Matching the room's real proportions
    (instead of snapping to one of three fixed shapes, as gpt-image-1 did)
    keeps the returned image aligned with the mask that was built from it.
    """
    width, height = size
    ratio = width / max(height, 1)

    # Pull the aspect ratio back inside 1:3..3:1 if the photo is extreme.
    if ratio > 3.0:
        width = height * 3
    elif ratio < 1 / 3:
        height = width * 3

    # Scale so the longest edge is 1536 — plenty of detail, comfortably inside
    # every ceiling and cheaper than going larger.
    longest = max(width, height)
    if longest != 1536:
        scale = 1536 / longest
        width, height = width * scale, height * scale

    def snap16(value: float) -> int:
        return max(16, int(round(value / 16)) * 16)

    return f"{snap16(width)}x{snap16(height)}"


# gpt-image-2.5 takes no seed, so options have to differ by instruction alone
# — and a vague mood word ("warmer") is not enough difference for it to act
# on: left with room to interpret, it tends to converge on the same obvious
# reading of a style. The first slot used to be "" (no instruction at all),
# which meant one option out of every batch was never told to differ from
# anything. Every slot now names something concrete and physical — a colour,
# a material, a silhouette — because that is what a model this literal
# actually diverges on.
VARIATIONS = (
    " Make one confident, specific choice — one accent colour, one material, "
    "one statement piece — and carry it through the whole room rather than "
    "scattering small variations.",
    " Take the quiet version: a tight neutral palette, natural materials, "
    "nothing glossy or saturated, more empty floor than furniture.",
    " Change the furniture's actual silhouettes, not just their colours — a "
    "different shape of bed frame, desk or seating — while keeping the same "
    "style and the same footprint in the room.",
    " Lead with a different dominant material than a plain reading would: "
    "stone, metal or lacquer where wood would be the obvious choice, or the "
    "reverse.",
)


def looks_inverted(original: Image.Image, drawn: bytes,
                   inpaint_mask: Image.Image) -> bool:
    """Did the parts that were supposed to be protected change the most?

    The mask convention here is the one thing this code cannot verify from the
    outside: the documentation for these models describes both "white areas are
    replaced" and "the alpha channel decides", and getting it backwards raises
    nothing at all. It repaints precisely the doorway and lovingly preserves
    the sofa.

    So it is measured instead. If the locked region moved substantially more
    than the editable one, the mask went in the wrong way round, and the fix
    is one environment variable — which is worth saying out loud rather than
    leaving somebody to wonder why their door became a window.
    """
    try:
        after = Image.open(io.BytesIO(drawn)).convert("RGB").resize(
            original.size, Image.BILINEAR)
    except Exception:
        return False

    before = np.asarray(original.convert("RGB"), dtype=np.int16)
    changed = np.abs(np.asarray(after, dtype=np.int16) - before).mean(axis=2)
    editable = np.asarray(inpaint_mask.convert("L")) > 127

    if editable.all() or not editable.any():
        return False        # nothing was locked, so nothing to get backwards

    locked_moved = float(changed[~editable].mean())
    editable_moved = float(changed[editable].mean())
    # Twice as much, and not merely noise. Diffusion bleeds a little over any
    # boundary, so a small difference is expected and is not this.
    return locked_moved > 12.0 and locked_moved > editable_moved * 2.0


async def redraw(image: Image.Image, inpaint_mask: Image.Image | None,
                 prompt: str, settings: Settings,
                 references: list[bytes] | None = None,
                 quality: str = "medium") -> bytes:
    """Redraw the room.

    With no mask — the default, and what ChatGPT does — the model edits the
    whole photograph with the original in front of it, and keeps what is in it.
    With a mask it repaints everything the mask leaves open, which on a strong
    editing model means erasing the furniture and inventing new furniture in
    its place. That is not what anybody wanted.

    `quality` is the caller's job to resolve from a tier (see
    config.tier_of) — this function has no notion of accounts or plans, only
    of the one dial OpenAI actually bills by.
    """
    photo = io.BytesIO()
    image.convert("RGB").save(photo, format="PNG")
    photo.seek(0)

    # Other views of the same room. The endpoint takes up to sixteen, and they
    # are what stop a full redesign inventing the wall the camera could not
    # see. One view is sent on its own rather than as a list of one, because a
    # light restyle needs no second view and the shape should not change
    # under it.
    views = [("room.png", photo, "image/png")]
    for index, extra in enumerate(references or [], start=1):
        views.append((f"view{index}.png", io.BytesIO(extra), "image/png"))

    mask = None
    if inpaint_mask is not None:
        mask = io.BytesIO(to_openai_mask(inpaint_mask,
                                         inverted=settings.invert_inpaint_mask))

    client = _client(settings)
    last: Exception | None = None

    for model in image_models(settings):
        for _name, stream, _type in views:
            stream.seek(0)
        call = {
            "model": model,
            "image": views if len(views) > 1 else views[0],
            "prompt": prompt[:4000],
            "size": _edit_size(image.size),
            # Never left to the default. The default is the expensive end.
            "quality": quality,
            "n": 1,
        }
        if mask is not None:
            mask.seek(0)
            call["mask"] = ("mask.png", mask, "image/png")
        try:
            response = await client.images.edit(**call)
        except Exception as exc:
            last = exc
            if _no_such_model(exc):
                # This account cannot see this model. Try the one below it
                # rather than failing on a name.
                log.info("Image model %s unavailable, falling back", model)
                continue
            raise OpenAIImageError(explain(exc)) from exc

        if not response.data or not getattr(response.data[0], "b64_json", None):
            raise OpenAIImageError("No picture came back.")
        log.info("Drew with %s", model)
        return base64.b64decode(response.data[0].b64_json)

    raise OpenAIImageError(
        "None of the image models are available on this key: "
        + ", ".join(image_models(settings))
        + (f" ({last})" if last else "")
    )
