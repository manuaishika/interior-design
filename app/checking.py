"""Looking at a finished design before anyone else does.

The drawing prompt asks for a lot — same room, same doors and windows, one bed
stays one bed, and (in a full redesign) every loose piece replaced — and an
image model keeps only some of it, some of the time. Until now nobody checked:
the first the client heard of a second bed was seeing it.

So each finished design is shown, next to the original photograph, to a vision
model that answers a short list of yes/no questions. If a serious rule was
broken it names the faults and the design is drawn once more with those faults
called out. One retry, not a loop: it bounds the cost and the wait, and a
second attempt that is told exactly what went wrong is right far more often
than a first that was told nothing.

The check is advisory in both directions: if it cannot run, the design ships as
it is; if it passes a design, that is a judgement, not a guarantee.
"""

from __future__ import annotations

import logging

from .config import Settings
from . import planning
from .generation import before_client

log = logging.getLogger(__name__)

JUDGE = """\
You are checking an AI-drawn interior redesign against the original photograph.
Image 1 is the ORIGINAL room. Image 2 is the REDESIGN drawn from it.
{expected}
Answer as JSON only:
{{"same_viewpoint": true,
  "structure_ok": true,
  "counts_ok": true,
  "unchanged_pieces": [""],
  "invented": [""],
  "problems": [""]}}

- same_viewpoint: is it the same room seen from the same spot, at about the same
  angle and crop? false if the camera moved, zoomed, or the room's proportions
  changed.
- structure_ok: every door and window is still there, in the same place and
  about the same size; nothing new (a door, a window, a doorway) has been added
  in a wall; the air conditioner and fixed lights have not moved. A built-in
  wardrobe must still be a wardrobe.
- counts_ok: the same number of beds, doors and windows as the original.
- unchanged_pieces: pieces of loose furniture or furnishings (bed, desk, chair,
  sofa, table, rug, curtains...) that look like the SAME piece as in the
  original, only cleaner or recoloured. Empty if each was genuinely replaced.
  Only list real, clearly visible ones.
- invented: things that should not be there: a second bed, a duplicated object,
  melted or distorted furniture, text, a watermark, more than two plants.
- problems: the faults to fix, as short imperative sentences ("Keep one bed
  only", "Put the window back on the left wall"). Empty if none.
Be strict about structure and counts, and fair about taste. Do not mark
something wrong just because you would have designed it differently.
"""


def _expected(depth: str, plan_text: str) -> str:
    if (depth or "").strip().lower() == "renovate":
        out = ("This was a FULL redesign: the loose furniture was meant to be "
               "replaced with different pieces.\n")
        if plan_text.strip():
            out += f"The plan it was meant to follow:\n{plan_text.strip()}\n"
        return out
    return ("This was a LIGHT restyle: the furniture was meant to stay, with new "
            "finishes, textiles and light. Do not list unchanged_pieces.\n")


def verdict(raw: dict | None, depth: str) -> list[str]:
    """The faults serious enough to redraw for, or [] for a pass.

    Serious means: the viewpoint moved, structure or counts broke, something
    was invented, or (full redesign only) half or more of the loose furniture
    visibly survived. Anything smaller is taste, and is left alone.
    """
    if not isinstance(raw, dict):
        return []
    faults: list[str] = []
    if raw.get("same_viewpoint") is False:
        faults.append("Keep the camera exactly where the original photograph "
                      "has it: same angle, same crop, same proportions.")
    if raw.get("structure_ok") is False:
        faults.append("Keep every door, window and fixed light exactly where "
                      "the original has them, and add no new openings.")
    if raw.get("counts_ok") is False:
        faults.append("Keep the same number of beds, doors and windows as the "
                      "original.")
    invented = [str(x).strip() for x in (raw.get("invented") or []) if str(x).strip()]
    if invented:
        faults.append("Remove: " + "; ".join(invented[:3]) + ".")

    if (depth or "").strip().lower() == "renovate":
        left = [str(x).strip() for x in (raw.get("unchanged_pieces") or [])
                if str(x).strip()]
        if len(left) >= 2:
            faults.append("These are still the old pieces and must be replaced "
                          "with genuinely different designs: "
                          + "; ".join(left[:4]) + ".")
    # Only keep specific, imperative sentences it wrote itself if a hard fault
    # is already there; on their own they are taste.
    if faults:
        extra = [str(x).strip() for x in (raw.get("problems") or []) if str(x).strip()]
        for line in extra[:2]:
            if line not in faults and len(faults) < 5:
                faults.append(line if line.endswith(".") else line + ".")
    return faults


async def check(original: bytes, result: bytes, *, depth: str, plan_text: str,
                settings: Settings) -> list[str]:
    """Faults in this design worth redrawing for. [] means ship it — and also
    means "could not check", by design."""
    if not settings.self_check:
        return []
    try:
        raw = await planning.ask_json(
            JUDGE.format(expected=_expected(depth, plan_text)),
            [original, result], settings, seed=7, max_tokens=700)
    except Exception as exc:        # noqa: BLE001 — advisory, see module docstring
        log.warning("Self-check unavailable (%s); shipping the design", exc)
        return []
    faults = verdict(raw, depth)
    if faults:
        log.info("Self-check found %d fault(s): %s", len(faults), faults)
    return faults


def with_faults(prompt: str, faults: list[str]) -> str:
    """The same prompt, with the faults called out before the client's words
    (which stay last)."""
    return before_client(
        prompt, "A previous attempt at this exact design had these faults. Do "
                "not repeat them:\n" + "\n".join(f"- {f}" for f in faults))
