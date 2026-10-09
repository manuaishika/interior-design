"""Changing one thing in a finished design without redrawing the rest.

Image models do not edit, they redraw: given a design and "turn the bed to
face the window" they hand back a whole new picture in which the bed is
turned, the curtains are a different cloth, the rug has a new pattern and the
bed itself is quietly a different bed. A mask does not stop that — for these
models it is a hint, not a boundary.

So an edit here is three things:

1. **Where.** A marked area is used as drawn, with any closed circle filled
   in (people circle the lamp; the ink is a ring). A typed change is located
   first: a vision model looks at the design and says which part of the
   picture the change is about — the bed, plus the floor it will occupy once
   turned — or that it is about the whole room ("warmer light everywhere").
2. **What.** The prompt says to make that one change and nothing else, and
   that a turned or moved piece is the same piece: same design, same colour,
   same material.
3. **Proof.** For anything local, the original picture is pasted back
   everywhere outside the area, so nothing outside it can change at all.
"""

from __future__ import annotations

import io
import logging

from PIL import Image

from . import planning
from .config import Settings
from .imaging import (coverage, fill_enclosed, grow, keep_outside,
                      mask_from_boxes)

log = logging.getLogger(__name__)

LOCATE = """\
The attached picture is a finished interior design. A client asked for this
change to it: "{instruction}"

Decide where in the picture the change happens. Return JSON only:
{{"scope": "local", "boxes": [[0.1, 0.4, 0.5, 0.9]], "subject": "the bed"}}

scope is "local" when the change is about particular pieces or a particular
part of the room (a lamp, the bed, the wall behind the sofa, a rug), and
"whole" only when it is about the entire picture at once (all the lighting,
the overall colour scheme, the whole style).

boxes are [left, top, right, bottom] as fractions of the picture, 0 to 1,
from the top-left: one box per piece or area that changes. If something is
to be rotated, turned, moved or resized, the box must cover BOTH where it is
now AND where it will be afterwards, plus a little floor around it for its
shadow. If something is added, the box covers where it will go. Keep boxes as
tight as that allows — everything outside them will be kept exactly.
subject names what is being changed, in a few words.
"""

EDIT = """\
This picture is a finished interior design. Make exactly one change to it:
{instruction}

Everything else stays exactly as it is in this picture. Every piece of
furniture keeps its design, colour, material, size and position. The walls,
floor, ceiling, lighting, curtains, decor and the view are unchanged, and so
is the camera: same angle, same crop.

If the change is to rotate, turn, flip, move or reposition something, it is
the SAME piece: identical design, colour, fabric, material and size. Only its
orientation or position changes. Do not restyle it, upgrade it or swap it for
a similar one. Where it used to stand, continue the floor and wall that are
already around it.
{area}
Never add, remove or move a door, window or other fixed part of the room.
Match the light, shadows, perspective and photographic finish of the picture
so the change is seamless.
"""

AREA = ("\nOnly the marked area may change. Blend it into what surrounds it, "
        "with no visible seam.\n")


async def locate(design: bytes, instruction: str,
                 settings: Settings) -> list[list[float]] | None:
    """Boxes for a local change, or None for a whole-picture one (and None if
    the locator is unavailable — the edit then falls back to the prompt)."""
    try:
        raw = await planning.ask_json(
            LOCATE.format(instruction=instruction.strip()[:500]), [design],
            settings, seed=11, max_tokens=300)
    except Exception as exc:          # noqa: BLE001 — falls back to a whole edit
        log.warning("Could not locate the edit (%s); editing the whole picture", exc)
        return None
    if not isinstance(raw, dict) or str(raw.get("scope", "")).lower() != "local":
        return None
    boxes = [b for b in (raw.get("boxes") or [])
             if isinstance(b, (list, tuple)) and len(b) == 4]
    return boxes or None


def area_for(image: Image.Image, *, drawn: Image.Image | None = None,
             boxes: list[list[float]] | None = None) -> Image.Image | None:
    """The area an edit may touch, or None for the whole picture.

    A drawn mark is filled and widened a little (a pen stroke is narrower than
    the thing it circles); located boxes are widened more (they are a guess).
    An area covering nearly everything is treated as the whole picture, and an
    empty one as no mark at all.
    """
    if drawn is not None:
        mask = drawn if drawn.size == image.size else drawn.resize(image.size)
        mask = grow(fill_enclosed(mask), 0.015)
    elif boxes:
        mask = grow(mask_from_boxes(image.size, boxes), 0.03)
    else:
        return None
    share = coverage(mask)
    if share < 0.0005 or share > 0.85:
        return None
    return mask


def finish(image: Image.Image, result: bytes, mask: Image.Image | None) -> bytes:
    """Paste the untouched original back outside the area."""
    if mask is None:
        return result
    edited = Image.open(io.BytesIO(result))
    out = io.BytesIO()
    keep_outside(image, edited, mask).save(out, format="PNG")
    return out.getvalue()
