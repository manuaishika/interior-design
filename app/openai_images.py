"""The whole job on one OpenAI key, with the mask still enforced.

Why this exists
---------------
There were two engines and each wanted something the other did not.

The free one needs no card and holds nothing: it *asks* the model to leave the
door alone. The full one holds everything but needs two paid accounts — OpenAI
to read the room, Replicate to segment and repaint it. So somebody with one
OpenAI key could read a room and then not get a picture, which is the worst
place to be after paying for something.

This closes that gap. OpenAI's image edit endpoint takes a mask, so the lock
can be real without Replicate existing at all:

    GPT-4o          -> where the doors, windows and walkways are
    a rasterised mask -> those regions, painted out
    gpt-image-1     -> repaints only what is left

What it gives up against SAM2
-----------------------------
Boxes, not outlines. SAM2 returns the actual silhouette of a door; this returns
a rectangle around it, so a little wall near the frame is protected too. That
costs some editable area around openings and protects slightly more than it
needs to — which is the right direction to be wrong. Destroying a doorway is
the failure that matters; under-editing the wall beside it is not.

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
top-left. Be generous: include the frame and architrave, not just the glass.

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


def _closest_size(size: tuple[int, int]) -> str:
    """gpt-image-1 takes three shapes. Pick the one nearest the room's own."""
    width, height = size
    ratio = width / height if height else 1.0
    if ratio > 1.2:
        return "1536x1024"
    if ratio < 0.83:
        return "1024x1536"
    return "1024x1024"


# gpt-image-1 takes no seed, so options have to differ by instruction.
VARIATIONS = (
    "",
    " Take a warmer, softer reading of this, with more textile and more "
    "layered lighting.",
    " Take a cooler, more pared-back reading, with fewer pieces and more "
    "empty floor.",
    " Take a bolder reading, with one strong colour and one sculptural piece "
    "as the focus.",
)


async def redraw(image: Image.Image, inpaint_mask: Image.Image, prompt: str,
                 settings: Settings) -> bytes:
    """Repaint only the unmasked part of the room."""
    photo = io.BytesIO()
    image.convert("RGB").save(photo, format="PNG")
    photo.seek(0)

    mask = io.BytesIO(to_openai_mask(inpaint_mask,
                                     inverted=settings.invert_inpaint_mask))
    mask.seek(0)

    try:
        response = await _client(settings).images.edit(
            model=settings.openai_image_model,
            image=("room.png", photo, "image/png"),
            mask=("mask.png", mask, "image/png"),
            prompt=prompt[:4000],
            size=_closest_size(image.size),
            n=1,
        )
    except Exception as exc:
        raise OpenAIImageError(f"Could not draw: {exc}") from exc

    if not response.data or not getattr(response.data[0], "b64_json", None):
        raise OpenAIImageError("No picture came back.")
    return base64.b64decode(response.data[0].b64_json)
