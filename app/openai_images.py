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


async def redraw(image: Image.Image, inpaint_mask: Image.Image, prompt: str,
                 settings: Settings) -> bytes:
    """Repaint only the unmasked part of the room."""
    photo = io.BytesIO()
    image.convert("RGB").save(photo, format="PNG")
    photo.seek(0)

    mask = io.BytesIO(to_openai_mask(inpaint_mask,
                                     inverted=settings.invert_inpaint_mask))
    mask.seek(0)

    client = _client(settings)
    last: Exception | None = None

    for model in image_models(settings):
        photo.seek(0)
        mask.seek(0)
        try:
            response = await client.images.edit(
                model=model,
                image=("room.png", photo, "image/png"),
                mask=("mask.png", mask, "image/png"),
                prompt=prompt[:4000],
                size=_closest_size(image.size),
                n=1,
            )
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
