"""Step 2 — mask-constrained image generation.

The locked regions from the analysis become an inpainting mask, so the model
only repaints unlocked areas and leaves doors, windows and walkways alone.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from typing import Any

import httpx
import replicate
from PIL import Image

from .config import Settings
from .imaging import image_to_base64, image_to_data_uri

log = logging.getLogger(__name__)


class GenerationError(RuntimeError):
    pass


# Two of these are not looks at all. "none" is for someone who wants the room
# fixed rather than restyled, and "brief" is for someone who would rather
# describe what they want than pick a label off a list — the commonest thing a
# real client does. Both live here rather than as a special case in the caller,
# so every path that resolves a style resolves these too.
STYLES: dict[str, str] = {
    "none": (
        "a clean, well-resolved contemporary interior that follows the room's "
        "own architecture rather than any particular decorating style"
    ),
    "brief": (
        "the interior described in the instructions that follow, and no other "
        "decorating style"
    ),
    "scandinavian": (
        "Scandinavian interior, pale oak, soft white walls, linen textiles, "
        "minimal uncluttered furniture, abundant natural light"
    ),
    "mid-century-modern": (
        "mid-century modern interior, walnut furniture with tapered legs, "
        "muted olive and mustard accents, clean low-profile silhouettes"
    ),
    "industrial": (
        "industrial loft interior, exposed brick, blackened steel, reclaimed "
        "wood, leather seating, factory pendant lighting"
    ),
    "japandi": (
        "Japandi interior, low natural wood furniture, neutral earth palette, "
        "paper lantern lighting, calm negative space, wabi-sabi ceramics"
    ),
    "bohemian": (
        "bohemian interior, layered patterned rugs, rattan and macrame, "
        "abundant houseplants, warm terracotta and ochre palette"
    ),
    "modern-luxury": (
        "modern luxury interior, marble and brass accents, deep velvet "
        "upholstery, sculptural lighting, refined neutral palette"
    ),
}

NEGATIVE_PROMPT = (
    "clutter, mess, laundry, clothes on the bed, bags, boxes, "
    "blurry, distorted geometry, warped walls, extra doors, extra windows, "
    "blocked doorway, merged furniture, furniture floating, "
    "low quality, watermark, text"
)


# A look and a purpose are different axes, and conflating them is what makes
# "Nursery" feel like a missing style. It is not one: a nursery can perfectly
# well be Japandi. The look supplies the palette and the materials; the room
# supplies what has to be in it and what must not be.
#
# Kept apart, six looks cover sixteen rooms. Merged, you would need ninety-six
# and a client would have to choose between the aesthetic they want and the
# room they actually have.
ROOM_BRIEFS: dict[str, str] = {
    "bedroom": "a bedroom, with a made bed, light to read by, and clothes put away",
    "living": "a living room, with seating that faces itself, a surface within "
              "reach of every seat, and lighting at more than one height",
    "dining": "a dining room, with a table, chairs that can pull fully out, and "
              "a light centred over the table",
    "kitchen": "a kitchen, with an unbroken run of worktop, storage above and "
               "below it, and light falling on the counter rather than behind you",
    "study": "a study, with a desk at working height, a chair that supports a "
             "back, task lighting, and shelving within arm's reach",
    "bathroom": "a bathroom, with sanitaryware, a mirror lit on the face rather "
                "than from above, and surfaces that tolerate water",
    "hallway": "a hallway, kept clear enough to walk through carrying things, "
               "with somewhere for coats and shoes and a mirror",
    "stairway": "a stairway or landing, with the treads and handrail completely "
                "unobstructed, light at the top and the bottom, and nothing "
                "stored on the steps",
    # The one that shows why this list exists. None of these requirements are
    # aesthetic, and no decorating style implies a single one of them.
    "nursery": "a nursery, with a cot holding nothing but a flat firm mattress, "
               "low storage a carer can reach one-handed while holding a child, "
               "a soft floor to kneel on, a blackout blind, and a chair to feed "
               "in. Nothing heavy, breakable or shelved above the cot. No blind "
               "cords, cables or anything with a loop within reach of it",
    "utility": "a utility room, with appliances plumbed in, somewhere to dry and "
               "fold, and hard-wearing surfaces",
    "balcony": "a balcony or terrace, with weatherproof furniture, planting, and "
               "the railing left intact and impossible to climb",
    "gym": "a home gym, with clear floor to move in, flooring that absorbs "
           "impact, a mirror, and equipment stored back against the walls",
    "studio-flat": "a studio flat, with sleeping, sitting, eating and storage "
                   "each given their own zone without blocking the route through",
    "open-plan": "an open-plan living and kitchen, the cooking and sitting zones "
                 "distinct but continuous",
    "retail": "a shop, café or salon, with a clear route for customers, a "
              "service point, display or seating, and commercial-grade finishes",
}


def room_brief(room: str) -> str:
    """What this kind of room has to be, regardless of how it is decorated."""
    return ROOM_BRIEFS.get((room or "").strip().lower(), "")


def build_prompt(
    style: str, extra: str = "", contents: str = "", keep: str = "",
    room: str = "",
) -> str:
    """Compose the generation prompt.

    `contents` and `keep` come from the analysis — what is actually in the room
    and what there is more than one of. Without them the model draws an average
    room of that type, which is how two single beds come back as one double.

    `room` says what the room is *for*. It is separate from the style on
    purpose: a nursery is not a look, it is a set of requirements, and it needs
    to be able to be a Japandi one.
    """
    key = style.strip().lower()
    base = STYLES.get(key, style.strip())
    verb = "photographed as" if key in ("none", "brief") else "restyled in"
    prompt = f"Interior design photograph of this room {verb} {base}."

    # Before the contents, because it governs what the contents should become.
    brief = room_brief(room)
    if brief:
        prompt += f" It must work as {brief}."

    if contents.strip():
        prompt += f" The room contains {contents.strip()}."
    if keep.strip():
        prompt += f" {keep.strip()}, in their existing positions."
    prompt += (
        " Tidy and uncluttered. Photorealistic, architectural photography, "
        "natural lighting, consistent perspective and proportions with the "
        "original room."
    )
    if extra.strip():
        prompt = f"{prompt} {extra.strip()}"
    return prompt


def _extract_image_ref(output: Any) -> Any:
    """Normalise Replicate's output to a single image reference."""
    if output is None:
        raise GenerationError("Inpainting model returned no output")
    if isinstance(output, dict):
        for key in ("image", "output", "images"):
            if key in output:
                return _extract_image_ref(output[key])
        raise GenerationError(f"Unrecognised generation output keys: {list(output)}")
    if isinstance(output, list):
        if not output:
            raise GenerationError("Inpainting model returned an empty list")
        return output[0]
    return output


async def _read_image(ref: Any, timeout: float) -> tuple[bytes, str | None]:
    """Return (png bytes, source url) for a URL string, FileOutput, or bytes."""
    if isinstance(ref, (bytes, bytearray)):
        return bytes(ref), None

    url = getattr(ref, "url", None) or (ref if isinstance(ref, str) else None)

    read = getattr(ref, "read", None)
    if callable(read):
        data = read()
        if asyncio.iscoroutine(data):
            data = await data
        if isinstance(data, (bytes, bytearray)):
            return bytes(data), url

    if not url:
        raise GenerationError(f"Cannot read image from {type(ref).__name__}")

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.content, url


async def generate_with_mask(
    image: Image.Image,
    inpaint_mask: Image.Image,
    prompt: str,
    settings: Settings,
    seed: int | None = None,
) -> tuple[str, str | None]:
    """Run mask-conditioned generation. Returns (base64 png, source url).

    `inpaint_mask` must already be in the provider's convention — see
    `imaging.build_inpaint_mask` and `Settings.invert_inpaint_mask`.
    """
    if not settings.replicate_api_token:
        raise GenerationError("REPLICATE_API_TOKEN is not set")
    if image.size != inpaint_mask.size:
        raise GenerationError(
            f"image size {image.size} != mask size {inpaint_mask.size}"
        )

    payload: dict[str, Any] = {
        "prompt": prompt,
        "negative_prompt": NEGATIVE_PROMPT,
        "image": image_to_data_uri(image),
        "mask": image_to_data_uri(inpaint_mask),
        "num_inference_steps": settings.generation_steps,
        "guidance_scale": settings.generation_guidance,
    }
    if seed is not None:
        payload["seed"] = seed

    log.info("Running inpainting (%s)", settings.inpaint_model)
    try:
        output = await asyncio.wait_for(
            replicate.async_run(settings.inpaint_model, input=payload),
            timeout=settings.request_timeout_s,
        )
    except asyncio.TimeoutError as exc:
        raise GenerationError(
            f"Generation timed out after {settings.request_timeout_s}s"
        ) from exc
    except GenerationError:
        raise
    except Exception as exc:
        raise GenerationError(f"Generation call failed: {exc}") from exc

    data, url = await _read_image(
        _extract_image_ref(output), settings.request_timeout_s
    )
    return base64.b64encode(data).decode("ascii"), url


def encode_mask(mask: Image.Image) -> str:
    return image_to_base64(mask)
