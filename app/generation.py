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


STYLES: dict[str, str] = {
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


def build_prompt(
    style: str, extra: str = "", contents: str = "", keep: str = ""
) -> str:
    """Compose the generation prompt.

    `contents` and `keep` come from the analysis — what is actually in the room
    and what there is more than one of. Without them the model draws an average
    room of that type, which is how two single beds come back as one double.
    """
    base = STYLES.get(style.strip().lower(), style.strip())
    prompt = f"Interior design photograph of this room restyled in {base}."
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
