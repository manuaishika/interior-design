"""Step 1a — segmentation masks from Meta's SAM2, hosted on Replicate.

Uses SAM2's automatic mask-generation mode: no prompts, no click points, just
"segment everything you can find in this room photo".
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx
import replicate
from PIL import Image

from .config import Settings
from .imaging import Mask, decode_mask, image_to_data_uri

log = logging.getLogger(__name__)


class SegmentationError(RuntimeError):
    pass


def _extract_mask_refs(output: Any) -> list[Any]:
    """Pull the per-mask entries out of whatever shape Replicate hands back.

    Replicate model output schemas drift between versions, and the client
    returns bare URL strings on older versions and `FileOutput` objects on
    newer ones. Rather than pin one shape, probe the plausible ones and fail
    loudly with the payload if none match.
    """
    if output is None:
        raise SegmentationError("SAM2 returned no output")

    if isinstance(output, dict):
        for key in ("individual_masks", "masks", "individual_mask", "segmentations"):
            value = output.get(key)
            if isinstance(value, list) and value:
                return value
        # A model variant that returns only a single combined mask is not
        # usable here — we need per-region masks to label them individually.
        if "combined_mask" in output:
            raise SegmentationError(
                "SAM2 returned only a combined mask; this pipeline needs "
                "per-region masks. Check that the model is running in "
                "automatic mask-generation mode."
            )
        raise SegmentationError(f"Unrecognised SAM2 output keys: {list(output)}")

    if isinstance(output, list):
        return output

    return [output]


async def _fetch_mask_bytes(client: httpx.AsyncClient, ref: Any) -> bytes:
    """Read one mask's bytes from a URL string, FileOutput, or raw bytes."""
    if isinstance(ref, (bytes, bytearray)):
        return bytes(ref)

    # replicate>=0.26 FileOutput: has .read() and .url
    read = getattr(ref, "read", None)
    if callable(read):
        data = read()
        if asyncio.iscoroutine(data):
            data = await data
        if isinstance(data, (bytes, bytearray)):
            return bytes(data)

    url = getattr(ref, "url", None) or (ref if isinstance(ref, str) else None)
    if not url:
        raise SegmentationError(f"Cannot read mask from {type(ref).__name__}")

    response = await client.get(url)
    response.raise_for_status()
    return response.content


async def segment_room(image: Image.Image, settings: Settings) -> list[Mask]:
    """Run SAM2 and return every mask it found, aligned to `image`.

    Filtering happens downstream in `imaging.filter_masks` — this function's job
    is just to get the raw masks back.
    """
    if not settings.replicate_api_token:
        raise SegmentationError("REPLICATE_API_TOKEN is not set")

    payload = {
        "image": image_to_data_uri(image),
        "points_per_side": settings.sam2_points_per_side,
        "pred_iou_thresh": settings.sam2_pred_iou_thresh,
        "stability_score_thresh": settings.sam2_stability_score_thresh,
    }

    log.info("Running SAM2 (%s) on %sx%s image", settings.sam2_model, *image.size)
    try:
        output = await asyncio.wait_for(
            replicate.async_run(settings.sam2_model, input=payload),
            timeout=settings.request_timeout_s,
        )
    except asyncio.TimeoutError as exc:
        raise SegmentationError(
            f"SAM2 timed out after {settings.request_timeout_s}s"
        ) from exc
    except SegmentationError:
        raise
    except Exception as exc:  # replicate raises a variety of error types
        raise SegmentationError(f"SAM2 call failed: {exc}") from exc

    refs = _extract_mask_refs(output)
    log.info("SAM2 returned %d masks", len(refs))

    async with httpx.AsyncClient(timeout=settings.request_timeout_s) as client:
        blobs = await asyncio.gather(
            *(_fetch_mask_bytes(client, ref) for ref in refs),
            return_exceptions=True,
        )

    masks: list[Mask] = []
    for index, blob in enumerate(blobs):
        if isinstance(blob, BaseException):
            log.warning("Skipping mask %d: %s", index, blob)
            continue
        try:
            masks.append(decode_mask(blob, image.size, mask_id=f"mask_{index:03d}"))
        except Exception as exc:
            log.warning("Skipping undecodable mask %d: %s", index, exc)

    if not masks:
        raise SegmentationError("SAM2 produced no usable masks")
    return masks
