"""Step 1b — label each SAM2 mask with a vision-language model.

Each region gets two views: a dimmed full-frame shot showing where it sits, and
a padded crop showing what it looks like. The crop alone is genuinely ambiguous
(a flat beige patch is a wall, a floor, or a ceiling depending only on where it
is), so the context view carries most of the signal.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass

from openai import AsyncOpenAI
from PIL import Image

from .config import CATEGORIES, Settings
from .imaging import Mask, crop_with_context, highlight_region, image_to_data_uri
from .models import BoundingBox

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You label segmented regions of an interior room photograph.

You get two images of the SAME region:
1. CONTEXT — the full room, with the region in full colour, everything else
   dimmed, and a red box around it.
2. CROP — a close-up of that region.

Classify the region into exactly one category:
- "furniture"  a movable furnishing or decor object (sofa, chair, table, bed,
               rug, lamp, plant, artwork, cushion, shelf, cabinet)
- "door"       a doorway, door leaf, or door frame
- "window"     a window, its frame, or its glazing
- "wall"       a flat vertical wall surface
- "floor"      a floor surface with nothing standing on it
- "walkway"    open circulation space people walk through: the path between
               furniture, in front of a door, or through the middle of a room
- "other"      none of the above, or too ambiguous to call (ceiling, clutter,
               a segmentation artefact, a region spanning several things)

Rules:
- Judge the region inside the red box, not the whole room.
- For "furniture", set `name` to the specific item ("sofa", "coffee table",
  "floor lamp"). For every other category, set `name` to the category itself.
- Prefer "walkway" over "floor" when the area reads as a route between or
  around furniture rather than incidental floor.
- Use "other" rather than guessing. A wrong "floor" on a doorway is worse than
  an honest "other".
- `confidence` is 0.0-1.0 and should reflect real uncertainty.
"""

RESPONSE_SCHEMA = {
    "name": "region_label",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "category": {"type": "string", "enum": list(CATEGORIES)},
            "name": {"type": "string"},
            "confidence": {"type": "number"},
            "notes": {"type": "string"},
        },
        "required": ["category", "name", "confidence", "notes"],
        "additionalProperties": False,
    },
}


@dataclass
class RegionLabel:
    mask_id: str
    category: str
    name: str
    confidence: float
    notes: str = ""

    @classmethod
    def unknown(cls, mask_id: str, reason: str) -> "RegionLabel":
        """Fallback when labeling fails.

        Deliberately "other" with zero confidence: config.LOCK_POLICY locks
        "other", so a region we failed to identify is preserved rather than
        regenerated.
        """
        return cls(
            mask_id=mask_id,
            category="other",
            name="unidentified region",
            confidence=0.0,
            notes=reason,
        )


def _clean(raw: dict, mask_id: str) -> RegionLabel:
    category = str(raw.get("category", "other")).strip().lower()
    if category not in CATEGORIES:
        log.warning("Mask %s: model returned category %r", mask_id, category)
        category = "other"

    name = str(raw.get("name") or "").strip() or category
    if category != "furniture":
        # Only furniture carries a distinct item name; keep the rest canonical
        # so downstream label comparisons stay simple.
        name = category

    try:
        confidence = float(raw.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0

    return RegionLabel(
        mask_id=mask_id,
        category=category,
        name=name,
        confidence=min(max(confidence, 0.0), 1.0),
        notes=str(raw.get("notes") or "")[:500],
    )


async def _label_one(
    client: AsyncOpenAI,
    image: Image.Image,
    mask: Mask,
    box: BoundingBox,
    settings: Settings,
    semaphore: asyncio.Semaphore,
) -> RegionLabel:
    context = highlight_region(image, mask.array, box)
    crop = crop_with_context(image, box, settings.crop_padding_frac)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "CONTEXT — region highlighted in the full room:"},
                {"type": "image_url", "image_url": {"url": image_to_data_uri(context)}},
                {"type": "text", "text": "CROP — close-up of the same region:"},
                {"type": "image_url", "image_url": {"url": image_to_data_uri(crop)}},
                {
                    "type": "text",
                    "text": (
                        f"The region covers {mask.area / mask.array.size:.1%} of the "
                        "image. Classify it."
                    ),
                },
            ],
        },
    ]

    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=settings.vlm_model,
                messages=messages,
                response_format={"type": "json_schema", "json_schema": RESPONSE_SCHEMA},
                max_tokens=300,
                temperature=0,
            )
        except Exception as exc:
            log.warning("Labeling failed for %s: %s", mask.mask_id, exc)
            return RegionLabel.unknown(mask.mask_id, f"labeling failed: {exc}")

    content = (response.choices[0].message.content or "").strip()
    try:
        return _clean(json.loads(content), mask.mask_id)
    except (json.JSONDecodeError, AttributeError) as exc:
        log.warning("Unparseable label for %s: %s", mask.mask_id, exc)
        return RegionLabel.unknown(mask.mask_id, "model returned unparseable JSON")


async def label_masks(
    image: Image.Image,
    masks: list[Mask],
    boxes: dict[str, BoundingBox],
    settings: Settings,
) -> list[RegionLabel]:
    """Label every mask concurrently, bounded by `label_concurrency`.

    One mask failing never fails the batch — it comes back as an "other" region,
    which the lock policy treats as locked.
    """
    if not masks:
        return []
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    client = AsyncOpenAI(
        api_key=settings.openai_api_key, timeout=settings.request_timeout_s
    )
    semaphore = asyncio.Semaphore(settings.label_concurrency)

    log.info("Labeling %d masks with %s", len(masks), settings.vlm_model)
    return list(
        await asyncio.gather(
            *(
                _label_one(client, image, mask, boxes[mask.mask_id], settings, semaphore)
                for mask in masks
            )
        )
    )
