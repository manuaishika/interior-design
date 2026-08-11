"""Orchestration: upload -> analyze -> (style) -> masked generation.

The analysis step slots in between upload and generation; the surrounding flow
is unchanged.
"""

from __future__ import annotations

import logging

from PIL import Image

from .config import Settings, is_locked
from .generation import build_prompt, encode_mask, generate_with_mask
from .imaging import (
    Mask,
    build_inpaint_mask,
    editable_fraction,
    fit_to_max_edge,
    filter_masks,
    load_image,
    mask_bounding_box,
    snap_to_multiple,
)
from .labeling import label_masks
from .models import GenerationResult, RoomAnalysis, RoomObject
from .segmentation import segment_room

log = logging.getLogger(__name__)


def prepare_image(data: bytes, settings: Settings) -> Image.Image:
    """Decode and normalise the upload.

    Every mask, bounding box and generated pixel is expressed in *this* image's
    coordinate space, so normalisation happens exactly once, up front.
    """
    image = load_image(data)
    image = fit_to_max_edge(image, settings.max_image_edge)
    return snap_to_multiple(image, 8)


async def analyze_room(
    image: Image.Image,
    settings: Settings,
    *,
    keep_mask_ids: set[str] | None = None,
    replace_mask_ids: set[str] | None = None,
) -> tuple[RoomAnalysis, dict[str, Mask]]:
    """Steps 1a-1c: segment, label, and assemble the structured JSON.

    `keep_mask_ids` / `replace_mask_ids` are the user's keep-vs-replace calls,
    applied as overrides on top of the category lock policy. Deciding that
    automatically is explicitly out of scope; this is just the input path.
    """
    keep_mask_ids = keep_mask_ids or set()
    replace_mask_ids = replace_mask_ids or set()
    overlap = keep_mask_ids & replace_mask_ids
    if overlap:
        raise ValueError(
            f"mask ids appear in both keep and replace: {sorted(overlap)}"
        )

    width, height = image.size
    image_area = width * height

    # 1a — SAM2
    raw_masks = await segment_room(image, settings)

    kept = filter_masks(
        raw_masks,
        image_area=image_area,
        min_area_frac=settings.min_mask_area_frac,
        max_area_frac=settings.max_mask_area_frac,
        dedupe_iou_thresh=settings.dedupe_iou_thresh,
        max_masks=settings.max_masks,
    )
    log.info("Kept %d of %d masks after filtering", len(kept), len(raw_masks))

    boxes = {}
    labelable: list[Mask] = []
    for mask in kept:
        box = mask_bounding_box(mask.array)
        if box is None:
            continue
        boxes[mask.mask_id] = box
        labelable.append(mask)

    # 1b — vision-language labeling
    labels = await label_masks(image, labelable, boxes, settings)

    # 1c — structured JSON
    by_id = {mask.mask_id: mask for mask in labelable}
    objects: list[RoomObject] = []
    for label in labels:
        mask = by_id[label.mask_id]
        box = boxes[label.mask_id]

        locked = is_locked(label.category)
        lock_source = "policy"
        if label.mask_id in keep_mask_ids:
            locked, lock_source = True, "user_override"
        elif label.mask_id in replace_mask_ids:
            locked, lock_source = False, "user_override"

        area = mask.area
        objects.append(
            RoomObject(
                label=label.name,
                mask_id=label.mask_id,
                bounding_box=box,
                locked=locked,
                category=label.category,
                confidence=label.confidence,
                area_px=area,
                area_frac=area / image_area,
                lock_source=lock_source,
                notes=label.notes,
            )
        )

    objects.sort(key=lambda o: o.area_px, reverse=True)

    analysis = RoomAnalysis(
        image_width=width,
        image_height=height,
        objects=objects,
        masks_returned=len(raw_masks),
        masks_labeled=len(objects),
    )
    return analysis, by_id


def compose_inpaint_mask(
    analysis: RoomAnalysis, masks: dict[str, Mask], settings: Settings
) -> Image.Image:
    """Union every locked object's mask into the generator's inpainting mask."""
    locked = [
        masks[obj.mask_id].array
        for obj in analysis.objects
        if obj.locked and obj.mask_id in masks
    ]
    mask = build_inpaint_mask(
        (analysis.image_width, analysis.image_height),
        locked,
        dilation_px=settings.locked_dilation_px,
        invert=settings.invert_inpaint_mask,
    )
    log.info(
        "Locked %d/%d regions; %.1f%% of the frame is editable",
        len(locked),
        len(analysis.objects),
        100 * editable_fraction(mask, inverted=settings.invert_inpaint_mask),
    )
    return mask


async def run_pipeline(
    data: bytes,
    style: str,
    settings: Settings,
    *,
    extra_prompt: str = "",
    seed: int | None = None,
    keep_mask_ids: set[str] | None = None,
    replace_mask_ids: set[str] | None = None,
) -> tuple[RoomAnalysis, GenerationResult]:
    """Full flow. Returns both the analysis JSON and the generated image."""
    image = prepare_image(data, settings)
    analysis, masks = await analyze_room(
        image,
        settings,
        keep_mask_ids=keep_mask_ids,
        replace_mask_ids=replace_mask_ids,
    )

    inpaint_mask = compose_inpaint_mask(analysis, masks, settings)
    prompt = build_prompt(style, extra_prompt)
    image_b64, image_url = await generate_with_mask(
        image, inpaint_mask, prompt, settings, seed=seed
    )

    generation = GenerationResult(
        image_base64=image_b64,
        image_url=image_url,
        inpaint_mask_base64=encode_mask(inpaint_mask),
        prompt=prompt,
    )
    return analysis, generation
