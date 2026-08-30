"""Orchestration: upload -> analyze -> (style) -> masked generation.

The analysis step slots in between upload and generation; the surrounding flow
is unchanged.
"""

from __future__ import annotations

import asyncio
import logging
import random

from PIL import Image

from .config import DEFAULT_PROFILE, Settings, is_locked
from .generation import build_prompt, encode_mask
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
from .models import GenerationResult, RoomAnalysis, RoomObject

log = logging.getLogger(__name__)


def _is_local(settings: Settings) -> bool:
    return settings.backend.strip().lower() == "local"


async def read_room(image: Image.Image, settings: Settings):
    """Passes 1a-1b: get regions and their labels, whichever backend is active.

    The hosted backend segments and labels in two steps (SAM2, then a VLM per
    region); the keyless one does both in a single forward pass. Everything
    downstream sees the same masks and labels either way.

    Imports are deferred so that neither backend's dependencies are required by
    the other — a keyless Colab run never imports `replicate` or `openai`.
    """
    if _is_local(settings):
        from .local_models import segment_and_label_local

        return await segment_and_label_local(image, settings)

    from .labeling import label_masks
    from .segmentation import segment_room

    raw_masks = await segment_room(image, settings)
    kept = filter_masks(
        raw_masks,
        image_area=image.size[0] * image.size[1],
        min_area_frac=settings.min_mask_area_frac,
        max_area_frac=settings.max_mask_area_frac,
        dedupe_iou_thresh=settings.dedupe_iou_thresh,
        max_masks=settings.max_masks,
    )
    log.info("Kept %d of %d masks after filtering", len(kept), len(raw_masks))

    boxes = {}
    labelable = []
    for mask in kept:
        box = mask_bounding_box(mask.array)
        if box is None:
            continue
        boxes[mask.mask_id] = box
        labelable.append(mask)

    labels = await label_masks(image, labelable, boxes, settings)
    return labelable, labels


async def render(image, inpaint_mask, prompt, settings, seed=None):
    """Pass 4, on whichever backend is active."""
    if _is_local(settings):
        from .local_models import generate_with_mask_local

        return await generate_with_mask_local(image, inpaint_mask, prompt, settings, seed=seed)

    from .generation import generate_with_mask

    return await generate_with_mask(image, inpaint_mask, prompt, settings, seed=seed)


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
    profile: str | None = None,
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

    # 1a + 1b — regions and their labels
    masks, labels = await read_room(image, settings)

    boxes = {}
    usable = []
    for mask in masks:
        box = mask_bounding_box(mask.array)
        if box is None:
            continue
        boxes[mask.mask_id] = box
        usable.append(mask)

    # 1c — structured JSON
    by_id = {mask.mask_id: mask for mask in usable}
    objects: list[RoomObject] = []
    for label in labels:
        if label.mask_id not in by_id:
            continue
        mask = by_id[label.mask_id]
        box = boxes[label.mask_id]

        locked = is_locked(label.category, profile)
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
        lock_profile=(profile or DEFAULT_PROFILE).strip().lower(),
        masks_returned=len(masks),
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
    variants: int | None = None,
    profile: str | None = None,
    keep_mask_ids: set[str] | None = None,
    replace_mask_ids: set[str] | None = None,
) -> tuple[RoomAnalysis, list[GenerationResult]]:
    """Full flow. Returns the analysis JSON and N rendered design options.

    Analysis runs once and is shared across the options: segmentation and
    labeling are the expensive, slow part, and every option is constrained by
    the same locked-region mask anyway.
    """
    count = settings.default_variants if variants is None else variants
    count = max(1, min(count, settings.max_variants))

    image = prepare_image(data, settings)
    analysis, masks = await analyze_room(
        image,
        settings,
        profile=profile,
        keep_mask_ids=keep_mask_ids,
        replace_mask_ids=replace_mask_ids,
    )

    inpaint_mask = compose_inpaint_mask(analysis, masks, settings)
    prompt = build_prompt(style, extra_prompt)
    mask_b64 = encode_mask(inpaint_mask)

    # Distinct seeds are what make the options differ. A caller-supplied seed
    # anchors the run so a set of options can be reproduced exactly.
    base = seed if seed is not None else random.randrange(1, 2**31 - 1)
    seeds = [base + offset for offset in range(count)]

    log.info("Rendering %d option(s) with seeds %s", count, seeds)
    rendered = await asyncio.gather(
        *(
            render(image, inpaint_mask, prompt, settings, seed=s)
            for s in seeds
        ),
        return_exceptions=True,
    )

    generations: list[GenerationResult] = []
    failures: list[BaseException] = []
    for index, (result, used_seed) in enumerate(zip(rendered, seeds)):
        if isinstance(result, BaseException):
            log.warning("Option %d failed: %s", index, result)
            failures.append(result)
            continue
        image_b64, image_url = result
        generations.append(
            GenerationResult(
                image_base64=image_b64,
                image_url=image_url,
                inpaint_mask_base64=mask_b64,
                prompt=prompt,
                seed=used_seed,
                variant_index=index,
            )
        )

    # Partial success is still useful — one usable option beats an error page.
    # Only a clean sweep of failures is fatal.
    if not generations:
        raise failures[0]

    return analysis, generations
