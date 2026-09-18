"""Orchestration: upload -> analyze -> (style) -> masked generation.

The analysis step slots in between upload and generation; the surrounding flow
is unchanged.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import random

from PIL import Image

from . import store
from .config import (DEFAULT_PROFILE, Settings, is_locked, resolve_backend,
                     tier_of)
from .describe import count_instances, describe_room, keep_clause
from .generation import PRESERVE, GenerationError, build_prompt, encode_mask
from .imaging import (
    build_inpaint_mask,
    image_to_png_bytes,
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
    return resolve_backend(settings) == "local"


def _is_free(settings: Settings) -> bool:
    return resolve_backend(settings) == "free"


def _is_openai(settings: Settings) -> bool:
    return resolve_backend(settings) == "openai"


async def edit_design(design: bytes, instruction: str, settings: Settings) -> bytes:
    """A follow-up edits the design that is on screen, not the room it came
    from.

    "Draw it again" reruns the whole pipeline from the original photograph,
    so a request as small as "change the wall colour" comes back as a
    different room with the one thing asked for lost among fifty others
    that were not. This is the other verb: the generated design is the
    subject, the instruction is the whole prompt, no style preset is
    re-applied, and PRESERVE still rides along so the edit cannot undo what
    the original render already protected — a second edit reintroducing the
    doorway the first one avoided would be worse than not editing at all.

    Chaining ("a third instruction edits the second result") is the
    caller's job, not this function's: it always edits exactly the bytes it
    is given, so a caller that keeps carrying the latest result forward
    gets a chain, and one that keeps passing the same bytes gets three
    independent edits of the same design.
    """
    if not _is_openai(settings):
        raise ValueError("Apply this change needs the OpenAI engine.")

    from . import openai_images

    image = load_image(design)
    prompt = f"{instruction.strip()}\n\n{PRESERVE}"
    return await openai_images.redraw(image, None, prompt, settings)


async def _furnish_contents(
    contents: str, items: list[dict] | None, style: str, room: str
) -> tuple[str, list[dict]]:
    """Steer the redraw towards real stock, where there is any.

    For each item the person did not tick as must-stay, look for a catalogue
    product in the same category, style and room and fold a description of
    it into `contents` — "a walnut desk, 1200mm wide" — so the generator
    aims at something orderable instead of inventing a generic one. With no
    catalogue loaded, or no items supplied, this is a no-op: contents comes
    back unchanged and no products are returned, exactly as the app behaved
    before a catalogue existed.
    """
    if not items or not store.is_configured():
        return contents, []

    from . import catalogue

    async with store.session() as db:
        phrases, matched = await catalogue.furnish(db, items, style=style, room=room)
    if phrases:
        contents = f"{contents}, {', '.join(phrases)}" if contents else ", ".join(phrases)
    return contents, matched


async def _run_free(data, style, settings, *, extra_prompt, variants, room="",
                    contents="", keep="", depth="", variant_offset=0):
    """The free path: no segmentation, no mask, one image model.

    There is nothing to segment because there is nothing to mask — the whole
    lock lives in the instruction. So this skips analysis entirely and returns
    an empty one, which is honest: no region was measured, so none is claimed.
    What the room contains still reaches the client through /api/read.
    """
    from . import google_ai

    count = max(1, min(variants or settings.default_variants,
                       settings.max_variants))
    image = prepare_image(data, settings)
    photo = image_to_png_bytes(image)
    prompt = build_prompt(style, extra_prompt, room=room, contents=contents,
                          keep=keep, depth=depth)

    # variant_offset lets the page ask for one design at a time instead of
    # waiting for a whole batch to finish before showing the first — see
    # run_pipeline's docstring. Each independent request still needs a
    # distinct wording nudge, or "one at a time" would draw the same option
    # three times over.
    drawn = await asyncio.gather(
        *(google_ai.redraw(photo, prompt, settings, variant=variant_offset + i)
          for i in range(count)),
        return_exceptions=True,
    )

    generations: list[GenerationResult] = []
    failures: list[BaseException] = []
    for index, result in enumerate(drawn):
        if isinstance(result, BaseException):
            log.warning("Free option %d failed: %s", index, result)
            failures.append(result)
            continue
        generations.append(GenerationResult(
            image_base64=base64.b64encode(result).decode("ascii"),
            inpaint_mask_base64="",
            prompt=prompt,
            variant_index=variant_offset + index,
        ))

    if not generations:
        raise failures[0] if failures else GenerationError("Nothing came back")

    analysis = RoomAnalysis(
        image_width=image.size[0], image_height=image.size[1],
        objects=[], masks_returned=0, masks_labeled=0,
    )
    return analysis, generations


async def _run_openai(data, style, settings, *, extra_prompt, variants, room="",
                      contents="", keep="", depth="", variant_offset=0,
                      items=None, references=None):
    """One OpenAI key. No Replicate anywhere.

    GPT-4o says where the doors, windows and walkways are; those boxes become
    a mask. By default the mask is built and returned but not sent — see
    openai_images' module docstring for why — so the protection is the same
    kind the free path uses: named in the prompt, not painted out in pixels.
    Set USE_INPAINT_MASK=true to send it and get the geometric guarantee back.

    Unlike the free path this returns a genuine analysis, because it genuinely
    measured something — boxes rather than outlines, but measured.
    """
    from . import openai_images

    # The tier decides how many, not the caller. A plan that sells four designs
    # and hands four to everybody is not a plan.
    allowed = min(tier_of(settings)["variants"], settings.max_variants)
    count = max(1, min(variants or settings.default_variants, allowed))
    image = prepare_image(data, settings)

    regions = await openai_images.find_structure(image, settings)
    masks = openai_images.boxes_to_masks(regions, image.size)

    objects: list[RoomObject] = []
    for mask, region in zip(masks, regions):
        box = mask_bounding_box(mask.array)
        if box is None:
            continue
        kind = str(region.get("kind") or "other").strip().lower()
        objects.append(RoomObject(
            label=kind, mask_id=mask.mask_id, bounding_box=box,
            locked=is_locked(kind, None), category=kind, confidence=0.0,
        ))

    # The mask is still built, and still returned, so the openings it found can
    # be checked. It is only *sent* if asked for — sending it is what erased the
    # furniture.
    inpaint_mask = build_inpaint_mask(
        image.size, [m.array for m in masks],
        dilation_px=settings.locked_dilation_px,
        invert=False,
    )
    sent_mask = inpaint_mask if settings.use_inpaint_mask else None

    contents, matched_products = await _furnish_contents(contents, items, style, room)

    # Without this the model is told to draw "a bedroom" and draws the average
    # one: the desk and the television it was never told about simply are not
    # in the picture it paints.
    prompt = build_prompt(style, extra_prompt, room=room, contents=contents,
                          keep=keep, depth=depth)
    # variant_offset: see _run_free's docstring note — one request per
    # design still needs count(=1) requests to land on different wording.
    drawn = await asyncio.gather(
        *(openai_images.redraw(
            image, sent_mask,
            prompt + openai_images.VARIATIONS[
                (variant_offset + i) % len(openai_images.VARIATIONS)],
            settings, references=references)
          for i in range(count)),
        return_exceptions=True,
    )

    mask_b64 = encode_mask(inpaint_mask)
    generations: list[GenerationResult] = []
    failures: list[BaseException] = []
    for index, result in enumerate(drawn):
        if isinstance(result, BaseException):
            log.warning("Option %d failed: %s", index, result)
            failures.append(result)
            continue
        if index == 0 and openai_images.looks_inverted(image, result, inpaint_mask):
            log.warning(
                "The locked regions changed more than the editable ones. The "
                "inpainting mask is probably the wrong way round for this "
                "model — set INVERT_INPAINT_MASK=true and try again."
            )
        generations.append(GenerationResult(
            image_base64=base64.b64encode(result).decode("ascii"),
            inpaint_mask_base64=mask_b64,
            prompt=prompt,
            variant_index=variant_offset + index,
        ))

    if not generations:
        raise failures[0] if failures else GenerationError("Nothing came back")

    analysis = RoomAnalysis(
        image_width=image.size[0], image_height=image.size[1],
        objects=objects, masks_returned=len(masks), masks_labeled=len(objects),
        products=matched_products,
    )
    return analysis, generations


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

    # Count what is actually in the room. Semantic segmentation gives one blob
    # per class, so two beds arrive as a single "bed" region — splitting it into
    # connected pieces recovers the number, which is the fact the generator
    # would otherwise invent for itself.
    contents: dict[str, int] = {}
    for label in labels:
        if label.category != "furniture" or label.mask_id not in by_id:
            continue
        n = count_instances(by_id[label.mask_id].array)
        if n:
            contents[label.name] = contents.get(label.name, 0) + n

    analysis = RoomAnalysis(
        image_width=width,
        image_height=height,
        objects=objects,
        lock_profile=(profile or DEFAULT_PROFILE).strip().lower(),
        contents=contents,
        described_as=describe_room(contents),
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
    variant_offset: int = 0,
    room: str = "",
    contents: str = "",
    keep: str = "",
    items: list[dict] | None = None,
    references: list[bytes] | None = None,
    profile: str | None = None,
    keep_mask_ids: set[str] | None = None,
    replace_mask_ids: set[str] | None = None,
) -> tuple[RoomAnalysis, list[GenerationResult]]:
    """Full flow. Returns the analysis JSON and N rendered design options.

    Analysis runs once and is shared across the options: segmentation and
    labeling are the expensive, slow part, and every option is constrained by
    the same locked-region mask anyway.

    `variant_offset` is for a caller doing one request per design instead of
    one request for the whole batch — asyncio.gather inside a single call
    means nothing appears until the slowest of the batch finishes, which for
    two or three 20-40 second images reads as the app having hung. Firing N
    independent variants=1 requests instead lets the page show the first as
    soon as it lands; each still needs a distinct wording nudge or all N
    would draw the same option, hence the offset.
    """
    if _is_free(settings):
        return await _run_free(data, style, settings, extra_prompt=extra_prompt,
                               variants=variants, room=room, contents=contents,
                               keep=keep, depth=profile or "",
                               variant_offset=variant_offset)
    if _is_openai(settings):
        return await _run_openai(data, style, settings,
                                 extra_prompt=extra_prompt, variants=variants,
                                 room=room, contents=contents, keep=keep,
                                 depth=profile or "", variant_offset=variant_offset,
                                 items=items, references=references)

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
    prompt = build_prompt(
        style,
        extra_prompt,
        contents=contents or analysis.described_as,
        # What the person ticked wins over what was merely counted.
        keep=keep or keep_clause(analysis.contents),
        depth=profile or "",
        room=room,
    )
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
