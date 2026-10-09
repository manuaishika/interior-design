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

from . import checking, planning, store
from .config import (DEFAULT_PROFILE, Settings, is_locked, resolve_backend,
                     tier_of)
from .describe import count_instances, describe_room, keep_clause
from .generation import (COMMERCIAL_ROOMS, PRESERVE, GenerationError, before_client, build_prompt,
                         effective_keep, encode_mask)
from .imaging import (
    build_inpaint_mask,
    build_region_mask,
    image_to_png_bytes,
    Mask,
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


async def edit_design(design: bytes, instruction: str, settings: Settings,
                      tier: str = "",
                      region: tuple[float, float, float, float] | None = None,
                      mask_image: Image.Image | None = None) -> bytes:
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

    `mask_image` or `region` narrow it further to a specific part of the
    design. This is the one place in the whole app a mask is the right
    tool rather than the wrong one — everywhere else it would be a
    demolition order for whatever a strong editing model decided the rest
    of the room should become, because the "everywhere except this" it
    protects is a door frame or two against an entire redesign. Here the
    model is told to touch nothing BUT the shape a person actually marked,
    which is exactly what a mask is for.

    `mask_image` is a real drawing — a person's own brush strokes, any
    shape, not limited to a box — already decoded to our convention
    (white = editable, black = preserved) and resized to match `design`
    if it did not arrive at the same resolution the canvas it was drawn on
    happened to be. `region` is the coarser (x, y, w, h) fallback, still
    exposed for a caller with no drawing surface to draw on. If both
    arrive, the actual drawing wins — it says more than four numbers can.
    Left at neither, the edit is the whole photograph, same as before.

    Chaining ("a third instruction edits the second result") is the
    caller's job, not this function's: it always edits exactly the bytes it
    is given, so a caller that keeps carrying the latest result forward
    gets a chain, and one that keeps passing the same bytes gets three
    independent edits of the same design.
    """
    if not _is_openai(settings):
        raise ValueError("Apply this change needs the OpenAI engine.")

    from . import editing, openai_images

    image = load_image(design)
    if mask_image is not None:
        area = editing.area_for(image, drawn=mask_image)
    elif region is not None:
        area = editing.area_for(image, drawn=build_region_mask(image.size, region))
    else:
        # A typed change: find where it is about, so it can be held there.
        area = editing.area_for(
            image, boxes=await editing.locate(design, instruction, settings))

    prompt = editing.EDIT.format(instruction=instruction.strip(),
                                 area=editing.AREA if area is not None else "")
    result = await openai_images.redraw(image, area, prompt, settings,
                                        quality=tier_of(tier)["quality"],
                                        keep_detail=True)
    return editing.finish(image, result, area)


async def _plan_for(data: bytes, references, style: str, room: str,
                    depth: str, keep: str, extra: str, option: int,
                    settings: Settings) -> dict | None:
    """The redesign plan for a full redesign, or None for anything else
    (and whenever planning is unavailable — see planning.make_plan)."""
    if (depth or "").strip().lower() != "renovate":
        return None
    from . import planning
    from .generation import STYLES, room_brief

    key = style.strip().lower()
    plan = await planning.make_plan(
        [data] + [r for r in (references or []) if r],
        style_text=STYLES.get(key, style.strip()),
        room_text=room_brief(room) or room,
        keep=keep, extra=extra, option=option,
        key_parts=(style, room, depth, extra), settings=settings)
    return plan


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
                    contents="", keep="", depth="", variant_offset=0,
                    references=None):
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
    plan = await _plan_for(data, references, style, room, depth, keep,
                           extra_prompt, variant_offset, settings)
    plan_text = planning.render_plan(plan, room in COMMERCIAL_ROOMS)
    # views stays 0 here: the free engine is sent one photograph, so the prompt
    # must not tell it that several were supplied.
    prompt = build_prompt(style, extra_prompt, room=room, contents=contents,
                          keep=keep, depth=depth, plan=plan_text)

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
            changes=planning.changes_of(plan),
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
                      items=None, references=None, tier=""):
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
    allowed = min(tier_of(tier)["variants"], settings.max_variants)
    count = max(1, min(variants or settings.default_variants, allowed))
    image = prepare_image(data, settings)

    # Finding doors and windows only matters if the mask is actually sent to
    # the image model (USE_INPAINT_MASK). By default it is not — protection is
    # in the prompt — so this was a GPT-4o call per design, several seconds and
    # a few cents each, whose answer nothing used. Skipped unless wanted.
    if settings.use_inpaint_mask:
        regions = await openai_images.find_structure(image, settings)
    else:
        regions = []
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
    plan = await _plan_for(data, references, style, room, depth, keep,
                           extra_prompt, variant_offset, settings)
    plan_text = planning.render_plan(plan, room in COMMERCIAL_ROOMS)
    prompt = build_prompt(style, extra_prompt, room=room, contents=contents,
                          keep=keep, depth=depth, plan=plan_text,
                          views=len(references or []))
    # variant_offset: see _run_free's docstring note — one request per
    # design still needs count(=1) requests to land on different wording.
    quality = tier_of(tier)["quality"]
    original_png = image_to_png_bytes(image)

    async def draw(text: str) -> bytes:
        """One drawing, with one more try if the model hiccuped."""
        try:
            return await openai_images.redraw(
                image, sent_mask, text, settings, references=references,
                quality=quality)
        except openai_images.OpenAIImageError as exc:
            if not openai_images.is_transient(exc):
                raise
            log.warning("Drawing failed (%s); trying once more", exc)
            await asyncio.sleep(2)
            return await openai_images.redraw(
                image, sent_mask, text, settings, references=references,
                quality=quality)

    async def one(i: int) -> tuple[bytes, str]:
        # A plan already makes each option its own design (the planner is
        # given the option number), and this nudge is one more sentence after
        # the client's own words, so it only runs without one.
        text = prompt if plan else before_client(
            prompt, openai_images.VARIATIONS[
                (variant_offset + i) % len(openai_images.VARIATIONS)])
        result = await draw(text)
        # Look at it next to the original. Serious faults get one redraw with
        # the faults named; the first attempt is kept if the redraw fails.
        faults = await checking.check(original_png, result, depth=depth,
                                      plan_text=plan_text, settings=settings)
        if not faults:
            return result, ""
        try:
            return await draw(checking.with_faults(text, faults)), \
                "Redrawn once to fix: " + " ".join(faults)
        except Exception as exc:     # noqa: BLE001 — the first attempt stands
            log.warning("Redraw after self-check failed (%s); keeping the first", exc)
            return result, ""

    drawn = await asyncio.gather(*(one(i) for i in range(count)),
                                 return_exceptions=True)

    mask_b64 = encode_mask(inpaint_mask)
    generations: list[GenerationResult] = []
    failures: list[BaseException] = []
    for index, result in enumerate(drawn):
        if isinstance(result, BaseException):
            log.warning("Option %d failed: %s", index, result)
            failures.append(result)
            continue
        result, checked = result
        if (index == 0 and settings.use_inpaint_mask
                and openai_images.looks_inverted(image, result, inpaint_mask)):
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
            changes=planning.changes_of(plan),
            checked=checked,
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
    tier: str = "",
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
    # See generation.effective_keep: the page ticks everything as must-stay,
    # which in a full redesign meant "replace nothing".
    keep = effective_keep(keep, items, contents, profile or "")

    if _is_free(settings):
        return await _run_free(data, style, settings, extra_prompt=extra_prompt,
                               variants=variants, room=room, contents=contents,
                               keep=keep, depth=profile or "",
                               variant_offset=variant_offset,
                               references=references)
    if _is_openai(settings):
        return await _run_openai(data, style, settings,
                                 extra_prompt=extra_prompt, variants=variants,
                                 room=room, contents=contents, keep=keep,
                                 depth=profile or "", variant_offset=variant_offset,
                                 items=items, references=references, tier=tier)

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
