"""Keyless backend: models that run in the process instead of behind an API.

The hosted backend needs two paid accounts. This one needs none. It swaps:

    SAM2 + GPT-4o  ->  a single ADE20K semantic segmentation model
    Replicate SD   ->  the same Stable Diffusion weights via `diffusers`

ADE20K is an indoor-scene dataset whose class list is very close to the
taxonomy this pipeline already uses — wall, floor, door, windowpane, bed,
wardrobe, desk, chair, sofa. One forward pass returns regions *already
labelled*, collapsing the read and name passes into one and removing the
per-region vision-language calls entirely.

What it costs: labels are coarser (`table`, never "reclaimed oak coffee
table"), regions are semantic rather than per-instance (two chairs merge), and
ADE20K has no walkway class, so circulation is derived rather than recognised.

Heavy imports are deliberately inside the functions: importing this module must
stay cheap for anyone running the hosted backend.
"""

from __future__ import annotations

import base64
import io
import logging
from typing import Any

import numpy as np
from PIL import Image

from .config import Settings
from .imaging import Mask, image_to_png_bytes
from .labeling import RegionLabel

log = logging.getLogger(__name__)


class LocalModelError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# ADE20K class name -> our category
# ---------------------------------------------------------------------------
#
# Matched on the model's own `id2label` strings at runtime rather than on class
# indices: index tables differ between checkpoints, but the names are stable.
# ADE20K names are comma-separated synonym lists ("windowpane, window").

_DOOR = ("door", "doorway", "screen door")
_WINDOW = ("windowpane", "window")
_WALL = ("wall",)
_FLOOR = ("floor", "flooring", "rug", "carpet", "carpeting", "mat")
_CEILING = ("ceiling", "sky")

_CLUTTER = (
    "bag", "clothes", "apparel", "towel", "book", "bottle", "box", "basket",
    "ashcan", "trash can", "plaything", "toy", "paper", "food", "plate",
    "glass", "blanket", "cover",
)

_FURNITURE = (
    "bed", "cabinet", "table", "curtain", "chair", "painting", "sofa", "shelf",
    "mirror", "armchair", "seat", "desk", "wardrobe", "closet", "lamp",
    "bathtub", "cushion", "chest of drawers", "pillow", "coffee table",
    "bookcase", "bench", "countertop", "stove", "kitchen island", "computer",
    "swivel chair", "television", "ottoman", "pot", "flower", "plant", "tree",
    "light", "chandelier", "fan", "sconce", "vase", "clock", "poster",
    "sculpture", "screen", "monitor", "stool", "radiator", "refrigerator",
)


def ade_category(raw_name: str) -> str:
    """Map an ADE20K class name onto one of `config.CATEGORIES`.

    Order matters: structure is checked before furnishing so that a "door"
    never falls through into the furniture bucket.
    """
    name = (raw_name or "").strip().lower()
    if not name:
        return "other"
    parts = [p.strip() for p in name.split(",") if p.strip()]

    def hit(vocab: tuple[str, ...]) -> bool:
        return any(p == v or p.startswith(v + " ") or v in p for p in parts for v in vocab)

    if hit(_DOOR):
        return "door"
    if hit(_WINDOW):
        return "window"
    if hit(_CEILING):
        return "other"
    if hit(_WALL):
        return "wall"
    if hit(_CLUTTER):
        return "clutter"
    if hit(_FURNITURE):
        return "furniture"
    if hit(_FLOOR):
        return "floor"
    return "other"


def ade_label(raw_name: str, category: str) -> str:
    """Human-readable label: the first synonym for named things, else the category."""
    if category not in ("furniture", "clutter"):
        return category
    first = (raw_name or "").split(",")[0].strip().lower()
    return first or category


def derive_walkway(
    floor: np.ndarray, furniture: list[np.ndarray], *, min_frac: float = 0.02
) -> np.ndarray | None:
    """Circulation space: floor that nothing is standing on.

    ADE20K has no walkway class, so this is inferred rather than recognised —
    the one place the keyless backend is genuinely weaker than the hosted one.
    Returns None when what is left is too small to be a route.
    """
    if not floor.any():
        return None
    open_floor = floor.copy()
    for f in furniture:
        open_floor &= ~f
    if open_floor.mean() < min_frac:
        return None
    return open_floor


# ---------------------------------------------------------------------------
# Segmentation + labeling, in one pass
# ---------------------------------------------------------------------------

_SEG_CACHE: dict[str, Any] = {}


def _load_segmenter(model_id: str):
    if model_id in _SEG_CACHE:
        return _SEG_CACHE[model_id]
    try:
        import torch
        from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise LocalModelError(
            "The keyless backend needs `transformers` and `torch`. "
            "Install with: pip install transformers torch"
        ) from exc

    log.info("Loading segmentation model %s", model_id)
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = SegformerForSemanticSegmentation.from_pretrained(model_id)
    model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")
    _SEG_CACHE[model_id] = (processor, model)
    return processor, model


async def segment_and_label_local(
    image: Image.Image, settings: Settings
) -> tuple[list[Mask], list[RegionLabel]]:
    """Return masks and their labels together — one model, one pass."""
    import torch

    processor, model = _load_segmenter(settings.local_seg_model)
    inputs = processor(images=image, return_tensors="pt")
    if next(model.parameters()).is_cuda:
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits

    # Upsample the class map back to the analysed image's resolution.
    upsampled = torch.nn.functional.interpolate(
        logits, size=(image.size[1], image.size[0]), mode="bilinear", align_corners=False
    )
    class_map = upsampled.argmax(dim=1)[0].cpu().numpy()

    id2label = model.config.id2label
    image_area = class_map.size

    masks: list[Mask] = []
    labels: list[RegionLabel] = []
    furniture_arrays: list[np.ndarray] = []
    floor_array: np.ndarray | None = None

    present = [int(c) for c in np.unique(class_map)]
    for index, class_id in enumerate(present):
        region = class_map == class_id
        if region.mean() < settings.min_mask_area_frac:
            continue

        raw_name = str(id2label.get(class_id, id2label.get(str(class_id), "")))
        category = ade_category(raw_name)
        mask_id = f"mask_{index:03d}"

        if category == "furniture":
            furniture_arrays.append(region)
        if category == "floor":
            floor_array = region if floor_array is None else (floor_array | region)

        masks.append(Mask(mask_id=mask_id, array=region))
        labels.append(
            RegionLabel(
                mask_id=mask_id,
                category=category,
                name=ade_label(raw_name, category),
                confidence=0.8,
                notes=f"ade20k:{raw_name}",
            )
        )

    # Circulation, inferred from what the floor has left over.
    if floor_array is not None:
        walkway = derive_walkway(floor_array, furniture_arrays)
        if walkway is not None and walkway.sum() / image_area >= settings.min_mask_area_frac:
            mask_id = f"mask_{len(masks):03d}"
            masks.append(Mask(mask_id=mask_id, array=walkway))
            labels.append(
                RegionLabel(
                    mask_id=mask_id, category="walkway", name="walkway",
                    confidence=0.55, notes="derived: floor minus furniture",
                )
            )

    if not masks:
        raise LocalModelError("The segmentation model found no usable regions")

    log.info("Local backend read %d regions", len(masks))
    return masks, labels


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

_PIPE_CACHE: dict[str, Any] = {}


def _load_inpainter(model_id: str):
    if model_id in _PIPE_CACHE:
        return _PIPE_CACHE[model_id]
    try:
        import torch
        from diffusers import StableDiffusionInpaintPipeline
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise LocalModelError(
            "The keyless backend needs `diffusers`. Install with: pip install diffusers"
        ) from exc

    cuda = torch.cuda.is_available()
    log.info("Loading inpainting model %s (%s)", model_id, "cuda" if cuda else "cpu")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if cuda else torch.float32,
        safety_checker=None,
    )
    pipe = pipe.to("cuda" if cuda else "cpu")
    pipe.set_progress_bar_config(disable=True)
    _PIPE_CACHE[model_id] = pipe
    return pipe


async def generate_with_mask_local(
    image: Image.Image,
    inpaint_mask: Image.Image,
    prompt: str,
    settings: Settings,
    seed: int | None = None,
) -> tuple[str, str | None]:
    """Inpaint locally. Same signature as the hosted generator, so the pipeline
    does not care which one it is holding."""
    import torch

    from .generation import NEGATIVE_PROMPT

    pipe = _load_inpainter(settings.local_inpaint_model)

    # SD 1.x inpainting is trained at 512; feeding it a 1536px room produces
    # mush. Generate at the model's own resolution, then restore the caller's
    # aspect ratio so masks and outputs still line up.
    target = settings.local_generation_size
    small_image = image.resize((target, target), Image.LANCZOS)
    small_mask = inpaint_mask.convert("L").resize((target, target), Image.NEAREST)

    generator = None
    if seed is not None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=device).manual_seed(int(seed))

    result = pipe(
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        image=small_image,
        mask_image=small_mask,
        num_inference_steps=settings.generation_steps,
        guidance_scale=settings.generation_guidance,
        generator=generator,
    ).images[0]

    out = result.resize(image.size, Image.LANCZOS)
    return base64.b64encode(image_to_png_bytes(out)).decode("ascii"), None
