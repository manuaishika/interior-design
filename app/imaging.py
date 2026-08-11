"""Pure image/mask helpers.

Deliberately free of network calls so the geometry can be unit-tested without
touching Replicate or OpenAI.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .models import BoundingBox


@dataclass
class Mask:
    """A single binary segmentation mask aligned to the analyzed image."""

    mask_id: str
    array: np.ndarray  # bool, shape (H, W)

    @property
    def area(self) -> int:
        return int(self.array.sum())

    @property
    def shape(self) -> tuple[int, int]:
        return self.array.shape  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------


def image_to_png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def image_to_base64(image: Image.Image) -> str:
    return base64.b64encode(image_to_png_bytes(image)).decode("ascii")


def image_to_data_uri(image: Image.Image) -> str:
    return f"data:image/png;base64,{image_to_base64(image)}"


def load_image(data: bytes) -> Image.Image:
    """Decode arbitrary upload bytes to RGB, honouring EXIF orientation.

    EXIF rotation matters here: SAM2 sees the transposed pixels, so if we
    skipped this the masks would come back rotated relative to our copy.
    """
    image = Image.open(io.BytesIO(data))
    try:
        from PIL import ImageOps

        image = ImageOps.exif_transpose(image)
    except Exception:  # pragma: no cover - malformed EXIF, not worth failing on
        pass
    return image.convert("RGB")


def fit_to_max_edge(image: Image.Image, max_edge: int) -> Image.Image:
    """Downscale so the longest edge is `max_edge`. Never upscales."""
    w, h = image.size
    longest = max(w, h)
    if longest <= max_edge:
        return image
    scale = max_edge / longest
    return image.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)


def snap_to_multiple(image: Image.Image, multiple: int = 8) -> Image.Image:
    """Crop to a size divisible by `multiple` (diffusion VAEs require this)."""
    w, h = image.size
    nw, nh = (w // multiple) * multiple, (h // multiple) * multiple
    if (nw, nh) == (w, h) or nw == 0 or nh == 0:
        return image
    return image.crop((0, 0, nw, nh))


# ---------------------------------------------------------------------------
# Mask decoding & geometry
# ---------------------------------------------------------------------------


def decode_mask(data: bytes, size: tuple[int, int], mask_id: str) -> Mask:
    """Decode a mask image (from SAM2) into a boolean array at `size`.

    SAM2 returns masks as single-channel images where the region is bright.
    Resized with NEAREST so we never invent grey edge pixels that would then
    be thresholded arbitrarily.
    """
    img = Image.open(io.BytesIO(data)).convert("L")
    if img.size != size:
        img = img.resize(size, Image.NEAREST)
    return Mask(mask_id=mask_id, array=np.array(img) > 127)


def mask_bounding_box(mask: np.ndarray) -> BoundingBox | None:
    """Tight bbox around the True pixels, or None for an empty mask."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return None
    y0, y1 = np.where(rows)[0][[0, -1]]
    x0, x1 = np.where(cols)[0][[0, -1]]
    return BoundingBox(
        x=int(x0), y=int(y0), width=int(x1 - x0 + 1), height=int(y1 - y0 + 1)
    )


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    if inter == 0:
        return 0.0
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union)


def filter_masks(
    masks: list[Mask],
    *,
    image_area: int,
    min_area_frac: float,
    max_area_frac: float,
    dedupe_iou_thresh: float,
    max_masks: int,
) -> list[Mask]:
    """Drop specks, drop whole-image masks, collapse duplicates, cap the count.

    Ordered largest-first so that when we truncate to `max_masks` we keep the
    regions that matter most for layout (and for the locked-region mask).
    """
    candidates = [
        m
        for m in masks
        if min_area_frac * image_area <= m.area <= max_area_frac * image_area
    ]
    candidates.sort(key=lambda m: m.area, reverse=True)

    kept: list[Mask] = []
    for mask in candidates:
        if any(mask_iou(mask.array, k.array) >= dedupe_iou_thresh for k in kept):
            continue
        kept.append(mask)
        if len(kept) >= max_masks:
            break
    return kept


def crop_with_context(
    image: Image.Image, box: BoundingBox, padding_frac: float
) -> Image.Image:
    """Crop around `box` with padding, clamped to the image bounds."""
    w, h = image.size
    pad_x = int(box.width * padding_frac)
    pad_y = int(box.height * padding_frac)
    x0 = max(0, box.x - pad_x)
    y0 = max(0, box.y - pad_y)
    x1 = min(w, box.x + box.width + pad_x)
    y1 = min(h, box.y + box.height + pad_y)
    if x1 <= x0 or y1 <= y0:
        return image.copy()
    return image.crop((x0, y0, x1, y1))


def highlight_region(
    image: Image.Image, mask: np.ndarray, box: BoundingBox, dim: float = 0.45
) -> Image.Image:
    """Full-frame view with everything outside the mask dimmed and a box drawn.

    Given to the VLM alongside the tight crop: the crop alone cannot distinguish
    a wall from a floor from a ceiling, but the in-context view can.
    """
    base = image.convert("RGB")
    dimmed = Image.blend(base, Image.new("RGB", base.size, (0, 0, 0)), dim)
    mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    composed = Image.composite(base, dimmed, mask_img)

    draw = ImageDraw.Draw(composed)
    x0, y0, x1, y1 = box.as_xyxy()
    draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=(255, 0, 0), width=3)
    return composed


# ---------------------------------------------------------------------------
# Inpainting mask composition
# ---------------------------------------------------------------------------


def build_inpaint_mask(
    size: tuple[int, int],
    locked_masks: list[np.ndarray],
    *,
    dilation_px: int = 0,
    invert: bool = False,
) -> Image.Image:
    """Compose the mask handed to the generator.

    Convention (before `invert`): WHITE = regenerate, BLACK = preserve, which is
    what Stable-Diffusion inpainting endpoints expect.

    Two deliberate choices:

    * The canvas starts fully white, so pixels no SAM2 mask covered are
      regenerated. Unsegmented background is open space, not something to
      protect.
    * Locked regions are painted black *last* and win every overlap. Where an
      unlocked sofa mask overlaps a locked window mask, the window wins — the
      failure mode we care about is destroying a door, not under-editing a
      couch.
    """
    width, height = size
    mask = np.zeros((height, width), dtype=bool)  # True = locked/preserve
    for locked in locked_masks:
        if locked.shape != (height, width):
            raise ValueError(
                f"locked mask shape {locked.shape} != image shape {(height, width)}"
            )
        mask |= locked

    # White = editable, black = preserved.
    img = Image.fromarray(np.where(mask, 0, 255).astype(np.uint8), mode="L")

    if dilation_px > 0 and mask.any():
        # MinFilter shrinks the white (editable) area, i.e. grows the locked
        # region — a cushion against diffusion bleeding over the boundary.
        size_px = dilation_px * 2 + 1
        img = img.filter(ImageFilter.MinFilter(min(size_px, 9)))
        # MinFilter caps out at a 9px kernel, so iterate for larger cushions.
        remaining = dilation_px - 4
        while remaining > 0:
            img = img.filter(ImageFilter.MinFilter(min(remaining * 2 + 1, 9)))
            remaining -= 4

    if invert:
        img = Image.eval(img, lambda v: 255 - v)
    return img


def editable_fraction(inpaint_mask: Image.Image, *, inverted: bool = False) -> float:
    """Share of the frame the generator is allowed to repaint."""
    arr = np.array(inpaint_mask.convert("L")) > 127
    if inverted:
        arr = ~arr
    return float(arr.mean())
