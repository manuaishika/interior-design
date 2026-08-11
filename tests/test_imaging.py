import numpy as np
import pytest
from PIL import Image

from app.imaging import (
    Mask,
    build_inpaint_mask,
    crop_with_context,
    decode_mask,
    editable_fraction,
    filter_masks,
    fit_to_max_edge,
    image_to_png_bytes,
    mask_bounding_box,
    mask_iou,
    snap_to_multiple,
)
from app.models import BoundingBox


def box_mask(h, w, y0, y1, x0, x1):
    arr = np.zeros((h, w), dtype=bool)
    arr[y0:y1, x0:x1] = True
    return arr


class TestBoundingBox:
    def test_tight_box(self):
        box = mask_bounding_box(box_mask(100, 100, 10, 30, 20, 50))
        assert (box.x, box.y, box.width, box.height) == (20, 10, 30, 20)

    def test_single_pixel(self):
        box = mask_bounding_box(box_mask(50, 50, 7, 8, 9, 10))
        assert (box.x, box.y, box.width, box.height) == (9, 7, 1, 1)

    def test_empty_mask_returns_none(self):
        assert mask_bounding_box(np.zeros((20, 20), dtype=bool)) is None

    def test_xyxy_roundtrip(self):
        assert BoundingBox(x=5, y=6, width=10, height=4).as_xyxy() == (5, 6, 15, 10)


class TestIoU:
    def test_identical(self):
        m = box_mask(50, 50, 0, 25, 0, 25)
        assert mask_iou(m, m) == pytest.approx(1.0)

    def test_disjoint(self):
        a = box_mask(50, 50, 0, 10, 0, 10)
        b = box_mask(50, 50, 30, 40, 30, 40)
        assert mask_iou(a, b) == 0.0

    def test_half_overlap(self):
        a = box_mask(50, 50, 0, 20, 0, 10)   # 200 px
        b = box_mask(50, 50, 10, 30, 0, 10)  # 200 px, 100 shared
        assert mask_iou(a, b) == pytest.approx(100 / 300)


class TestFilterMasks:
    def make(self, specs, size=100):
        return [
            Mask(mask_id=f"mask_{i}", array=box_mask(size, size, *s))
            for i, s in enumerate(specs)
        ]

    def test_drops_specks_and_whole_image(self):
        masks = self.make([
            (0, 100, 0, 100),  # whole image -> dropped
            (0, 50, 0, 50),    # 25% -> kept
            (0, 2, 0, 2),      # 0.04% -> dropped
        ])
        kept = filter_masks(
            masks, image_area=10_000, min_area_frac=0.004, max_area_frac=0.95,
            dedupe_iou_thresh=0.8, max_masks=10,
        )
        assert [m.mask_id for m in kept] == ["mask_1"]

    def test_dedupes_near_identical(self):
        masks = self.make([(0, 50, 0, 50), (0, 50, 0, 49)])
        kept = filter_masks(
            masks, image_area=10_000, min_area_frac=0.0, max_area_frac=1.0,
            dedupe_iou_thresh=0.8, max_masks=10,
        )
        assert len(kept) == 1

    def test_keeps_distinct_regions(self):
        masks = self.make([(0, 30, 0, 30), (60, 90, 60, 90)])
        kept = filter_masks(
            masks, image_area=10_000, min_area_frac=0.0, max_area_frac=1.0,
            dedupe_iou_thresh=0.8, max_masks=10,
        )
        assert len(kept) == 2

    def test_caps_count_keeping_largest(self):
        masks = self.make([(0, 10, 0, 10), (0, 60, 0, 60), (0, 30, 0, 30)])
        kept = filter_masks(
            masks, image_area=10_000, min_area_frac=0.0, max_area_frac=1.0,
            dedupe_iou_thresh=0.99, max_masks=2,
        )
        assert [m.mask_id for m in kept] == ["mask_1", "mask_2"]


class TestInpaintMask:
    def test_unsegmented_area_is_editable(self):
        mask = build_inpaint_mask((40, 40), [])
        assert np.array(mask).min() == 255  # everything white == regenerate

    def test_locked_region_is_preserved(self):
        locked = box_mask(40, 40, 5, 15, 5, 15)
        arr = np.array(build_inpaint_mask((40, 40), [locked]))
        assert arr[10, 10] == 0     # inside the locked region -> preserved
        assert arr[30, 30] == 255   # elsewhere -> editable

    def test_locked_wins_over_overlap(self):
        """A locked window overlapping an unlocked sofa must stay preserved."""
        window = box_mask(40, 40, 0, 20, 0, 20)
        arr = np.array(build_inpaint_mask((40, 40), [window]))
        # The unlocked sofa is simply absent from the locked list, and the
        # canvas defaults to editable, so the overlap resolves to preserved.
        assert arr[10, 10] == 0

    def test_multiple_locked_regions_union(self):
        a = box_mask(40, 40, 0, 10, 0, 10)
        b = box_mask(40, 40, 30, 40, 30, 40)
        arr = np.array(build_inpaint_mask((40, 40), [a, b]))
        assert arr[5, 5] == 0 and arr[35, 35] == 0 and arr[20, 20] == 255

    def test_dilation_grows_the_locked_region(self):
        locked = box_mask(60, 60, 20, 40, 20, 40)
        plain = np.array(build_inpaint_mask((60, 60), [locked], dilation_px=0))
        grown = np.array(build_inpaint_mask((60, 60), [locked], dilation_px=6))
        assert (grown == 0).sum() > (plain == 0).sum()
        assert grown[19, 30] == 0  # a pixel just outside is now protected too

    def test_dilation_noop_when_nothing_locked(self):
        arr = np.array(build_inpaint_mask((30, 30), [], dilation_px=8))
        assert (arr == 255).all()

    def test_invert_flips_convention(self):
        locked = box_mask(30, 30, 0, 10, 0, 10)
        normal = np.array(build_inpaint_mask((30, 30), [locked]))
        flipped = np.array(build_inpaint_mask((30, 30), [locked], invert=True))
        assert (flipped == 255 - normal).all()

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape"):
            build_inpaint_mask((40, 40), [np.zeros((10, 10), dtype=bool)])

    def test_mask_is_width_height_ordered(self):
        """PIL is (w, h) and numpy is (h, w) — the classic place to swap them."""
        mask = build_inpaint_mask((80, 40), [])
        assert mask.size == (80, 40)
        assert np.array(mask).shape == (40, 80)

    def test_editable_fraction(self):
        locked = box_mask(40, 40, 0, 20, 0, 40)  # exactly half the frame
        mask = build_inpaint_mask((40, 40), [locked])
        assert editable_fraction(mask) == pytest.approx(0.5)
        inverted = build_inpaint_mask((40, 40), [locked], invert=True)
        assert editable_fraction(inverted, inverted=True) == pytest.approx(0.5)


class TestImagePrep:
    def test_fit_downscales_longest_edge(self):
        assert fit_to_max_edge(Image.new("RGB", (2000, 1000)), 500).size == (500, 250)

    def test_fit_never_upscales(self):
        assert fit_to_max_edge(Image.new("RGB", (100, 80)), 500).size == (100, 80)

    def test_snap_to_multiple_of_eight(self):
        assert snap_to_multiple(Image.new("RGB", (101, 67)), 8).size == (96, 64)

    def test_snap_leaves_aligned_images_alone(self):
        assert snap_to_multiple(Image.new("RGB", (96, 64)), 8).size == (96, 64)

    def test_decode_mask_resizes_to_image(self):
        src = Image.fromarray((box_mask(20, 20, 0, 10, 0, 10) * 255).astype("uint8"), "L")
        mask = decode_mask(image_to_png_bytes(src), (40, 40), "m0")
        assert mask.array.shape == (40, 40)
        assert mask.array[5, 5] and not mask.array[35, 35]

    def test_crop_clamps_to_bounds(self):
        image = Image.new("RGB", (100, 100))
        crop = crop_with_context(image, BoundingBox(x=0, y=0, width=10, height=10), 0.5)
        assert crop.size[0] <= 100 and crop.size[1] <= 100
