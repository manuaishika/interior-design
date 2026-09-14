"""Tests for the one-key engine: GPT-4o finds the structure, gpt-image-1
repaints around it.

The mask conversion gets the most attention here because it is the piece that
fails silently. Ours is white-repaints, OpenAI's is transparent-repaints, and
inverting it does not raise anything — it carefully repaints the door and
preserves the sofa. Only a test catches that.
"""

import io

import numpy as np
import pytest
from PIL import Image

from app.config import Settings, resolve_backend
from app.imaging import build_inpaint_mask
from app.openai_images import (
    OpenAIImageError, boxes_to_masks, to_openai_mask,
)


def settings(**kw):
    kw.setdefault("openai_api_key", "sk-test")
    return Settings(**kw)


class TestWhichEngine:
    def test_one_openai_key_is_enough(self):
        """The gap this closes: an OpenAI key alone used to read a room and
        then fail to draw it, because drawing needed Replicate."""
        assert resolve_backend(settings()) == "openai"

    def test_replicate_still_wins_when_present(self):
        """SAM2 gives outlines rather than boxes, so it stays the best one."""
        assert resolve_backend(
            settings(replicate_api_token="r")) == "hosted"

    def test_an_explicit_choice_still_wins(self):
        assert resolve_backend(settings(backend="free")) == "free"


class TestBoxesToMasks:
    def test_a_box_becomes_the_right_pixels(self):
        masks = boxes_to_masks(
            [{"kind": "door", "box": [0.0, 0.0, 0.5, 1.0]}], (100, 40))
        assert len(masks) == 1
        assert masks[0].array.shape == (40, 100)
        assert masks[0].array[:, :50].all()
        assert not masks[0].array[:, 50:].any()

    def test_the_kind_survives_into_the_id(self):
        masks = boxes_to_masks(
            [{"kind": "window", "box": [0.1, 0.1, 0.2, 0.2]}], (100, 100))
        assert masks[0].mask_id.startswith("window")

    def test_out_of_range_boxes_are_clamped_not_crashed_on(self):
        """Models return 1.02 and -0.01 often enough that this must not throw
        an index error in the middle of somebody's render."""
        masks = boxes_to_masks(
            [{"kind": "door", "box": [-0.5, -0.2, 1.4, 1.9]}], (50, 50))
        assert masks[0].array.all()

    def test_a_hairline_box_still_covers_a_pixel(self):
        """A thin window rounded to zero width would contribute nothing and
        silently stop being protected."""
        masks = boxes_to_masks(
            [{"kind": "window", "box": [0.5, 0.5, 0.5001, 0.5001]}], (100, 100))
        assert masks[0].area >= 1

    @pytest.mark.parametrize("junk", [
        {"kind": "door"},                              # no box
        {"kind": "door", "box": [0.1, 0.2]},           # too few
        {"kind": "door", "box": [0.9, 0.1, 0.1, 0.9]}, # right <= left
        {"kind": "door", "box": ["a", "b", "c", "d"]}, # not numbers
    ])
    def test_nonsense_is_skipped_quietly(self, junk):
        assert boxes_to_masks([junk], (50, 50)) == []

    def test_nothing_found_is_not_an_error(self):
        """A room with no door or window in frame is a normal photograph."""
        assert boxes_to_masks([], (50, 50)) == []


class TestMaskConversion:
    def our_mask(self):
        """White repaints, black preserves: a locked strip down the left."""
        locked = np.zeros((40, 100), dtype=bool)
        locked[:, :30] = True
        return build_inpaint_mask((100, 40), [locked], dilation_px=0)

    def test_locked_areas_become_opaque(self):
        """OpenAI preserves what is opaque. The door must be opaque."""
        out = Image.open(io.BytesIO(to_openai_mask(self.our_mask())))
        alpha = np.array(out.getchannel("A"))
        assert alpha[:, :30].min() == 255        # door: kept

    def test_editable_areas_become_transparent(self):
        out = Image.open(io.BytesIO(to_openai_mask(self.our_mask())))
        alpha = np.array(out.getchannel("A"))
        assert alpha[:, 40:].max() == 0          # the rest: repainted

    def test_the_conversion_is_not_inverted(self):
        """The whole point. Inverted, this repaints the door and keeps the
        sofa, and nothing anywhere would raise."""
        out = Image.open(io.BytesIO(to_openai_mask(self.our_mask())))
        alpha = np.array(out.getchannel("A"))
        assert alpha[:, 0] .mean() > alpha[:, -1].mean()

    def test_the_output_is_rgba_png(self):
        """A mask with no alpha channel is silently ignored by the endpoint."""
        out = Image.open(io.BytesIO(to_openai_mask(self.our_mask())))
        assert out.mode == "RGBA" and out.format == "PNG"

    def test_the_invert_flag_flips_it(self):
        plain = to_openai_mask(self.our_mask())
        flipped = to_openai_mask(self.our_mask(), inverted=True)
        assert plain != flipped

    def test_size_is_preserved(self):
        out = Image.open(io.BytesIO(to_openai_mask(self.our_mask())))
        assert out.size == (100, 40)


class TestShape:
    @pytest.mark.parametrize("size,expected", [
        ((1600, 900), "1536x864"),       # landscape room, kept landscape
        ((900, 1600), "864x1536"),       # portrait, kept portrait
        ((1000, 1000), "1536x1536"),     # square stays square
        ((1024, 768), "1536x1152"),      # 4:3 preserved
    ])
    def test_the_room_proportions_are_kept(self, size, expected):
        """gpt-image-2.5 takes an arbitrary WxH, so the output should match the
        room's real proportions rather than snapping to a square and losing a
        quarter of the width — which is how two beds became one."""
        from app.openai_images import _edit_size

        assert _edit_size(size) == expected

    @pytest.mark.parametrize("size", [(4000, 900), (300, 1400)])
    def test_extreme_panoramas_are_pulled_inside_the_limits(self, size):
        """The edit endpoint refuses an aspect ratio past 3:1. Whatever the
        photo, the requested size has to be accepted."""
        from app.openai_images import _edit_size

        w, h = (int(n) for n in _edit_size(size).split("x"))
        assert w % 16 == 0 and h % 16 == 0
        assert 1 / 3 - 1e-6 <= w / h <= 3 + 1e-6
        assert max(w, h) <= 3840


class TestErrors:
    @pytest.mark.asyncio
    async def test_a_missing_key_says_where_to_get_one(self):
        from app import openai_images

        mask = build_inpaint_mask((10, 10), [], dilation_px=0)
        with pytest.raises(OpenAIImageError, match="platform.openai.com"):
            await openai_images.redraw(Image.new("RGB", (10, 10)), mask,
                                       "x", Settings())


class TestThePipeline:
    @pytest.mark.asyncio
    async def test_structure_becomes_a_mask_and_an_analysis(self, monkeypatch):
        """Unlike the free path this reports what it found, because it
        genuinely measured something."""
        from app.pipeline import run_pipeline

        async def find(image, s):
            return [{"kind": "door", "box": [0.0, 0.0, 0.2, 1.0]},
                    {"kind": "window", "box": [0.8, 0.1, 1.0, 0.6]}]

        async def redraw(image, mask, prompt, s):
            buf = io.BytesIO()
            Image.new("RGB", (32, 32), (1, 2, 3)).save(buf, "PNG")
            return buf.getvalue()

        monkeypatch.setattr("app.openai_images.find_structure", find)
        monkeypatch.setattr("app.openai_images.redraw", redraw)

        buf = io.BytesIO()
        Image.new("RGB", (200, 150), (180, 170, 160)).save(buf, "PNG")
        analysis, generations = await run_pipeline(
            buf.getvalue(), "japandi", settings(), variants=2)

        labels = {o.label for o in analysis.objects}
        assert labels == {"door", "window"}
        assert all(o.locked for o in analysis.objects)
        assert len(generations) == 2
        # The mask actually used is returned, which is how you check the door
        # really was protected.
        assert generations[0].inpaint_mask_base64

    @pytest.mark.asyncio
    async def test_options_ask_for_different_rooms(self, monkeypatch):
        """No seed on this model either, so the variation is in the words."""
        from app.pipeline import run_pipeline

        asked = []

        async def find(image, s):
            return []

        async def redraw(image, mask, prompt, s):
            asked.append(prompt)
            buf = io.BytesIO()
            Image.new("RGB", (8, 8)).save(buf, "PNG")
            return buf.getvalue()

        monkeypatch.setattr("app.openai_images.find_structure", find)
        monkeypatch.setattr("app.openai_images.redraw", redraw)

        buf = io.BytesIO()
        Image.new("RGB", (100, 100)).save(buf, "PNG")
        await run_pipeline(buf.getvalue(), "japandi", settings(), variants=3)
        assert len(set(asked)) == 3

    def test_no_variation_is_a_blank_instruction(self):
        """The first slot used to be "" — no instruction at all — so option
        one of every batch was never told to differ from anything, which is
        why a run of "different" designs came back looking the same."""
        from app.openai_images import VARIATIONS

        assert all(v.strip() for v in VARIATIONS)

    @pytest.mark.asyncio
    async def test_the_room_brief_still_travels(self, monkeypatch):
        """A nursery has to stay a nursery on this engine too."""
        from app.pipeline import run_pipeline

        asked = []

        async def find(image, s):
            return []

        async def redraw(image, mask, prompt, s):
            asked.append(prompt)
            buf = io.BytesIO()
            Image.new("RGB", (8, 8)).save(buf, "PNG")
            return buf.getvalue()

        monkeypatch.setattr("app.openai_images.find_structure", find)
        monkeypatch.setattr("app.openai_images.redraw", redraw)

        buf = io.BytesIO()
        Image.new("RGB", (100, 100)).save(buf, "PNG")
        await run_pipeline(buf.getvalue(), "japandi", settings(),
                           variants=1, room="nursery")
        assert "cot" in asked[0]


class TestHealth:
    def test_it_reports_the_mask_is_off_by_default(self, monkeypatch):
        """A mask on gpt-image-2.5 erased the furniture it was not told to
        touch, so it stopped being sent by default — health must say so
        rather than keep claiming a guarantee that no longer holds."""
        from fastapi.testclient import TestClient
        import app.main as main
        from app.main import app

        monkeypatch.setattr(main, "get_settings", settings)
        with TestClient(app) as c:
            body = c.get("/api/health").json()
        assert body["engine"] == "openai"
        assert body["can_read"] is True and body["can_draw"] is True
        assert body["locks_are_enforced"] is False

    def test_it_reports_an_enforced_lock_when_the_mask_is_switched_back_on(
        self, monkeypatch,
    ):
        from fastapi.testclient import TestClient
        import app.main as main
        from app.main import app

        monkeypatch.setattr(
            main, "get_settings",
            lambda: settings(use_inpaint_mask=True))
        with TestClient(app) as c:
            body = c.get("/api/health").json()
        assert body["engine"] == "openai"
        assert body["locks_are_enforced"] is True
