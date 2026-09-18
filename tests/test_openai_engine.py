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

        async def redraw(image, mask, prompt, s, **kw):
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

        async def redraw(image, mask, prompt, s, **kw):
            asked.append(prompt)
            buf = io.BytesIO()
            Image.new("RGB", (8, 8)).save(buf, "PNG")
            return buf.getvalue()

        monkeypatch.setattr("app.openai_images.find_structure", find)
        monkeypatch.setattr("app.openai_images.redraw", redraw)

        buf = io.BytesIO()
        Image.new("RGB", (100, 100)).save(buf, "PNG")
        # A paid tier, because free is capped at two and this is about the
        # instructions differing, not about the cap.
        await run_pipeline(buf.getvalue(), "japandi", settings(tier="room"),
                           variants=3)
        assert len(set(asked)) == 3

    @pytest.mark.asyncio
    async def test_one_request_per_design_still_differs(self, monkeypatch):
        """FIX-NEXT.md #5: the page now fires one variants=1 request per
        design instead of one request for the whole batch, so the first can
        appear without waiting for the slowest. Every one of those separate
        calls used to land on i=0 and therefore the identical VARIATIONS
        slot — variant_offset is what a caller doing this has to set."""
        from app.pipeline import run_pipeline

        asked = []

        async def find(image, s):
            return []

        async def redraw(image, mask, prompt, s, **kw):
            asked.append(prompt)
            buf = io.BytesIO()
            Image.new("RGB", (8, 8)).save(buf, "PNG")
            return buf.getvalue()

        monkeypatch.setattr("app.openai_images.find_structure", find)
        monkeypatch.setattr("app.openai_images.redraw", redraw)

        buf = io.BytesIO()
        Image.new("RGB", (100, 100)).save(buf, "PNG")
        indices = []
        for offset in range(3):
            _, generations = await run_pipeline(
                buf.getvalue(), "japandi", settings(),
                variants=1, variant_offset=offset)
            indices.append(generations[0].variant_index)

        assert len(set(asked)) == 3          # three different prompts...
        assert indices == [0, 1, 2]           # ...correctly numbered as one batch

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

        async def redraw(image, mask, prompt, s, **kw):
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


class TestTheTierDecides:
    """Quality was never sent, so every render used OpenAI's default — the
    expensive end of a range spanning roughly thirty-five to one. And every
    plan handed out the same number of designs, which is not a plan."""

    @pytest.mark.asyncio
    async def test_quality_is_always_sent(self, monkeypatch):
        from app import openai_images

        sent = {}

        class FakeImages:
            @staticmethod
            async def edit(**kw):
                sent.update(kw)
                buf = io.BytesIO()
                Image.new("RGB", (8, 8)).save(buf, "PNG")
                import base64
                return type("R", (), {"data": [type("D", (), {
                    "b64_json": base64.b64encode(buf.getvalue()).decode()})()]})()

        monkeypatch.setattr(openai_images, "_client",
                            lambda s: type("C", (), {"images": FakeImages})())
        await openai_images.redraw(Image.new("RGB", (200, 150)), None,
                                   "redesign", settings())
        assert "quality" in sent and sent["quality"]

    @pytest.mark.parametrize("tier,quality,variants", [
        ("free", "medium", 2),
        ("room", "high", 3),
        ("studio", "xhigh", 4),
    ])
    def test_each_tier_gets_something_different(self, tier, quality, variants):
        from app.config import Settings, tier_of

        got = tier_of(Settings(tier=tier))
        assert got["quality"] == quality
        assert got["variants"] == variants

    def test_an_unknown_tier_is_the_cheap_one(self):
        """Failing open on a paid tier bills somebody for a plan they do not
        have."""
        from app.config import Settings, tier_of

        assert tier_of(Settings(tier="gold"))["quality"] == "medium"
        assert tier_of(Settings())["quality"] == "medium"

    @pytest.mark.asyncio
    async def test_the_free_tier_cannot_be_asked_for_four(self, monkeypatch):
        from app.pipeline import run_pipeline

        drawn = []

        async def find(image, s):
            return []

        async def redraw(image, mask, prompt, s, **kw):
            drawn.append(prompt)
            buf = io.BytesIO()
            Image.new("RGB", (8, 8)).save(buf, "PNG")
            return buf.getvalue()

        monkeypatch.setattr("app.openai_images.find_structure", find)
        monkeypatch.setattr("app.openai_images.redraw", redraw)

        buf = io.BytesIO()
        Image.new("RGB", (100, 100)).save(buf, "PNG")
        await run_pipeline(buf.getvalue(), "japandi", settings(tier="free"),
                           variants=4)
        assert len(drawn) == 2


class TestExtraViews:
    """A full redesign changes the ceiling, the floor and the furniture, then
    has to invent whatever one frame could not show. The edit endpoint takes
    up to sixteen reference images; it was being sent one."""

    @pytest.mark.asyncio
    async def test_other_views_are_sent_alongside(self, monkeypatch):
        from app import openai_images

        sent = {}

        class FakeImages:
            @staticmethod
            async def edit(**kw):
                sent.update(kw)
                buf = io.BytesIO()
                Image.new("RGB", (8, 8)).save(buf, "PNG")
                import base64
                return type("R", (), {"data": [type("D", (), {
                    "b64_json": base64.b64encode(buf.getvalue()).decode()})()]})()

        monkeypatch.setattr(openai_images, "_client",
                            lambda s: type("C", (), {"images": FakeImages})())

        other = io.BytesIO()
        Image.new("RGB", (50, 50)).save(other, "PNG")
        await openai_images.redraw(Image.new("RGB", (200, 150)), None,
                                   "redesign", settings(),
                                   references=[other.getvalue()])

        assert isinstance(sent["image"], list)
        assert len(sent["image"]) == 2

    @pytest.mark.asyncio
    async def test_one_view_is_still_sent_on_its_own(self, monkeypatch):
        """Not wrapped in a list of one — a light restyle needs no second
        view and the shape should not change under it."""
        from app import openai_images

        sent = {}

        class FakeImages:
            @staticmethod
            async def edit(**kw):
                sent.update(kw)
                buf = io.BytesIO()
                Image.new("RGB", (8, 8)).save(buf, "PNG")
                import base64
                return type("R", (), {"data": [type("D", (), {
                    "b64_json": base64.b64encode(buf.getvalue()).decode()})()]})()

        monkeypatch.setattr(openai_images, "_client",
                            lambda s: type("C", (), {"images": FakeImages})())
        await openai_images.redraw(Image.new("RGB", (200, 150)), None,
                                   "redesign", settings())
        assert not isinstance(sent["image"], list)


class TestApplyThisChange:
    """TODO.md #2: a follow-up edits the design on screen, not the room.

    "Draw it again" already reran the whole pipeline from the original
    photograph — this pins the other verb, pipeline.edit_design, which the
    /api/edit endpoint calls: the generated design has to be the subject
    handed to the image model, the style preset must not be silently
    reapplied, PRESERVE has to still ride along, and a chain of edits has to
    each build on the previous result rather than all reaching back to the
    first design.
    """

    def _png(self, colour=(10, 20, 30)):
        buf = io.BytesIO()
        Image.new("RGB", (32, 32), colour).save(buf, "PNG")
        return buf.getvalue()

    @pytest.mark.asyncio
    async def test_the_design_is_the_subject_not_the_room_photo(self, monkeypatch):
        from app.pipeline import edit_design

        seen = {}

        async def redraw(image, mask, prompt, s, **kw):
            seen["image"] = image
            seen["mask"] = mask
            return self._png()

        monkeypatch.setattr("app.openai_images.redraw", redraw)

        design = self._png(colour=(200, 100, 50))
        await edit_design(design, "make the wall deep olive", settings())

        expected = Image.open(io.BytesIO(design)).convert("RGB").tobytes()
        assert seen["image"].convert("RGB").tobytes() == expected
        assert seen["mask"] is None

    @pytest.mark.asyncio
    async def test_the_style_preset_is_not_reapplied(self, monkeypatch):
        from app.pipeline import edit_design

        seen = {}

        async def redraw(image, mask, prompt, s, **kw):
            seen["prompt"] = prompt
            return self._png()

        monkeypatch.setattr("app.openai_images.redraw", redraw)
        await edit_design(self._png(), "make the wall deep olive", settings())

        assert "make the wall deep olive" in seen["prompt"]
        assert "Redesign this room as" not in seen["prompt"]
        assert "japandi" not in seen["prompt"].lower()

    @pytest.mark.asyncio
    async def test_preserve_still_rides_along(self, monkeypatch):
        """Or the second edit reintroduces the doorway the first avoided."""
        from app.generation import PRESERVE
        from app.pipeline import edit_design

        seen = {}

        async def redraw(image, mask, prompt, s, **kw):
            seen["prompt"] = prompt
            return self._png()

        monkeypatch.setattr("app.openai_images.redraw", redraw)
        await edit_design(self._png(), "make the wall deep olive", settings())
        assert PRESERVE in seen["prompt"]

    @pytest.mark.asyncio
    async def test_three_edits_chain(self, monkeypatch):
        """A third instruction edits the second result, not the first."""
        from app.pipeline import edit_design

        seen_images = []

        async def redraw(image, mask, prompt, s, **kw):
            seen_images.append(image.tobytes())
            # A different colour each time, so the next call's input is
            # provably this output and not some earlier one.
            return self._png(colour=(10 * (len(seen_images) + 1), 0, 0))

        monkeypatch.setattr("app.openai_images.redraw", redraw)

        design = self._png(colour=(1, 1, 1))
        first = await edit_design(design, "a", settings())
        second = await edit_design(first, "b", settings())
        await edit_design(second, "c", settings())

        assert seen_images[1] == Image.open(io.BytesIO(first)).convert("RGB").tobytes()
        assert seen_images[2] == Image.open(io.BytesIO(second)).convert("RGB").tobytes()

    @pytest.mark.asyncio
    async def test_needs_the_openai_engine(self):
        from app.config import Settings
        from app.pipeline import edit_design

        with pytest.raises(ValueError, match="OpenAI"):
            await edit_design(self._png(), "make it warmer",
                              Settings(google_api_key="g"))


class TestApplyThisChangeEndpoint:
    def _png(self):
        buf = io.BytesIO()
        Image.new("RGB", (32, 32)).save(buf, "PNG")
        return buf.getvalue()

    def test_it_edits_and_returns_an_image(self, monkeypatch):
        from fastapi.testclient import TestClient

        import app.main as main
        from app.main import app

        async def redraw(image, mask, prompt, s, **kw):
            return self._png()

        monkeypatch.setattr(main, "get_settings", settings)
        monkeypatch.setattr("app.openai_images.redraw", redraw)

        with TestClient(app) as c:
            r = c.post("/api/edit",
                      files={"design": ("d.png", self._png(), "image/png")},
                      data={"instruction": "make the wall deep olive"})
        assert r.status_code == 200
        assert r.json()["image_base64"]

    def test_an_empty_instruction_is_refused(self, monkeypatch):
        from fastapi.testclient import TestClient

        import app.main as main
        from app.main import app

        monkeypatch.setattr(main, "get_settings", settings)

        with TestClient(app) as c:
            r = c.post("/api/edit",
                      files={"design": ("d.png", self._png(), "image/png")},
                      data={"instruction": "   "})
        assert r.status_code == 400

    def test_the_wrong_engine_is_refused_not_500d(self, monkeypatch):
        from fastapi.testclient import TestClient

        import app.main as main
        from app.config import Settings
        from app.main import app

        monkeypatch.setattr(main, "get_settings",
                            lambda: Settings(google_api_key="g"))

        with TestClient(app) as c:
            r = c.post("/api/edit",
                      files={"design": ("d.png", self._png(), "image/png")},
                      data={"instruction": "make it warmer"})
        assert r.status_code == 400
