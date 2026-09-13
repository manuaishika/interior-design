"""Tests for the renders that came back as somebody else's room.

The complaint, from real output: the desk gone, the television gone, the
office chair gone, a cupboard turned into a doorway, an air conditioner moved
to a wall it could not be on, and a one-bed room returned with two beds.

The cause was not the prompt. It was the mask. Everything except the doors and
windows was inside the repaint zone, so the model was being told, correctly
and precisely, to erase all of it and invent something new.
"""

import io

import pytest
from PIL import Image

from app.config import Settings
from app.generation import PRESERVE, build_prompt


def room():
    return Image.new("RGB", (200, 150), (180, 170, 160))


class TestTheMaskIsNoLongerADemolitionOrder:
    def test_it_is_off_by_default(self):
        assert Settings().use_inpaint_mask is False

    @pytest.mark.asyncio
    async def test_no_mask_is_sent(self, monkeypatch):
        """The whole bug in one assertion."""
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
        await openai_images.redraw(room(), None, "redesign",
                                   Settings(openai_api_key="k"))
        assert "mask" not in sent
        assert sent["image"] is not None

    @pytest.mark.asyncio
    async def test_a_mask_is_sent_when_asked_for(self, monkeypatch):
        """Still the right tool for a weak inpainting model, so it stays
        reachable rather than being deleted."""
        from app import openai_images
        from app.imaging import build_inpaint_mask

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
        mask = build_inpaint_mask((200, 150), [], dilation_px=0)
        await openai_images.redraw(room(), mask, "redesign",
                                   Settings(openai_api_key="k"))
        assert "mask" in sent

    @pytest.mark.asyncio
    async def test_the_pipeline_holds_the_mask_back(self, monkeypatch):
        from app.pipeline import run_pipeline

        seen = {}

        async def find(image, s):
            return [{"kind": "door", "box": [0.0, 0.0, 0.2, 1.0]}]

        async def redraw(image, mask, prompt, s):
            seen["mask"] = mask
            buf = io.BytesIO()
            Image.new("RGB", (8, 8)).save(buf, "PNG")
            return buf.getvalue()

        monkeypatch.setattr("app.openai_images.find_structure", find)
        monkeypatch.setattr("app.openai_images.redraw", redraw)

        buf = io.BytesIO()
        room().save(buf, "PNG")
        analysis, generations = await run_pipeline(
            buf.getvalue(), "none", Settings(openai_api_key="k"), variants=1)

        assert seen["mask"] is None
        # but it was still computed, so the door it found can be checked
        assert generations[0].inpaint_mask_base64
        assert analysis.objects


class TestTheWordsDoTheWork:
    """With no mask, the prompt is the only thing protecting the room."""

    def test_the_camera_must_not_move(self):
        """"it changed the angle again. On the left side it took away the
        space." """
        assert "camera position" in PRESERVE
        assert "Do not crop" in PRESERVE
        assert "proportions" in PRESERVE

    def test_a_wardrobe_may_not_become_a_door(self):
        """"there is a cupboard in the room which has been made into a door." """
        assert "wardrobe stays a wardrobe" in PRESERVE
        assert "never become a doorway" in PRESERVE

    def test_the_air_conditioner_cannot_move_walls(self):
        """"the AC is on the left side. In the fourth picture — that's not
        humanly possible." """
        assert "air conditioner" in PRESERVE
        assert "cannot move to a different wall" in PRESERVE
        assert "only ever one of it" in PRESERVE

    def test_one_bed_stays_one_bed(self):
        """"why the fuck did it give it two beds? Did I ask for that?" """
        assert "one bed, the result shows one bed" in PRESERVE
        assert "never invent a second bed" in PRESERVE
        assert "hotel room" in PRESERVE

    def test_nothing_is_added_that_was_not_there(self):
        assert "Do not add furniture this room does not already have" in PRESERVE

    def test_every_style_carries_it(self):
        """Including "None", which is what was selected when this went wrong."""
        for style in ("none", "brief", "japandi", "industrial"):
            assert "recognisably the same room" in build_prompt(style)

    def test_contents_are_required_to_survive(self):
        prompt = build_prompt("none", contents="a desk, a television and a chair")
        assert "Every one of these stays" in prompt
        assert "none of them may be removed" in prompt

    def test_restyling_is_still_allowed(self):
        """"we can also change the desk, we can change the quality of the chair
        — but eradicating it does not serve our purpose." """
        prompt = build_prompt("none", contents="a desk")
        assert "You may restyle them" in prompt
        assert "better version of the same thing" in prompt
