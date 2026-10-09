"""Editing a design changes that one thing, not the design.

Reported: adding a comment, marking an item, or asking to rotate a piece
changed the rest of the design too. Image models regenerate the whole
picture even with a mask, a drawn circle marked only its ring of ink, and a
typed change had no area at all. Now a typed change is located first, circles
are filled, the prompt says a turned piece is the same piece, and the
original is pasted back outside the area so nothing else can move.
"""

import asyncio
import io

import numpy as np
import pytest
from PIL import Image

from app import editing, openai_images, planning
from app.config import Settings
from app.pipeline import edit_design


def png(size=(200, 100), colour=(10, 20, 30)):
    buf = io.BytesIO()
    Image.new("RGB", size, colour).save(buf, "PNG")
    return buf.getvalue()


def run(c):
    return asyncio.run(c)


def stub_draw(monkeypatch, seen):
    async def redraw(image, mask, prompt, s, **kw):
        seen["mask"], seen["prompt"], seen["kw"] = mask, prompt, kw
        return png(colour=(250, 0, 0))
    monkeypatch.setattr(openai_images, "redraw", redraw)


class TestATypedChangeIsHeldToWhereItIsAbout:
    def test_rotate_the_bed_is_located_and_the_rest_is_kept(self, monkeypatch):
        seen, asked = {}, []
        stub_draw(monkeypatch, seen)

        async def ask(prompt, photos, settings, seed=0, max_tokens=0):
            asked.append(prompt)
            return {"scope": "local", "boxes": [[0.1, 0.2, 0.4, 0.8]], "subject": "the bed"}

        monkeypatch.setattr(planning, "ask_json", ask)
        out = run(edit_design(png(), "rotate the bed to face the window",
                              Settings(openai_api_key="k")))
        assert "rotate the bed" in asked[0]
        assert "BOTH where it is now AND where it will be" in asked[0].replace("\n", " ")
        arr = np.array(Image.open(io.BytesIO(out)).convert("RGB"))
        assert tuple(arr[50, 190]) == (10, 20, 30)     # outside: original
        assert arr[50, 50][0] > 200                    # inside: the edit

    def test_a_whole_room_change_is_not_boxed_in(self, monkeypatch):
        seen = {}
        stub_draw(monkeypatch, seen)

        async def ask(*a, **k):
            return {"scope": "whole", "boxes": []}

        monkeypatch.setattr(planning, "ask_json", ask)
        run(edit_design(png(), "make all the light warmer", Settings(openai_api_key="k")))
        assert seen["mask"] is None
        assert "Only the marked area" not in seen["prompt"]

    def test_if_the_locator_is_down_the_edit_still_happens(self, monkeypatch):
        seen = {}
        stub_draw(monkeypatch, seen)
        run(edit_design(png(), "a rug", Settings(openai_api_key="k")))   # conftest refuses
        assert seen["mask"] is None and "a rug" in seen["prompt"]

    def test_a_box_covering_nearly_everything_is_a_whole_edit(self):
        img = Image.new("RGB", (200, 100))
        assert editing.area_for(img, boxes=[[0, 0, 1, 1]]) is None
        assert editing.area_for(img, boxes=[[0.2, 0.2, 0.4, 0.4]]) is not None


class TestTheSamePieceOnlyTurned:
    def test_the_prompt_forbids_swapping_the_piece(self):
        text = editing.EDIT
        assert "it is\nthe SAME piece" in text
        assert "Do not restyle it, upgrade it or swap it for\na similar one" in text
        assert "Everything else stays exactly as it is in this picture" in text

    def test_edits_ask_the_model_to_match_the_input_closely(self, monkeypatch):
        seen = {}
        stub_draw(monkeypatch, seen)
        run(edit_design(png(), "turn the chair", Settings(openai_api_key="k"),
                        region=(0.1, 0.1, 0.3, 0.3)))
        assert seen["kw"]["keep_detail"] is True


class TestHighFidelityFallsBackQuietly:
    def test_a_model_that_refuses_the_setting_is_asked_again_without_it(self, monkeypatch):
        import base64
        calls = []

        class Images:
            async def edit(self, **call):
                calls.append(dict(call))
                if "input_fidelity" in call:
                    raise RuntimeError("Unknown parameter: 'input_fidelity'")
                return type("R", (), {"data": [type("D", (), {
                    "b64_json": base64.b64encode(png()).decode()})()]})()

        monkeypatch.setattr(openai_images, "_client", lambda s: type("C", (), {"images": Images()})())
        out = run(openai_images.redraw(Image.new("RGB", (64, 64)), None, "x",
                                       Settings(openai_api_key="k"), keep_detail=True))
        assert out == png()
        assert "input_fidelity" in calls[0] and "input_fidelity" not in calls[1]
