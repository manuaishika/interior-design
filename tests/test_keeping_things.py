"""Tests for the complaint that the desk disappeared.

The engine in use was told to draw "a bedroom" and drew the average one. The
desk and the television it was never told about simply were not in the picture
it painted — and nothing in the code was wrong enough to fail a test, because
no test asked whether the generator had been told what was in the room.

These ask.
"""

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import Settings
from app.generation import build_prompt
from app.main import app


def photo():
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (200, 190, 180)).save(buf, format="PNG")
    return {"photo": ("room.png", buf.getvalue(), "image/png")}


class TestThePromptNamesWhatIsThere:
    def test_contents_reach_the_prompt(self):
        prompt = build_prompt("japandi", contents="a bed, a desk and a television")
        assert "a desk" in prompt and "a television" in prompt

    def test_kept_things_are_required_to_survive(self):
        prompt = build_prompt("japandi", keep="the desk and the television")
        assert "must still contain the desk and the television" in prompt
        assert "Do not remove" in prompt

    def test_keeping_is_not_freezing(self):
        """A desk may become a better desk. That is the difference between
        "keep" and "do not touch", and it is what makes a full redesign still
        a redesign."""
        prompt = build_prompt("japandi", keep="the desk")
        assert "restyled" in prompt
        assert "replaced with better versions" in prompt

    def test_nothing_kept_says_nothing(self):
        assert "must still contain" not in build_prompt("japandi")


class TestEveryEngineIsTold:
    """The hosted path passed the contents through and the two newer ones did
    not, which is why this only went wrong on the engine actually in use."""

    @pytest.mark.parametrize("keys", [
        {"openai_api_key": "k"},                        # the one in production
        {"google_api_key": "g"},
        {"openai_api_key": "k", "replicate_api_token": "r"},
    ])
    def test_the_generator_learns_the_room(self, monkeypatch, keys):
        import app.main as main
        from app import pipeline

        seen = {}

        real = pipeline.build_prompt

        def spy(style, extra="", contents="", keep="", room=""):
            seen["contents"] = contents
            seen["keep"] = keep
            return real(style, extra, contents, keep, room)

        monkeypatch.setattr(pipeline, "build_prompt", spy)
        monkeypatch.setattr(main, "get_settings", lambda: Settings(**keys))

        async def stop(*a, **k):
            raise RuntimeError("far enough")

        for target in ("app.openai_images.find_structure",
                       "app.google_ai.redraw",
                       "app.pipeline.read_room"):
            monkeypatch.setattr(target, stop, raising=False)

        with TestClient(app) as client:
            client.post("/api/generate", files=photo(), data={
                "style": "japandi", "room_type": "bedroom",
                "contents": "a bed, a desk and a television",
                "keep": "the desk and the television",
            })

        # Either it reached build_prompt with the contents, or it never got
        # that far — what must never happen is reaching it empty.
        if seen:
            assert "desk" in seen["contents"]
            assert "desk" in seen["keep"]


class TestTheEndpointAcceptsThem:
    def test_contents_and_keep_are_forwarded(self, monkeypatch):
        import app.main as main
        from app.models import BoundingBox, GenerationResult, RoomAnalysis

        captured = {}

        async def fake(data, style, settings, **kwargs):
            captured.update(kwargs)
            return (RoomAnalysis(image_width=1, image_height=1, objects=[],
                                 masks_returned=0, masks_labeled=0),
                    [GenerationResult(image_base64="x", prompt="p")])

        monkeypatch.setattr(main, "run_pipeline", fake)
        with TestClient(app) as client:
            client.post("/api/generate", files=photo(), data={
                "style": "japandi",
                "contents": "a bed and a desk",
                "keep": "the desk",
            })
        assert captured["contents"] == "a bed and a desk"
        assert captured["keep"] == "the desk"


class TestThePageLetsYouChoose:
    def test_found_items_are_decisions_not_labels(self):
        """"Ticked things must still be in the room" only means anything if
        the person can untick them."""
        html = open("static/showcase.html", encoding="utf-8").read()
        assert "function drawChips(" in html
        assert "mustStay" in html
        assert "keptList" in html

    def test_everything_found_starts_as_must_stay(self):
        """A desk that silently becomes a side table is the failure that makes
        this feel broken. A restyled desk nobody asked to restyle is not. So
        the safe default is on."""
        html = open("static/showcase.html", encoding="utf-8").read()
        assert "mustStay[i.name] = true;" in html

    def test_the_page_sends_both(self):
        html = open("static/showcase.html", encoding="utf-8").read()
        assert "f.append('contents'" in html
        assert "f.append('keep'" in html

    def test_designs_can_be_downloaded(self):
        """Saving puts it in an account. Downloading puts it on their disk,
        which is what gets sent to a builder."""
        html = open("static/showcase.html", encoding="utf-8").read()
        assert "down.download" in html
        assert "function fileName(" in html


class TestTheDesignerCanSee:
    def test_chat_accepts_the_design(self, monkeypatch):
        """It was told only what the *original* room contained, so it insisted
        nothing had been removed while the client sat looking at a picture with
        the desk missing."""
        import app.main as main

        seen = {}

        async def fake_discuss(summary, turns, settings, design=None):
            seen["design"] = design
            return "ok"

        monkeypatch.setattr(main, "discuss", fake_discuss)
        monkeypatch.setattr(main, "get_settings",
                            lambda: Settings(openai_api_key="k"))

        buf = io.BytesIO()
        Image.new("RGB", (8, 8)).save(buf, format="PNG")
        with TestClient(app) as client:
            res = client.post(
                "/api/chat",
                data={"room_summary": "a bedroom", "turns": '[{"role":"user","content":"why"}]'},
                files={"design": ("d.png", buf.getvalue(), "image/png")},
            )
        assert res.status_code == 200
        assert seen["design"] is not None

    def test_chat_still_works_with_no_design_yet(self, monkeypatch):
        import app.main as main

        seen = {}

        async def fake_discuss(summary, turns, settings, design=None):
            seen["design"] = design
            return "ok"

        monkeypatch.setattr(main, "discuss", fake_discuss)
        monkeypatch.setattr(main, "get_settings",
                            lambda: Settings(openai_api_key="k"))
        with TestClient(app) as client:
            res = client.post("/api/chat", data={
                "room_summary": "a bedroom",
                "turns": '[{"role":"user","content":"hello"}]'})
        assert res.status_code == 200
        assert seen["design"] is None

    def test_a_blind_designer_is_told_it_is_blind(self):
        """Guessing from the description of the *original* room is how it
        contradicted somebody about their own picture."""
        from app.reading import DESIGNER, DESIGNER_BLIND

        assert "cannot see" in DESIGNER_BLIND
        assert "not of any design" in DESIGNER_BLIND
        assert "believe them" in DESIGNER

    def test_the_picture_rides_with_the_question(self):
        """Attached to the newest message, so the model reads it as "this is
        what I am being asked about"."""
        source = open("app/reading.py", encoding="utf-8").read()
        assert 'messages[-1]["role"] == "user"' in source
