"""Tests for the free path: one no-card key, both halves of the job.

The network is stubbed at httpx, so what is exercised is everything this
project actually owns — request shape, response parsing, the two casings
Google uses on either side of the wire, error translation, and the gate.
"""

import base64
import io

import httpx
import pytest
from PIL import Image

from app.config import Settings, resolve_backend
from app.google_ai import GoogleError


def settings(**kw):
    return Settings(google_api_key="test-key", backend="free", **kw)


def png(size=(64, 48)):
    buf = io.BytesIO()
    Image.new("RGB", size, (200, 190, 180)).save(buf, format="PNG")
    return buf.getvalue()


def reply(payload, status=200):
    """Stand in for one round trip to Google."""
    seen = {}

    async def post(self, url, **kw):
        seen["url"] = url
        seen["params"] = kw.get("params")
        seen["json"] = kw.get("json")
        return httpx.Response(
            status, json=payload,
            request=httpx.Request("POST", "https://example.test"),
        )

    return post, seen


def text_part(text):
    return {"candidates": [{"content": {"parts": [{"text": text}]}}]}


class TestReading:
    @pytest.mark.asyncio
    async def test_reads_a_room(self, monkeypatch):
        from app import google_ai

        post, seen = reply(text_part(
            '{"is_room": true, "room": "A small bedroom.", '
            '"items": [{"name": "bed", "count": 2, "treatment": "redraw"}], '
            '"directions": []}'
        ))
        monkeypatch.setattr(httpx.AsyncClient, "post", post)

        out = await google_ai.read_room(png(), "bedroom", "SURVEY", settings())
        assert out["room"] == "A small bedroom."
        assert out["items"][0]["count"] == 2

    @pytest.mark.asyncio
    async def test_sends_the_key_and_the_photo(self, monkeypatch):
        from app import google_ai

        photo = png()
        post, seen = reply(text_part('{"is_room": true}'))
        monkeypatch.setattr(httpx.AsyncClient, "post", post)
        await google_ai.read_room(photo, "bedroom", "SURVEY", settings())

        assert seen["params"] == {"key": "test-key"}
        assert "gemini" in seen["url"] and seen["url"].endswith(":generateContent")

        parts = seen["json"]["contents"][0]["parts"]
        assert parts[0]["text"] == "SURVEY"
        assert base64.b64decode(parts[1]["inline_data"]["data"]) == photo
        # JSON mode, or the answer arrives wrapped in prose and fences.
        assert seen["json"]["generationConfig"]["response_mime_type"] == \
            "application/json"

    @pytest.mark.asyncio
    async def test_the_gate_holds_on_the_free_engine_too(self, monkeypatch):
        """The paid reader turns away a photo of a dog. So must this one, or
        the cheap path becomes the one that wastes the quota."""
        from app.reading import NotARoomError, read_room

        post, _ = reply(text_part(
            '{"is_room": false, "subject": "a golden retriever"}'))
        monkeypatch.setattr(httpx.AsyncClient, "post", post)

        with pytest.raises(NotARoomError, match="golden retriever"):
            await read_room(png(), "bedroom", settings())

    @pytest.mark.asyncio
    async def test_both_readers_are_given_the_same_instructions(self, monkeypatch):
        """One prompt, two engines. If they drift, the free demo stops
        predicting what the paid one will do."""
        from app.reading import SURVEY, read_room

        post, seen = reply(text_part('{"is_room": true, "room": "x"}'))
        monkeypatch.setattr(httpx.AsyncClient, "post", post)
        await read_room(png(), "kitchen", settings())

        sent = seen["json"]["contents"][0]["parts"][0]["text"]
        assert sent == SURVEY.format(room="kitchen")


class TestDrawing:
    @pytest.mark.asyncio
    async def test_returns_the_picture(self, monkeypatch):
        from app import google_ai

        drawn = png((32, 32))
        post, _ = reply({"candidates": [{"content": {"parts": [
            {"inlineData": {"mimeType": "image/png",
                            "data": base64.b64encode(drawn).decode()}}
        ]}}]})
        monkeypatch.setattr(httpx.AsyncClient, "post", post)

        assert await google_ai.redraw(png(), "Japandi", settings()) == drawn

    @pytest.mark.asyncio
    async def test_accepts_either_casing(self, monkeypatch):
        """Requests go out snake_case and responses come back camelCase. Rather
        than depend on which side of that line this API is on today, read both."""
        from app import google_ai

        drawn = png((16, 16))
        post, _ = reply({"candidates": [{"content": {"parts": [
            {"inline_data": {"data": base64.b64encode(drawn).decode()}}
        ]}}]})
        monkeypatch.setattr(httpx.AsyncClient, "post", post)

        assert await google_ai.redraw(png(), "Japandi", settings()) == drawn

    @pytest.mark.asyncio
    async def test_the_lock_travels_in_the_sentence(self, monkeypatch):
        """There is no mask on this path, so if the instruction stops saying
        'do not move the door', nothing else is stopping it."""
        from app import google_ai

        post, seen = reply({"candidates": [{"content": {"parts": [
            {"inlineData": {"data": base64.b64encode(png()).decode()}}
        ]}}]})
        monkeypatch.setattr(httpx.AsyncClient, "post", post)
        await google_ai.redraw(png(), "Japandi", settings())

        sent = seen["json"]["contents"][0]["parts"][0]["text"].lower()
        for word in ("door", "window", "wall", "perspective"):
            assert word in sent

    @pytest.mark.asyncio
    async def test_options_ask_for_different_rooms(self, monkeypatch):
        """No seed on this engine, so the variation has to be in the words. Two
        identical instructions would return two near-identical rooms."""
        from app import google_ai

        asked = []

        async def post(self, url, **kw):
            asked.append(kw["json"]["contents"][0]["parts"][0]["text"])
            return httpx.Response(200, json={"candidates": [{"content": {"parts": [
                {"inlineData": {"data": base64.b64encode(png()).decode()}}]}}]},
                request=httpx.Request("POST", "https://example.test"))

        monkeypatch.setattr(httpx.AsyncClient, "post", post)
        for i in range(3):
            await google_ai.redraw(png(), "Japandi", settings(), variant=i)

        assert len(set(asked)) == 3

    @pytest.mark.asyncio
    async def test_a_refusal_is_repeated_not_swallowed(self, monkeypatch):
        """When prose comes back where a picture should be, that sentence is
        the whole explanation."""
        from app import google_ai

        post, _ = reply(text_part("I can't edit photographs of people."))
        monkeypatch.setattr(httpx.AsyncClient, "post", post)

        with pytest.raises(GoogleError, match="photographs of people"):
            await google_ai.redraw(png(), "Japandi", settings())


class TestErrors:
    @pytest.mark.asyncio
    async def test_a_missing_key_says_where_to_get_one(self):
        from app import google_ai

        with pytest.raises(GoogleError, match="aistudio.google.com"):
            await google_ai.redraw(png(), "Japandi", Settings(backend="free"))

    @pytest.mark.asyncio
    async def test_a_rate_limit_says_to_wait(self, monkeypatch):
        """The free tier's defining failure. It must not read as 'broken'."""
        from app import google_ai

        post, _ = reply({"error": {"message": "quota"}}, status=429)
        monkeypatch.setattr(httpx.AsyncClient, "post", post)

        with pytest.raises(GoogleError, match="rate limit"):
            await google_ai.redraw(png(), "Japandi", settings())

    @pytest.mark.asyncio
    async def test_a_bad_key_says_so_plainly(self, monkeypatch):
        from app import google_ai

        post, _ = reply({"error": {"message": "API key not valid"}}, status=403)
        monkeypatch.setattr(httpx.AsyncClient, "post", post)

        with pytest.raises(GoogleError, match="GOOGLE_API_KEY"):
            await google_ai.redraw(png(), "Japandi", settings())


class TestHealthAndPipeline:
    def test_health_names_the_live_engine(self):
        """The page decides what to offer from this, and says it out loud."""
        from fastapi.testclient import TestClient
        from app.main import app
        import app.main as main

        main.get_settings.cache_clear()
        body = TestClient(app).get("/api/health").json()
        assert body["engine"] == resolve_backend(Settings())

    def test_free_health_admits_the_locks_are_only_asked_for(self, monkeypatch):
        """The paid path masks the door so the generator cannot repaint it.
        This one asks. That difference is the reason to pay, so it is reported
        rather than glossed over."""
        from fastapi.testclient import TestClient
        import app.main as main
        from app.main import app

        monkeypatch.setattr(main, "get_settings", lambda: settings())
        body = TestClient(app).get("/api/health").json()

        assert body["engine"] == "free"
        assert body["can_read"] is True and body["can_draw"] is True
        assert body["locks_are_enforced"] is False

    def test_hosted_health_enforces_its_locks(self, monkeypatch):
        from fastapi.testclient import TestClient
        import app.main as main
        from app.main import app

        monkeypatch.setattr(main, "get_settings",
                            lambda: Settings(openai_api_key="o",
                                             replicate_api_token="r"))
        body = TestClient(app).get("/api/health").json()
        assert body["engine"] == "hosted"
        assert body["locks_are_enforced"] is True

    @pytest.mark.asyncio
    async def test_generate_skips_segmentation_entirely(self, monkeypatch):
        """Nothing is segmented because nothing is masked, so the analysis
        comes back empty rather than claiming regions nobody measured."""
        from app.pipeline import run_pipeline

        async def redraw(photo, prompt, s, variant=0):
            return png((24, 24))

        monkeypatch.setattr("app.google_ai.redraw", redraw)
        analysis, generations = await run_pipeline(
            png(), "japandi", settings(), variants=2)

        assert analysis.masks_returned == 0
        assert analysis.objects == []
        assert len(generations) == 2
        assert generations[0].inpaint_mask_base64 == ""
        assert generations[0].image_base64

    @pytest.mark.asyncio
    async def test_one_failed_option_does_not_lose_the_others(self, monkeypatch):
        from app.pipeline import run_pipeline

        async def flaky(photo, prompt, s, variant=0):
            if variant == 0:
                raise GoogleError("hiccup")
            return png((24, 24))

        monkeypatch.setattr("app.google_ai.redraw", flaky)
        _, generations = await run_pipeline(png(), "japandi", settings(),
                                            variants=3)
        assert len(generations) == 2

    @pytest.mark.asyncio
    async def test_every_option_failing_is_an_error(self, monkeypatch):
        from app.pipeline import run_pipeline

        async def dead(photo, prompt, s, variant=0):
            raise GoogleError("quota")

        monkeypatch.setattr("app.google_ai.redraw", dead)
        with pytest.raises(GoogleError):
            await run_pipeline(png(), "japandi", settings(), variants=2)
