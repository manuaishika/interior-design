import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.models import BoundingBox, GenerationResult, RoomAnalysis, RoomObject


@pytest.fixture
def client():
    return TestClient(app)


def photo_bytes(size=(64, 64)):
    buf = io.BytesIO()
    Image.new("RGB", size, (200, 190, 180)).save(buf, format="PNG")
    return buf.getvalue()


def upload(name="room.png"):
    return {"photo": (name, photo_bytes(), "image/png")}


ANALYSIS = RoomAnalysis(
    image_width=64,
    image_height=64,
    objects=[
        RoomObject(
            label="sofa", mask_id="mask_000",
            bounding_box=BoundingBox(x=1, y=2, width=10, height=8),
            locked=False, category="furniture", confidence=0.9,
        ),
        RoomObject(
            label="door", mask_id="mask_001",
            bounding_box=BoundingBox(x=40, y=5, width=12, height=40),
            locked=True, category="door", confidence=0.95,
        ),
    ],
    masks_returned=9, masks_labeled=2,
)

GENERATION = GenerationResult(
    image_base64="aW1hZ2U=", image_url="https://example.test/out.png",
    inpaint_mask_base64="bWFzaw==", prompt="Interior design photograph…",
)


class TestMetaEndpoints:
    def test_health(self, client):
        body = client.get("/api/health").json()
        assert body["status"] == "ok" and "segmentation" in body["models"]

    def test_styles_include_lock_policy(self, client):
        body = client.get("/api/styles").json()
        assert any(s["id"] == "japandi" for s in body["styles"])
        assert body["lock_profiles"]["renovate"]["door"] is True
        assert body["lock_profiles"]["renovate"]["furniture"] is False
        assert body["lock_profiles"]["renovate"]["clutter"] is False

    def test_site_is_served(self, client):
        res = client.get("/")
        assert res.status_code == 200
        assert "Second Draft" in res.text

    def test_the_site_is_five_pages_in_one_file(self, client):
        """Nav, studio, gallery, method, pricing — routed in the page so this
        deploys anywhere that can serve a static file."""
        html = client.get("/").text
        for page in ("studio", "explore", "how", "pricing"):
            assert 'data-go="' + page + '"' in html
            assert 'id="v-' + page + '"' in html
        assert 'id="v-home"' in html
        assert "/api/generate" in html

    def test_the_first_screen_explains_itself(self, client):
        """Someone arriving cold needs to know what this is before an upload
        box means anything. An earlier edit dropped this silently, so it is
        pinned here."""
        html = client.get("/").text
        assert 'class="steps how"' in html
        assert "It reads the room" in html

    def test_results_have_somewhere_to_live(self, client):
        """Both reference products lead with results. Ours needed a place for a
        finished design to land, which is most of why it read as a chat box."""
        html = client.get("/").text
        assert 'id="pane"' in html and 'id="shots"' in html
        assert "Your designs appear here" in html

    def test_one_button_does_the_thing(self, client):
        html = client.get("/").text
        assert 'id="go"' in html and 'id="plus"' in html

    def test_styles_show_what_they_look_like(self, client):
        """"Japandi" means nothing to most people, so each look carries the
        colours and materials it is built from."""
        html = client.get("/").text
        assert "colours: [" in html
        assert "pale oak" in html and "walnut" in html

    def test_every_look_is_drawn_not_just_named(self, client):
        """A swatch of four colours still leaves "Japandi" as a word. Each look
        is drawn as a room — that is the part a client can actually judge."""
        html = client.get("/").text
        assert 'id="looks"' in html
        assert "function scene(" in html
        assert "viewBox=\"0 0 300 220\"" in html
        assert html.count("scene: {") == 6

    def test_no_two_looks_are_the_same_picture(self, client):
        """Six identical bedrooms in six palettes all read as one picture, so
        each look is drawn on a different kind of room, and the gallery carries
        every combination."""
        html = client.get("/").text
        assert "function furniture(" in html
        assert "LOOK_ROOMS" in html
        for kind in ("living", "dining", "kitchen", "office", "bath"):
            assert "'" + kind + "'" in html
        assert 'id="explore"' in html

    def test_page_survives_without_the_claude_runtime(self, client):
        """Deployed on your own server there is no `claude` object at all;
        reading it unguarded would break the whole page."""
        html = client.get("/").text
        assert "typeof claude !== 'undefined'" in html

    def test_reading_and_chat_work_from_your_own_server(self, client):
        """The deployable path: vision endpoints that need no GPU."""
        html = client.get("/").text
        assert "/api/read" in html and "/api/chat" in html

    def test_site_carries_a_conversation(self, client):
        """After the reading, the client can argue with it about their room."""
        html = client.get("/").text
        assert 'id="thread"' in html
        assert "roomBrief" in html

    def test_no_plumbing_is_shown_to_the_user(self, client):
        """A server address box is a developer's concern. It reaches the page
        through a query string, never a field someone has to fill in."""
        html = client.get("/").text
        assert "Server address" not in html
        assert "function api(" in html
        assert "URLSearchParams" in html


class TestGenerateEndpoint:
    def test_returns_image_and_json(self, client, monkeypatch):
        async def fake_pipeline(data, style, settings, **kwargs):
            return ANALYSIS, [GENERATION]

        monkeypatch.setattr("app.main.run_pipeline", fake_pipeline)
        res = client.post("/api/generate", files=upload(), data={"style": "japandi"})
        assert res.status_code == 200

        body = res.json()
        assert body["generations"][0]["image_base64"] == "aW1hZ2U="
        assert body["generations"][0]["inpaint_mask_base64"] == "bWFzaw=="
        assert len(body["analysis"]["objects"]) == 2

        door = next(o for o in body["analysis"]["objects"] if o["label"] == "door")
        assert door["locked"] is True
        assert door["bounding_box"] == {"x": 40, "y": 5, "width": 12, "height": 40}

    def test_returns_several_options(self, client, monkeypatch):
        async def fake_pipeline(data, style, settings, **kwargs):
            return ANALYSIS, [
                GENERATION.model_copy(update={"variant_index": i, "seed": 100 + i})
                for i in range(3)
            ]

        monkeypatch.setattr("app.main.run_pipeline", fake_pipeline)
        body = client.post(
            "/api/generate", files=upload(),
            data={"style": "japandi", "variants": 3},
        ).json()
        assert [g["variant_index"] for g in body["generations"]] == [0, 1, 2]
        assert [g["seed"] for g in body["generations"]] == [100, 101, 102]

    def test_forwards_variant_and_profile(self, client, monkeypatch):
        captured = {}

        async def fake_pipeline(data, style, settings, **kwargs):
            captured.update(kwargs)
            return ANALYSIS, [GENERATION]

        monkeypatch.setattr("app.main.run_pipeline", fake_pipeline)
        client.post(
            "/api/generate", files=upload(),
            data={"style": "japandi", "variants": 4, "profile": "restyle"},
        )
        assert captured["variants"] == 4
        assert captured["profile"] == "restyle"

    def test_forwards_user_keep_replace_choices(self, client, monkeypatch):
        captured = {}

        async def fake_pipeline(data, style, settings, **kwargs):
            captured.update(kwargs)
            return ANALYSIS, [GENERATION]

        monkeypatch.setattr("app.main.run_pipeline", fake_pipeline)
        client.post(
            "/api/generate",
            files=upload(),
            data={
                "style": "industrial",
                "keep_mask_ids": "mask_000, mask_004",
                "replace_mask_ids": "mask_002",
            },
        )
        assert captured["keep_mask_ids"] == {"mask_000", "mask_004"}
        assert captured["replace_mask_ids"] == {"mask_002"}

    def test_style_is_required(self, client):
        assert client.post("/api/generate", files=upload()).status_code == 422

    def test_rejects_non_image(self, client):
        res = client.post(
            "/api/generate",
            files={"photo": ("notes.txt", b"hello", "text/plain")},
            data={"style": "japandi"},
        )
        assert res.status_code == 400

    def test_rejects_empty_upload(self, client):
        res = client.post(
            "/api/generate",
            files={"photo": ("room.png", b"", "image/png")},
            data={"style": "japandi"},
        )
        assert res.status_code == 400

    def test_upstream_failure_becomes_502(self, client, monkeypatch):
        from app.segmentation import SegmentationError

        async def boom(*a, **k):
            raise SegmentationError("SAM2 exploded")

        monkeypatch.setattr("app.main.run_pipeline", boom)
        res = client.post("/api/generate", files=upload(), data={"style": "japandi"})
        assert res.status_code == 502 and "SAM2 exploded" in res.json()["detail"]

    def test_contradictory_overrides_become_400(self, client, monkeypatch):
        async def boom(*a, **k):
            raise ValueError("mask ids appear in both keep and replace: ['mask_000']")

        monkeypatch.setattr("app.main.run_pipeline", boom)
        res = client.post(
            "/api/generate",
            files=upload(),
            data={"style": "japandi", "keep_mask_ids": "mask_000",
                  "replace_mask_ids": "mask_000"},
        )
        assert res.status_code == 400


class TestNotARoom:
    """A photo of a dog used to sail straight through: the reader invented a
    room, and the generator would have been billed to repaint it. The read pass
    runs before the expensive half, so that is where the gate belongs."""

    def test_reader_refuses_what_is_not_a_room(self, client, monkeypatch):
        from app.reading import NotARoomError

        async def not_a_room(*a, **k):
            raise NotARoomError("a golden retriever on a lawn")

        monkeypatch.setattr("app.main.read_room", not_a_room)
        res = client.post("/api/read", files=upload())

        # 422, not 502 — nothing is broken, the picture is just not a room.
        assert res.status_code == 422
        detail = res.json()["detail"]
        assert "golden retriever" in detail
        assert "not a room" in detail

    def test_the_message_says_what_to_send_instead(self, client, monkeypatch):
        from app.reading import NotARoomError

        async def not_a_room(*a, **k):
            raise NotARoomError("a screenshot")

        monkeypatch.setattr("app.main.read_room", not_a_room)
        detail = client.post("/api/read", files=upload()).json()["detail"]
        assert "empty" in detail          # a bare room is still a room
        assert "door or a window" in detail

    def test_an_empty_room_is_still_a_room(self):
        """The whole product is open space that can be transformed, so bare and
        unfinished rooms must pass the gate rather than trip it."""
        from app.reading import SURVEY

        prompt = SURVEY.lower()
        assert "bare" in prompt and "unfinished" in prompt
        for rejected in ("person", "animal", "screenshot", "landscape"):
            assert rejected in prompt

    def test_a_real_reading_passes_through_untouched(self, monkeypatch):
        import asyncio, json
        from types import SimpleNamespace
        from app.config import Settings
        from app import reading

        body = {"is_room": True, "room": "A small bedroom.", "items": [],
                "directions": []}

        class FakeClient:
            class chat:
                class completions:
                    @staticmethod
                    async def create(**kw):
                        msg = SimpleNamespace(content=json.dumps(body))
                        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

        monkeypatch.setattr(reading, "_client", lambda settings: FakeClient)
        out = asyncio.run(reading.read_room(b"x", "bedroom", Settings()))
        assert out["room"] == "A small bedroom."


class TestAnalyzeEndpoint:
    def test_returns_analysis_only(self, client, monkeypatch):
        async def fake_analyze(image, settings, **kwargs):
            return ANALYSIS, {}

        monkeypatch.setattr("app.main.analyze_room", fake_analyze)
        body = client.post("/api/analyze", files=upload()).json()
        assert "generation" not in body
        assert body["analysis"]["masks_returned"] == 9
