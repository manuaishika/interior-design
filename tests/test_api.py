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

    def test_site_is_a_tool_not_a_brochure(self, client):
        """Three panes: what you are working on, the canvas, the controls.
        The client called the old scrolling page complicated; this is the fix."""
        html = client.get("/").text
        assert 'class="app"' in html
        assert 'class="rail"' in html and 'class="canvas"' in html
        assert 'class="panel"' in html
        assert "/api/generate" in html

    def test_one_button_does_the_thing(self, client):
        html = client.get("/").text
        assert 'id="design"' in html
        assert 'id="drop"' in html

    def test_site_reads_rooms_without_an_engine(self, client):
        """Reading runs in the page via Claude, so the published link is useful
        even with no GPU behind it. Only drawing needs the engine."""
        html = client.get("/").text
        assert "claude.use('sample')" in html
        assert 'id="read"' in html

    def test_site_carries_a_conversation(self, client):
        """After the reading, the client can argue with it about their room."""
        html = client.get("/").text
        assert 'id="thread"' in html
        assert "roomBrief" in html

    def test_site_can_point_at_a_remote_engine(self, client):
        """The page is a fixed address; the engine moves independently."""
        html = client.get("/").text
        assert "engineUrl" in html
        assert "function api(" in html


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


class TestAnalyzeEndpoint:
    def test_returns_analysis_only(self, client, monkeypatch):
        async def fake_analyze(image, settings, **kwargs):
            return ANALYSIS, {}

        monkeypatch.setattr("app.main.analyze_room", fake_analyze)
        body = client.post("/api/analyze", files=upload()).json()
        assert "generation" not in body
        assert body["analysis"]["masks_returned"] == 9
