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


def io_read():
    """The served page, read from disk so a test can inspect its data."""
    import pathlib
    return pathlib.Path("static/showcase.html").read_text(encoding="utf-8")


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

    def test_a_look_is_shown_not_just_named(self, client):
        """Hovering a look shows rooms in that look, beside the pill. It must
        not navigate: browsing eight looks should cost no page loads."""
        html = client.get("/").text
        assert 'id="peek"' in html and "function peek(" in html
        assert "mouseover" in html
        # the popup carries no route of its own
        peek = html[html.index('id="peek"'):html.index('id="veil"')]
        assert "data-go" not in peek and "href" not in peek

    def test_reference_photos_fall_back_to_drawings(self, client):
        """Photographs are not here yet and half a set may arrive first, so the
        fallback is per-image and silent rather than a broken frame."""
        html = client.get("/").text
        assert "photosFor" in html
        assert "photo.onerror" in html
        assert "looks/" in html

    def test_the_studio_takes_more_than_the_six_drawable_rooms(self, client):
        """The reader is a vision model, not a list of six. A stairway or a
        balcony is exactly the awkward space people most want redrawn."""
        html = client.get("/").text
        assert "ROOM_TYPES" in html
        for room in ("Hallway", "Stairway", "Balcony", "Somewhere else"):
            assert room in html

    def test_not_choosing_a_look_is_a_choice(self, client):
        """Two of the options are not looks: one for a room that wants
        resolving rather than restyling, one for someone who would rather
        describe what they want than pick a label."""
        html = client.get("/").text
        assert "NON_STYLES" in html
        assert "Stick to the brief" in html
        assert ">None<" in html or "'None'" in html

    def test_the_comments_box_says_what_it_is_for(self, client):
        html = client.get("/").text
        assert "Extra comments" in html
        assert 'class="hint"' in html

    def test_none_and_brief_reach_the_generator_as_real_prompts(self):
        """They are resolved in the same place as every other style, so no
        caller has to special-case them."""
        from app.generation import STYLES, build_prompt

        assert "none" in STYLES and "brief" in STYLES
        for key in ("none", "brief"):
            prompt = build_prompt(key, extra="More storage.")
            assert "none" not in prompt.split()      # not the bare word
            assert "More storage." in prompt
        assert "no other decorating style" in build_prompt("brief")

    def test_the_room_reaches_the_generator(self, client, monkeypatch):
        """The studio asked which room and then threw the answer away: it only
        ever reached the reader. Picking "nursery" drew a generic bedroom."""
        captured = {}

        async def fake_pipeline(data, style, settings, **kwargs):
            captured.update(kwargs)
            return ANALYSIS, [GENERATION]

        monkeypatch.setattr("app.main.run_pipeline", fake_pipeline)
        client.post("/api/generate", files=upload(),
                    data={"style": "japandi", "room_type": "nursery"})
        assert captured["room"] == "nursery"

    def test_the_page_sends_the_room_when_drawing(self, client):
        html = client.get("/").text
        assert "f.append('room_type', room.id)" in html

    def test_purpose_and_look_are_separate_axes(self):
        """A nursery is not a missing style — it is a set of requirements, and
        it has to be able to be a Japandi one. Kept apart, six looks cover
        sixteen rooms; merged, you would need ninety-six."""
        from app.generation import build_prompt

        japandi_nursery = build_prompt("japandi", room="nursery")
        assert "Japandi" in japandi_nursery
        assert "cot" in japandi_nursery

        industrial_nursery = build_prompt("industrial", room="nursery")
        assert "brick" in industrial_nursery      # the look still applies
        assert "cot" in industrial_nursery        # so do the requirements

    def test_a_nursery_carries_what_no_style_implies(self):
        """None of these are aesthetic choices, which is exactly why they
        cannot live in the look."""
        from app.generation import build_prompt

        prompt = build_prompt("japandi", room="nursery").lower()
        for requirement in ("cot", "blackout", "cords", "above the cot"):
            assert requirement in prompt

    def test_every_offered_room_has_a_brief(self):
        """The dropdown and the briefs are two lists that must not drift. A
        room offered without one silently draws a generic room."""
        import re
        from app.generation import ROOM_BRIEFS

        html = io_read()
        block = html[html.index("var ROOM_TYPES = ["):]
        block = block[:block.index("];")]
        offered = set(re.findall(r"id: '([a-z-]+)'", block))

        # "other" deliberately has none: the comments box carries it instead.
        assert offered - {"other"} <= set(ROOM_BRIEFS)

    def test_an_unknown_room_is_simply_quiet(self):
        """Someone picking "somewhere else" should get a clean prompt, not the
        word "None" wedged into a sentence."""
        from app.generation import build_prompt

        for room in ("other", "", "spaceship"):
            prompt = build_prompt("japandi", room=room)
            assert "It must work as" not in prompt

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

    def test_a_professionally_shot_room_is_still_a_room(self):
        """Real failure: a genuine, furnished living room — shallow depth of
        field, moody lighting, the kind of photo a listing or a magazine
        would use — was rejected as "not a room" while the model's own
        subject line described it as "a living room with a sofa and coffee
        table". Describing furniture and then calling the result not-a-room
        is the model contradicting itself, and the prompt now says so."""
        from app.reading import SURVEY

        prompt = SURVEY.lower()
        assert "shallow depth of field" in prompt
        assert "contradiction" in prompt
        assert "when genuinely unsure, it counts" in prompt

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
