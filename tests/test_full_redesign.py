"""A full redesign has to actually redesign.

The complaint: Full redesign came back as the same furniture, cleaner. Causes
found and pinned here:

1. The studio ticks every item it finds as "must stay", and that list was sent
   as-is, telling the image model the room "must still contain" the bed, desk
   and chair (replaced at most by better versions of the same thing).
2. The instruction to go the whole way came last, after a long list of what
   must not change, and never said what the new pieces should be.
3. Extra photographs were sent with no explanation, so the old furniture in
   them was a second copy to imitate.

The fix is a plan made before drawing (app/planning.py) plus wording and
keep-handling in app/generation.py. Nothing here touches the page.
"""

import asyncio
import io

import pytest
from PIL import Image

from app import planning
from app.config import Settings
from app.generation import (build_prompt, effective_keep, is_structural,
                            DEPTHS)

ITEMS = [{"name": "bed", "count": 1}, {"name": "desk", "count": 1},
         {"name": "chair", "count": 1}, {"name": "window", "count": 1},
         {"name": "door", "count": 1}]
EVERYTHING = "bed, desk, chair, window, door"


def jpeg(color):
    buf = io.BytesIO()
    Image.new("RGB", (64, 48), color).save(buf, "JPEG")
    return buf.getvalue()


class TestTheDefaultNoLongerBlocksTheRedesign:
    def test_everything_ticked_means_only_the_structure_stays(self):
        assert effective_keep(EVERYTHING, ITEMS, "", "renovate") == "window, door"

    def test_a_deliberate_choice_is_respected_exactly(self):
        """Someone who unticked the bed but kept the desk meant it."""
        assert effective_keep("desk, window, door", ITEMS, "", "renovate") \
            == "desk, window, door"

    def test_a_light_restyle_still_keeps_everything(self):
        assert effective_keep(EVERYTHING, ITEMS, "", "restyle") == EVERYTHING

    def test_it_works_from_the_contents_text_when_there_are_no_items(self):
        assert effective_keep("bed, desk", [], "bed, 2 desk", "renovate") == ""
        assert effective_keep("bed, desk", [], "bed, desk", "renovate") == ""

    @pytest.mark.parametrize("name,structural", [
        ("window", True), ("door", True), ("air conditioner", True),
        ("AC", True), ("ceiling fan", True), ("built-in wardrobe", True),
        ("radiator", True), ("desk", False), ("bed", False),
        ("accent chair", False), ("wall art", False), ("bath mat", False),
    ])
    def test_what_counts_as_the_building(self, name, structural):
        assert is_structural(name) is structural

    def test_a_kept_piece_is_kept_exactly_in_a_full_redesign(self):
        prompt = build_prompt("japandi", keep="desk, window", depth="renovate")
        assert "Keep exactly as they are: desk, window" in prompt
        assert "better versions of the same thing" not in prompt


class TestTheInstructionLeadsAndSaysWhatFailureIs:
    def test_the_job_comes_before_the_limits(self):
        prompt = build_prompt("japandi", depth="renovate")
        assert prompt.index("FULL REDESIGN") < prompt.index("recognisably the same room")

    def test_polish_is_named_as_failing(self):
        text = DEPTHS["renovate"]
        assert "only cleaner, brighter, shinier" in text
        assert "has FAILED this instruction" in text
        assert "different silhouettes, different materials, different colours" in text

    def test_a_nearly_empty_room_is_furnished(self):
        assert "If the room is nearly empty, furnish it fully." in DEPTHS["renovate"]

    def test_a_light_restyle_is_unchanged(self):
        prompt = build_prompt("japandi", depth="restyle")
        assert "Do not replace the pieces themselves" in prompt
        assert "FULL REDESIGN" not in prompt

    def test_other_photos_are_explained_not_imitated(self):
        prompt = build_prompt("japandi", depth="renovate", views=2)
        assert "Do not copy the old furniture from them" in prompt
        assert "Several photographs are supplied" not in build_prompt("japandi", depth="renovate")


class TestThePlan:
    def test_it_is_keyed_on_the_photos_not_on_which_is_primary(self):
        a, b, c = jpeg("red"), jpeg("green"), jpeg("blue")
        args = ("japandi", "bedroom", "renovate", "", "", 0)
        assert planning.plan_key([a, b, c], *args) == planning.plan_key([c, a, b], *args)
        assert planning.plan_key([a, b], *args) != planning.plan_key([a, c], *args)

    def test_a_different_option_is_a_different_plan(self):
        a = jpeg("red")
        assert planning.plan_key([a], "j", "b", "renovate", "", "", 0) \
            != planning.plan_key([a], "j", "b", "renovate", "", "", 1)

    def test_junk_is_dropped_and_an_empty_plan_is_none(self):
        assert planning._clean({"replace": [{"current": "", "new": "x"}]}) is None
        assert planning._clean("nope") is None
        good = planning._clean({"replace": [{"current": "old bed", "new": "oak bed"}],
                                "finishes": {"walls": "sand", "bogus": "x"}})
        assert good["replace"] == [{"current": "old bed", "new": "oak bed"}]
        assert good["finishes"] == {"walls": "sand"}

    def test_it_reads_as_instructions(self):
        text = planning.render_plan({
            "replace": [{"current": "white desk", "new": "a walnut desk"},
                        {"current": "laundry basket", "new": "removed"}],
            "add": [{"piece": "a 2m oak wardrobe", "where": "on the right wall"}],
            "finishes": {"walls": "limewash"}, "fixed": ["window on the back wall"]})
        assert "Take out the white desk. In its place: a walnut desk." in text
        assert "Remove the laundry basket." in text
        assert "Add a 2m oak wardrobe, on the right wall." in text
        assert "Leave exactly where they are: window on the back wall." in text
        assert planning.render_plan(None) == ""


@pytest.fixture(autouse=True)
def fresh_cache():
    planning._cache.clear()
    planning._inflight.clear()
    yield
    planning._cache.clear()


def run(coro):
    return asyncio.run(coro)


PLAN = {"replace": [{"current": "bed", "new": "oak bed"}], "add": [],
        "finishes": {}, "fixed": []}


class TestPlanningNeverBreaksDrawing:
    def test_a_failure_returns_none_and_is_not_remembered(self, monkeypatch):
        calls = []

        async def boom(*a, **k):
            calls.append(1)
            raise RuntimeError("rate limited")

        monkeypatch.setattr(planning, "_ask", boom)
        kw = dict(style_text="s", room_text="r", key_parts=("s", "r", "renovate", ""),
                  settings=Settings(openai_api_key="k"))
        assert run(planning.make_plan([jpeg("red")], **kw)) is None
        assert run(planning.make_plan([jpeg("red")], **kw)) is None
        assert len(calls) == 2          # tried again rather than caching the failure

    def test_a_success_is_shared_across_angles(self, monkeypatch):
        calls = []

        async def ask(prompt, photos, settings, seed):
            calls.append(len(photos))
            await asyncio.sleep(0.05)
            return PLAN

        monkeypatch.setattr(planning, "_ask", ask)
        a, b = jpeg("red"), jpeg("green")
        kw = dict(style_text="s", room_text="r", key_parts=("s", "r", "renovate", ""),
                  settings=Settings(openai_api_key="k"))

        async def both():
            # two angle requests of one room, arriving together, photos in
            # different order, each with the other as its reference
            return await asyncio.gather(
                planning.make_plan([a, b], **kw), planning.make_plan([b, a], **kw))

        one, two = run(both())
        assert one == two == PLAN
        assert calls == [2]             # one call, not one per angle


class TestWhatReachesTheImageModel:
    """End to end through the real pipeline, with only the paid calls stubbed."""

    def _run(self, monkeypatch, depth, keep=EVERYTHING, plan=PLAN):
        from app import openai_images
        from app.pipeline import run_pipeline

        seen = {}

        async def find(image, s):
            return []

        async def redraw(image, mask, prompt, s, **kw):
            seen["prompt"] = prompt
            seen["references"] = kw.get("references")
            buf = io.BytesIO()
            Image.new("RGB", (8, 8)).save(buf, "PNG")
            return buf.getvalue()

        async def fake_plan(*a, **k):
            return plan

        monkeypatch.setattr(openai_images, "find_structure", find)
        monkeypatch.setattr(openai_images, "redraw", redraw)
        monkeypatch.setattr(planning, "_ask", lambda *a, **k: fake_plan())

        run(run_pipeline(jpeg("red"), "japandi", Settings(openai_api_key="k"),
                         extra_prompt="a reading corner", variants=1,
                         room="bedroom", contents="a bed, a desk, a chair",
                         keep=keep, items=ITEMS, profile=depth,
                         references=[jpeg("green")], tier="room"))
        return seen

    def test_a_full_redesign_carries_the_plan_and_drops_the_blanket_keep(self, monkeypatch):
        seen = self._run(monkeypatch, "renovate")
        prompt = seen["prompt"]
        assert "Take out the bed. In its place: oak bed." in prompt
        assert "Keep exactly as they are: window, door" in prompt
        assert "bed, desk, chair" not in prompt.split("Keep exactly as they are:")[1][:40]
        assert "Do not copy the old furniture" in prompt
        assert prompt.rstrip().endswith("The client asks: a reading corner")
        assert len(seen["references"]) == 1

    def test_a_light_restyle_makes_no_plan_and_keeps_everything(self, monkeypatch):
        seen = self._run(monkeypatch, "restyle")
        assert "THE REDESIGN" not in seen["prompt"]
        assert "must still contain bed, desk, chair, window, door" in seen["prompt"]

    def test_drawing_still_works_when_planning_fails(self, monkeypatch):
        seen = self._run(monkeypatch, "renovate", plan=None)
        assert "FULL REDESIGN" in seen["prompt"]
        assert "THE REDESIGN" not in seen["prompt"]
