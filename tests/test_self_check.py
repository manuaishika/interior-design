"""The agent looks at each design before the client does.

A finished design is shown next to the original; if a hard rule broke it is
redrawn once with the faults named. These tests pin what counts as a fault,
what the retry does, and that none of it can stop a design being delivered.
"""

import asyncio
import io

import pytest
from PIL import Image

from app import checking, openai_images, planning
from app.config import Settings
from app.pipeline import run_pipeline


def png():
    buf = io.BytesIO()
    Image.new("RGB", (64, 48), (120, 110, 100)).save(buf, "PNG")
    return buf.getvalue()


CLEAN = {"same_viewpoint": True, "structure_ok": True, "counts_ok": True,
         "unchanged_pieces": [], "invented": [], "problems": []}


class TestWhatCountsAsAFault:
    def test_a_clean_design_passes(self):
        assert checking.verdict(CLEAN, "renovate") == []
        assert checking.verdict(None, "renovate") == []
        assert checking.verdict("nonsense", "renovate") == []

    @pytest.mark.parametrize("field,word", [
        ("same_viewpoint", "camera"), ("structure_ok", "door, window"),
        ("counts_ok", "number of beds")])
    def test_each_hard_rule_is_a_fault(self, field, word):
        faults = checking.verdict({**CLEAN, field: False}, "restyle")
        assert len(faults) == 1 and word in faults[0]

    def test_something_invented_is_a_fault(self):
        faults = checking.verdict({**CLEAN, "invented": ["a second bed"]}, "restyle")
        assert faults == ["Remove: a second bed."]

    def test_a_full_redesign_that_kept_the_old_furniture_is_a_fault(self):
        raw = {**CLEAN, "unchanged_pieces": ["the white desk", "the grey chair"]}
        faults = checking.verdict(raw, "renovate")
        assert faults and "still the old pieces" in faults[0]

    def test_one_unchanged_piece_is_taste_not_a_fault(self):
        assert checking.verdict({**CLEAN, "unchanged_pieces": ["the rug"]}, "renovate") == []

    def test_a_light_restyle_is_expected_to_keep_the_furniture(self):
        raw = {**CLEAN, "unchanged_pieces": ["the desk", "the chair", "the bed"]}
        assert checking.verdict(raw, "restyle") == []

    def test_free_text_problems_alone_do_not_trigger_a_redraw(self):
        assert checking.verdict({**CLEAN, "problems": ["curtains are ugly"]}, "renovate") == []

    def test_but_they_ride_along_once_something_hard_failed(self):
        raw = {**CLEAN, "counts_ok": False, "problems": ["Keep one bed only"]}
        assert "Keep one bed only." in checking.verdict(raw, "renovate")


class TestTheRedrawPrompt:
    def test_faults_go_before_the_clients_words_which_stay_last(self):
        out = checking.with_faults("Do it.\n\nThe client asks: a reading corner",
                                   ["Keep one bed."])
        assert out.index("Keep one bed.") < out.index("The client asks:")
        assert out.endswith("a reading corner")

    def test_without_client_words_they_are_appended(self):
        assert checking.with_faults("Do it.", ["Keep one bed."]).endswith("- Keep one bed.")


def run(coro):
    return asyncio.run(coro)


class TestItNeverBlocksADelivery:
    def test_it_can_be_switched_off(self, monkeypatch):
        called = []

        async def ask(*a, **k):
            called.append(1)
            return CLEAN

        monkeypatch.setattr(planning, "ask_json", ask)
        out = run(checking.check(png(), png(), depth="renovate", plan_text="",
                                 settings=Settings(openai_api_key="k", self_check=False)))
        assert out == [] and not called

    def test_a_failing_checker_means_ship_it(self, monkeypatch):
        async def boom(*a, **k):
            raise RuntimeError("down")

        monkeypatch.setattr(planning, "ask_json", boom)
        assert run(checking.check(png(), png(), depth="renovate", plan_text="",
                                  settings=Settings(openai_api_key="k"))) == []


class TestThroughThePipeline:
    def setup_pipeline(self, monkeypatch, verdicts, draw_errors=()):
        """Real run_pipeline; only the paid calls are stubbed."""
        drawn = []
        errors = list(draw_errors)

        async def redraw(image, mask, prompt, s, **kw):
            drawn.append(prompt)
            if errors:
                raise errors.pop(0)
            return png()

        asked = []

        async def ask(prompt, photos, settings, seed=0, max_tokens=0):
            asked.append(len(photos))
            return verdicts.pop(0) if verdicts else CLEAN

        async def find(image, s):
            return []

        async def nap(_):
            return None

        monkeypatch.setattr(openai_images, "redraw", redraw)
        monkeypatch.setattr(openai_images, "find_structure", find)
        monkeypatch.setattr(planning, "ask_json", ask)
        monkeypatch.setattr(asyncio, "sleep", nap)
        return drawn, asked

    def go(self, depth="restyle", **kw):
        return run(run_pipeline(
            png(), "japandi", Settings(openai_api_key="k", **kw), variants=1,
            room="bedroom", extra_prompt="a reading corner", profile=depth,
            tier="room"))

    def test_a_clean_design_is_drawn_once(self, monkeypatch):
        drawn, asked = self.setup_pipeline(monkeypatch, [CLEAN])
        _, gens = self.go()
        assert len(drawn) == 1
        assert asked == [2]                   # original and result, side by side
        assert gens[0].checked == ""

    def test_a_broken_rule_gets_one_redraw_with_the_fault_named(self, monkeypatch):
        drawn, _ = self.setup_pipeline(
            monkeypatch, [{**CLEAN, "counts_ok": False, "invented": ["a second bed"]}])
        _, gens = self.go()
        assert len(drawn) == 2
        assert "Remove: a second bed." in drawn[1]
        assert "Remove: a second bed." not in drawn[0]
        assert drawn[1].rstrip().endswith("The client asks: a reading corner")
        assert gens[0].checked.startswith("Redrawn once to fix:")

    def test_it_never_loops(self, monkeypatch):
        bad = {**CLEAN, "counts_ok": False}
        drawn, asked = self.setup_pipeline(monkeypatch, [bad, bad, bad])
        self.go()
        assert len(drawn) == 2 and len(asked) == 1

    def test_if_the_redraw_fails_the_first_design_is_kept(self, monkeypatch):
        drawn, _ = self.setup_pipeline(
            monkeypatch, [{**CLEAN, "counts_ok": False}],
            draw_errors=[])
        # first draw succeeds, second raises
        calls = {"n": 0}
        real = openai_images.redraw

        async def flaky(*a, **k):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("boom")
            return await real(*a, **k)

        monkeypatch.setattr(openai_images, "redraw", flaky)
        _, gens = self.go()
        assert len(gens) == 1 and gens[0].checked == ""

    def test_a_transient_failure_is_retried_once(self, monkeypatch):
        drawn, _ = self.setup_pipeline(
            monkeypatch, [CLEAN],
            draw_errors=[openai_images.OpenAIImageError("No picture came back.")])
        _, gens = self.go()
        assert len(drawn) == 2 and len(gens) == 1

    def test_a_billing_or_verification_failure_is_not_retried(self, monkeypatch):
        drawn, _ = self.setup_pipeline(
            monkeypatch, [CLEAN],
            draw_errors=[openai_images.OpenAIImageError(
                "That key's organisation has no credit on it.")])
        with pytest.raises(openai_images.OpenAIImageError):
            self.go()
        assert len(drawn) == 1

    def test_the_structure_call_is_skipped_unless_the_mask_is_sent(self, monkeypatch):
        self.setup_pipeline(monkeypatch, [CLEAN])
        found = []

        async def find(image, s):
            found.append(1)
            return []

        monkeypatch.setattr(openai_images, "find_structure", find)
        self.go()
        assert found == []
        self.go(use_inpaint_mask=True)
        assert found == [1]

    def test_a_full_redesign_reports_what_it_changed(self, monkeypatch):
        self.setup_pipeline(monkeypatch, [CLEAN])
        plan = {"replace": [{"current": "white desk", "new": "a walnut desk"},
                            {"current": "laundry basket", "new": "removed"}],
                "add": [{"piece": "an oak wardrobe", "where": "on the right wall"}],
                "finishes": {}, "fixed": []}

        async def fake(*a, **k):
            return plan

        monkeypatch.setattr(planning, "_ask", fake)
        _, gens = self.go(depth="renovate")
        assert gens[0].changes == ["white desk → a walnut desk",
                                   "laundry basket: removed",
                                   "Added an oak wardrobe, on the right wall"]
