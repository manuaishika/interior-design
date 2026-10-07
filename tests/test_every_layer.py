"""A full redesign must not forget a layer.

Reported: a full redesign gave a new false ceiling, a TV console and a floor
lamp, and left the walls exactly as they were. The plan had asked for walls
only as a colour, the drawing took whichever changes it liked, and nothing
checked. Now the plan has a mandatory slot for every layer of a space (and a
real design, not just paint, for the walls), whatever the planner leaves blank
is filled with a default instruction, large and commercial spaces are covered
zone by zone, and the self-check flags a layer that came back unchanged.
"""

import pytest

from app import checking, planning
from app.generation import COMMERCIAL_ROOMS, DEPTHS, ROOM_BRIEFS, build_prompt

BARE = {"replace": [{"current": "old sofa", "new": "a rust boucle sofa"}],
        "add": [{"piece": "a TV console", "where": "on the left wall"}],
        "finishes": {"ceiling": "a false ceiling with cove lighting"},
        "zones": [], "fixed": []}

LABELS = [label for _, label in planning.LAYERS]


class TestNoLayerIsEverDropped:
    def test_the_layers_are_the_whole_space(self):
        assert [k for k, _ in planning.LAYERS] == [
            "walls", "wall_features", "wall_decor", "ceiling", "floor",
            "lighting", "window_dressing", "textiles"]

    def test_a_plan_that_only_did_the_ceiling_still_instructs_every_layer(self):
        text = planning.render_plan(BARE)
        for label in LABELS:
            assert f"- {label}:" in text
        assert "a false ceiling with cove lighting" in text      # its own words win
        assert "the walls change by design, not only by colour" in text

    def test_walls_default_to_a_real_design_not_a_colour(self):
        text = planning.render_plan(BARE)
        assert "fluted or slatted wood" in text
        assert "not just paint" in text

    def test_a_commercial_space_gets_a_branded_feature_wall(self):
        home = planning.render_plan(BARE)
        shop = planning.render_plan(BARE, commercial=True)
        assert "carries the brand" in shop and "carries the brand" not in home

    def test_the_planner_is_told_every_layer_is_mandatory(self):
        for key, _ in planning.LAYERS:
            assert key in planning.PLANNER
        assert "A layer left as it is now counts as a failure" in planning.PLANNER
        assert "a colour alone is not a design" in planning.PLANNER

    def test_which_layers_the_planner_skipped_is_known(self):
        assert planning.missing_layers(BARE)[0] == "walls"
        assert "ceiling" not in planning.missing_layers(BARE)


class TestLargeSpacesAreCoveredZoneByZone:
    def test_the_planner_is_asked_for_zones_in_a_large_space(self):
        assert "LARGE space" in planning.PLANNER and '"zones"' in planning.PLANNER

    def test_zones_survive_cleaning_and_reach_the_drawing(self):
        plan = planning._clean({
            "replace": [{"current": "desks", "new": "oak workstations"}],
            "zones": [{"zone": "reception", "changes": "a curved walnut desk"},
                      {"zone": "", "changes": "x"}]})
        assert plan["zones"] == [{"zone": "reception", "changes": "a curved walnut desk"}]
        assert "- Reception: a curved walnut desk." in planning.render_plan(plan)

    def test_every_commercial_room_has_a_brief(self):
        assert COMMERCIAL_ROOMS <= set(ROOM_BRIEFS)

    def test_the_wording_covers_a_large_room_not_just_the_near_part(self):
        assert "not just the part nearest the camera" in DEPTHS["renovate"]


class TestEvenWithoutAPlan:
    def test_the_instruction_alone_names_every_layer(self):
        text = build_prompt("japandi", depth="renovate")
        for word in ("the ceiling", "the walls", "the floor", "the lighting",
                     "the curtains", "the soft furnishings"):
            assert word in text
        assert "Doing the ceiling and the TV unit and leaving the walls as they were is half a redesign" in text


CLEAN = {"same_viewpoint": True, "structure_ok": True, "counts_ok": True,
         "unchanged_pieces": [], "unchanged_layers": [], "invented": [], "problems": []}


class TestTheCheckCatchesAForgottenLayer:
    def test_unchanged_walls_are_a_fault_on_their_own(self):
        faults = checking.verdict({**CLEAN, "unchanged_layers": ["walls"]}, "renovate")
        assert len(faults) == 1
        assert "walls look unchanged" in faults[0] and "not just a new colour" in faults[0]

    def test_one_other_layer_is_taste_two_is_a_fault(self):
        assert checking.verdict({**CLEAN, "unchanged_layers": ["floor"]}, "renovate") == []
        faults = checking.verdict({**CLEAN, "unchanged_layers": ["floor", "lighting"]}, "renovate")
        assert faults and "floor, lighting" in faults[0]

    def test_a_light_restyle_is_not_held_to_it(self):
        assert checking.verdict({**CLEAN, "unchanged_layers": ["walls", "floor", "ceiling"]}, "restyle") == []

    def test_the_judge_is_asked_about_layers(self):
        assert "unchanged_layers" in checking.JUDGE
        assert '"walls" (colour AND any design' in checking.JUDGE


class TestTheWhatChangedList:
    def test_layers_are_listed_with_the_pieces(self):
        plan = {**BARE, "finishes": {"walls": "fluted oak behind the bed", "ceiling": "cove ceiling"}}
        lines = planning.changes_of(plan)
        assert "Walls: fluted oak behind the bed" in lines
        assert "Ceiling: cove ceiling" in lines
        assert "old sofa → a rust boucle sofa" in lines
