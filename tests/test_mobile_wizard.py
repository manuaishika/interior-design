"""The phone asks one question at a time, and shows results first.

Under 760px the studio controls become cards: home or business, how far,
which room, photos, a look, then a summary with the button. The laptop page
must not change at all, so the facts pinned here are mostly about what is
*scoped away* from wide screens. There is no JS test runner in this repo (see
test_progressive_draw.py), so this pins the source; the behaviour itself was
driven on an emulated phone, and the desktop page was compared pixel for
pixel before and after.
"""

import pathlib
import re


def page(path="static/showcase.html"):
    return pathlib.Path(path).read_text(encoding="utf-8")


def phone_css(source: str) -> str:
    """The body of the question-card @media (max-width: 760px) block."""
    start = source.index("question cards (phone only)")
    block = source.index("@media (max-width: 760px) {", start)
    end = source.index("\n  }\n", block)
    return source[block:end]


class TestTheLaptopPageIsUntouched:
    def test_every_step_is_transparent_on_wide_screens(self):
        assert ".step { display: contents; }" in page()

    def test_the_new_pieces_are_hidden_outside_the_phone_query(self):
        source = page()
        base = source[source.index(".step { display: contents; }"):]
        base = base[:base.index("@media")]
        for hidden in (".step .q", ".step .qhint", ".wiz-nav",
                       ".wiz-sum", '.step[data-g="setting"]'):
            assert hidden in base

    def test_cards_only_hide_things_once_the_script_asks_for_it(self):
        """Hiding hangs off `.controls.wiz`, a class the script adds. If the
        script dies, a phone gets the plain form, never a blank one."""
        css = phone_css(page())
        assert ".controls.wiz .step { display: none;" in css
        assert ".controls.wiz .step.cur { display: block; }" in css
        assert re.search(r"^\s*\.controls \.step \{ display: none", css, re.M) is None


class TestTheQuestions:
    def test_six_cards_in_order_with_home_or_business_first(self):
        assert ("var WIZ_STEPS = ['setting', 'depth', 'room', 'photos', "
                "'look', 'final'];") in page()

    def test_markup_has_a_group_per_question(self):
        source = page()
        for g in ("setting", "depth", "photos", "room", "look", "final"):
            assert f'data-g="{g}"' in source

    def test_the_room_is_a_dropdown_whose_list_follows_home_or_business(self):
        """A study can be a home office or an office, a studio a flat or a
        space — so the list depends on the first answer, and it is a plain
        dropdown, not a wall of options."""
        source = page()
        assert "var HOME_ROOMS = [" in source
        assert "var BUSINESS_ROOMS = [" in source
        assert "wiz.setting === 'business' ? BUSINESS_ROOMS : HOME_ROOMS" in source
        assert "roomChips" not in source
        assert ".controls.wiz #roomSel { display: block;" in phone_css(source)

    def test_every_business_room_has_a_brief_on_the_server(self):
        import re
        from app.generation import ROOM_BRIEFS

        source = page()
        block = source[source.index("var BUSINESS_ROOMS = ["):]
        block = block[:block.index("];")]
        ids = set(re.findall(r"id: '([a-z-]+)'", block))
        assert ids - {"other"} <= set(ROOM_BRIEFS)
        block = source[source.index("var HOME_ROOMS = ["):]
        block = block[:block.index("];")]
        ids = set(re.findall(r"id: '([a-z-]+)'", block))
        assert ids - {"other"} <= set(ROOM_BRIEFS)

    def test_the_laptop_room_list_is_the_old_one(self):
        """Wide screens keep ROOM_TYPES, retail included."""
        source = page()
        assert "{ id: 'retail',      name: 'Shop, café or salon' }," in source

    def test_the_cards_drive_the_same_state_as_the_form(self):
        """Choosing a room sets the real <select>, so a phone and a laptop
        send identical requests."""
        source = page()
        assert "$('roomSel').value = id;" in source
        assert "$('roomSel').onchange();" in source

    def test_next_is_never_a_dead_button(self):
        source = page()
        body = source[source.index("$('wizNext').onclick"):]
        body = body[:body.index("if (MQ)")]
        assert "!haveEnough()" in body
        assert "say(" in body and "return;" in body
        assert "disabled" not in body

    def test_messages_reach_the_visible_card(self):
        """#msg lives on the last card only, so say() also writes #wizMsg."""
        source = page()
        body = source[source.index("function say("):]
        body = body[:body.index("\n  }\n")]
        assert "$('wizMsg').textContent" in body

    def test_answers_are_kept_but_only_by_the_phone_flow(self):
        source = page()
        assert "var WIZ_KEY = 'second-draft-answers';" in source
        body = source[source.index("function wizSave()"):]
        body = body[:body.index("function wizSetRoom")]
        assert "if (!wiz.on || wiz.loading) return;" in body
        assert "try {" in body   # a private window must not break the page


class TestResultsComeFirst:
    def test_results_move_above_the_questions_on_a_phone(self):
        assert ".panel-body.has-run .pane { order: -1;" in phone_css(page())

    def test_the_page_does_not_scroll_past_the_designs_as_they_land(self):
        assert "overflow-anchor: none" in phone_css(page())

    def test_a_run_and_a_reopened_design_both_mark_it(self):
        source = page()
        go = source[source.index("$('go').onclick = async function"):]
        go = go[:go.index("$('go').disabled = true;")]
        assert "markRun();" in go
        shown = source[source.index("function showResults()"):]
        assert "markRun();" in shown[:shown.index("}")]

    def test_the_action_buttons_wrap_instead_of_leaving_the_screen(self):
        assert ".shot-row { flex-wrap: wrap; }" in phone_css(page())


class TestDocsCopyMatches:
    def test_the_same_cards_exist_there(self):
        source = page("docs/index.html")
        assert "var WIZ_STEPS = [" in source
        assert ".panel-body.has-run .pane { order: -1;" in source
