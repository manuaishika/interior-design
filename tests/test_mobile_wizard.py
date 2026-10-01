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


def phone_sizing(source: str) -> str:
    start = source.index("phone sizing ----------")
    block = source.index("@media (max-width: 760px) {", start)
    return source[block:source.index("</style>", block)]


class TestPhoneScale:
    """A phone used to get the laptop's sizes in one long column."""

    def test_it_is_the_last_rule_and_phone_only(self):
        source = page()
        css = phone_sizing(source)
        assert css.startswith("@media (max-width: 760px) {")
        assert source.index("phone sizing ----------") > source.index("question cards (phone only)")

    def test_pictures_go_two_across(self):
        css = phone_sizing(page())
        assert ".grid { column-count: 2;" in css
        assert "grid-template-columns: repeat(2, minmax(0, 1fr))" in css

    def test_inputs_stay_at_16px_so_ios_does_not_zoom_in(self):
        css = page()
        assert ".askrow input { font-size: 16px; }" in phone_sizing(css)
        assert "textarea.brief { min-height: 70px; font-size: 16px; }" in phone_css(css)


class TestCompareStaysOnThePicture:
    def test_the_drag_surface_is_the_image_only(self):
        source = page()
        body = source[source.index("function openCompare("):]
        body = body[:body.index("$('compareRange')")]
        assert "frame.appendChild(overlay)" in body
        assert "pointerdown" in body and "setPointerCapture" in body

    def test_a_vertical_swipe_still_scrolls_the_page(self):
        assert "touch-action: pan-y" in page()

    def test_the_hidden_slider_no_longer_covers_anything(self):
        source = page()
        rule = source[source.index(".shots figure .overlay input[type=range] {"):]
        rule = rule[:rule.index("}")]
        assert "pointer-events: none" in rule
        assert "inset: 0" not in rule

    def test_the_button_says_how_to_get_out(self):
        assert "cmp.textContent = 'Close compare';" in page()


class TestThePhoneStudioIsQuiet:
    """What a phone showed that it should not have."""

    def test_the_empty_results_panel_waits_for_a_run(self):
        css = phone_sizing(page())
        assert ".panel-body:not(.has-run) .pane { display: none; }" in css

    def test_the_conversation_waits_for_a_design(self):
        source = page()
        assert ".panel-body:not(.has-design) .convo-bar { display: none; }" in phone_sizing(source)
        body = source[source.index("function appendDesign("):]
        assert "classList.add('has-design')" in body[:900]

    def test_the_engine_line_no_longer_wraps_over_the_panel_title(self):
        css = phone_sizing(page())
        assert ".panel-top, .panel-top > span:first-child, #state { display: none; }" in css

    def test_the_results_scroll_with_the_page_not_in_a_box_of_their_own(self):
        css = phone_sizing(page())
        assert ".pane-scroll { flex: none; overflow: visible; }" in css

    def test_the_studio_intro_is_one_short_line_on_a_phone(self):
        source = page()
        assert '<span\n       class="lede-more">' in source
        assert ".lede-more { display: none; }" in phone_sizing(source)
        # and the wide page keeps the whole sentence
        assert "listed underneath, so" in source


class TestMarkupSaysWhatToDoNext:
    def body(self):
        source = page()
        start = source.index("function startMarking(")
        return source[start:source.index("function designFeedback", start)]

    def test_three_steps_are_named_and_the_current_one_is_marked(self):
        body = self.body()
        assert "'1  Circle it', '2  Say what', '3  Tap Change'" in body
        assert "el.classList.toggle('on', k === at);" in body

    def test_the_next_thing_to_do_pulses(self):
        body = self.body()
        assert "input.classList.toggle('nudge', at === 1);" in body
        assert "go.classList.toggle('nudge', at === 2);" in body
        assert "@keyframes nudge" in page()

    def test_a_tip_sits_on_the_picture_until_the_first_stroke(self):
        body = self.body()
        assert "tip.textContent = 'Draw a circle around it';" in body
        assert "painted = true; tip.remove(); guide();" in body

    def test_change_is_never_a_dead_button(self):
        body = self.body()
        assert "go.disabled" not in body
        assert "First draw a circle on the picture" in body
        assert "input.focus();" in body

    def test_the_result_is_scrolled_to_and_flashes(self):
        body = self.body()
        assert "landed.scrollIntoView(" in body
        assert "landed.classList.add('landed')" in body

    def test_compare_on_an_edit_shows_the_design_it_came_from(self):
        assert "b64ToBlob(g.image_base64));" in self.body()

    def test_a_failed_edit_is_said_next_to_the_picture(self):
        body = self.body()
        assert "note.classList.add('bad');" in body
        assert "say((e && e.message)" not in body   # #msg is out of sight on a phone


class TestTheLaptopStudioFitsOneScreen:
    def block(self, query):
        source = page()
        start = source.index("the studio on a laptop: one screen")
        rest = source[start:]
        at = rest.index(query)
        return rest[at:rest.index("\n  }\n", at)]

    def test_the_panel_takes_the_rest_of_the_screen(self):
        css = self.block("@media (min-width: 860px) {")
        assert "#v-studio .panel-body { height: calc(100vh" in css
        assert "#v-studio .controls { overflow-y: auto;" in css

    def test_the_button_stays_in_reach(self):
        assert "#v-studio .controls #go { position: sticky; bottom: 0;" in self.block("@media (min-width: 860px) {")

    def test_wide_screens_put_the_questions_in_two_independent_columns(self):
        """Two flex columns, not grid rows: with rows, a tall look list on the
        right stretched the photos row on the left and left a gap."""
        css = self.block("@media (min-width: 1100px) {")
        assert "#v-studio .controls .col { display: flex; flex-direction: column;" in css
        assert "flex-direction: row;" in css
        assert '.step[data-g="setting"] { display: none; }' in css
        assert ".col { display: contents; }" in page()

    def test_a_short_laptop_screen_still_fits_without_scrolling(self):
        css = self.block("@media (min-width: 860px) and (max-height: 780px) {")
        assert "min-height: 430px;" in css
        assert "#lookHint" in css

    def test_the_header_lines_up_with_the_wider_studio(self):
        css = self.block("@media (min-width: 1100px) {")
        assert "body:has(#v-studio:not(.hidden)) .top .wrap { max-width: 1440px; }" in css

    def test_none_of_it_reaches_a_phone(self):
        source = page()
        start = source.index("the studio on a laptop: one screen")
        assert "max-width" not in source[start:source.index("{", start)]


class TestSignInIsQuietForNow:
    def test_the_button_and_the_save_prompt_are_hidden_while_accounts_are_off(self):
        source = page()
        assert "var SIGN_IN_OFF = true;" in source
        assert "$('login').classList.toggle('hidden', !signInVisible());" in source
        assert "keep.classList.add('hidden')" in source

    def test_an_access_code_studio_still_shows_it(self):
        assert "return !SIGN_IN_OFF || needsCode || Boolean(account);" in page()

    def test_the_pricing_page_stops_promising_a_sign_in_difference(self):
        source = page()
        assert "Up to 12 designs a day, no account needed" in source
        assert "when signed in, 2 without" not in source
