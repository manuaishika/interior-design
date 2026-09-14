"""Tests for FIX-NEXT.md #4: one panel, not a panel and three boxes.

The results pane is now the whole right-hand side. Designs, "What it
found" and "What it would cost" are tabs inside it; the conversation is a
bar pinned along the bottom, always visible regardless of which tab is
open. Every id the rest of the app (and its tests) depend on had to move
rather than be renamed — #shots, #thread, #ask, #askGo, #redraw, #chips,
#cost, #costBlock, #savedBlock — or a lot of unrelated tests fail for no
useful reason.
"""

import pathlib
import re


def page(path="static/showcase.html"):
    return pathlib.Path(path).read_text(encoding="utf-8")


PRESERVED_IDS = ("shots", "thread", "ask", "askGo", "redraw", "chips",
                 "cost", "costBlock", "savedBlock")


class TestNothingWasRenamed:
    def test_every_id_the_tests_rely_on_still_exists(self):
        source = page()
        for element_id in PRESERVED_IDS:
            assert f'id="{element_id}"' in source, f"#{element_id} is gone"

    def test_the_old_three_box_wrapper_is_gone(self):
        """.readout / .box / .talk-head described a panel-plus-three-cards
        layout that no longer exists — dead classes left behind read as
        the fix having not actually happened."""
        source = page()
        assert 'id="readout"' not in source
        assert 'class="talk-head"' not in source


class TestThePaneHoldsAllThreeTabs:
    def test_three_tabs_are_named(self):
        source = page()
        tabs = re.findall(r'data-tab="([\w-]+)"', source)
        assert {"designs", "found", "cost"} <= set(tabs)

    def test_costblock_is_the_cost_tab_not_a_page_section(self):
        """#costBlock kept its id but stopped being a standalone <section> —
        it is now one of the tabs inside #pane."""
        source = page()
        start = source.index('id="costBlock"')
        tag_open = source.rindex("<", 0, start)
        assert 'data-tab="cost"' in source[tag_open:start + 20]

    def test_switching_tabs_is_one_function(self):
        source = page()
        assert "function switchTab(" in source
        assert "tabhidden" in source

    def test_an_empty_cost_tab_hides_its_own_button(self):
        """Nothing worse than a tab that opens onto nothing — drawCost must
        hide the button, not just the content, when there are no lines."""
        source = page()
        start = source.index("function drawCost(")
        end = source.index("\n  }\n", start)
        body = source[start:end]
        assert "costTabBtn" in body


class TestTheConversationIsPinnedToTheBottom:
    def test_a_single_convo_bar_holds_thread_and_input(self):
        source = page()
        start = source.index('id="convoBar"')
        end = source.index("</div>", source.index('id="askGo"'))
        block = source[start:end]
        assert 'id="thread"' in block
        assert 'id="ask"' in block
        assert 'id="askGo"' in block
        assert 'id="redraw"' in block

    def test_it_renders_last_despite_reading_first_in_source(self):
        """test_talking_comes_before_the_inventory pins that "Talk it
        through" precedes "What it found" in the HTML — that has to stay
        true even though the conversation now *renders* at the bottom.
        The visual order comes from CSS `order`, not document position."""
        source = page()
        assert source.index("Talk it through") < source.index("What it found")

        css = re.search(r"<style>(.*?)</style>", source, re.S).group(1)
        convo_order = re.search(r"\.convo-bar\s*\{[^}]*order:\s*(\d+)", css)
        tabs_order = re.search(r"\.pane-tabs\s*\{[^}]*order:\s*(\d+)", css)
        assert convo_order and tabs_order
        assert int(convo_order.group(1)) > int(tabs_order.group(1))

    def test_the_bar_is_always_visible_once_a_room_is_read(self):
        """Not one of the switchable tabs — showResults/hideResults handle
        it independently of switchTab."""
        source = page()
        assert "function showResults() {" in source
        show = source[source.index("function showResults() {"):
                      source.index("function hideResults()")]
        assert "convoBar" in show
        assert "paneTabs" in show


class TestDocsCopyMatches:
    def test_the_tabs_exist_there_too(self):
        source = page("docs/index.html")
        tabs = re.findall(r'data-tab="([\w-]+)"', source)
        assert {"designs", "found", "cost"} <= set(tabs)
