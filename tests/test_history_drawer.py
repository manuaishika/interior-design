"""Tests for FIX-NEXT.md #3: History is a drawer, not a nav item.

"Your chats" sat in the top navigation next to Studio, Explore, How it
works and Pricing — which says it is a section of the product. It is a
drawer somebody opens occasionally, so both it and "Your designs" moved
into one History control inside the studio panel's own header, and their
two separate lists became one merged, newest-first list. Nothing about the
endpoints or tables changed; this is presentation only.
"""

import pathlib


def page(path="static/showcase.html"):
    return pathlib.Path(path).read_text(encoding="utf-8")


def nav_block(source: str) -> str:
    start = source.index('<nav id="nav">')
    end = source.index("</nav>", start)
    return source[start:end]


class TestTheTopBarIsJustTheProduct:
    def test_your_chats_and_your_designs_are_gone_from_the_nav(self):
        nav = nav_block(page())
        assert "navSaved" not in nav
        assert "navHistory" not in nav
        assert "Your chats" not in nav
        assert "Your designs" not in nav

    def test_the_four_real_sections_are_still_there(self):
        nav = nav_block(page())
        for section in ("studio", "explore", "how", "pricing"):
            assert f'data-go="{section}"' in nav


class TestHistoryLivesInThePanelHeader:
    def test_a_single_history_control_sits_next_to_the_engine_line(self):
        source = page()
        top = source.index('class="panel-top"')
        body = source.index('class="panel-body"')
        header = source[top:body]
        assert "Second Draft engine" in header
        assert 'id="historyBtn"' in header

    def test_it_is_hidden_until_somebody_is_signed_in(self):
        source = page()
        top = source.index('id="historyBtn"')
        tag = source[top - 80:top + 40]
        assert "hidden" in tag

    def test_one_merged_list_not_two(self):
        """Two id="saved"/id="history" lists became one — the drawer shows
        chats and designs together, sorted by the same date."""
        source = page()
        assert 'id="saved"' in source
        assert 'id="history"' not in source
        assert source.count('id="savedBlock"') == 1


class TestDocsCopyMatches:
    def test_the_nav_is_trimmed_there_too(self):
        nav = nav_block(page("docs/index.html"))
        assert "navSaved" not in nav
        assert "navHistory" not in nav
