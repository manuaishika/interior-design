"""Tests for TODO.md #4's frontend half: the +/- control caps to what the
account's tier allows, with a line saying why — not a button that silently
stops doing anything.

There is no JS test runner in this repo (see test_progressive_draw.py), so
this pins the source facts: the cap comes from /api/session's own `tier`
field rather than a hardcoded 4, and hitting the cap disables the button
and fills in a reason rather than leaving it clickable and inert.
"""

import pathlib


def page(path="static/showcase.html"):
    return pathlib.Path(path).read_text(encoding="utf-8")


def function_body(source: str, signature: str) -> str:
    start = source.index(signature)
    marker = "\n  function "
    async_marker = "\n  async function "
    end_candidates = [i for i in (
        source.find(marker, start + 1), source.find(async_marker, start + 1))
        if i != -1]
    end = min(end_candidates) if end_candidates else len(source)
    return source[start:end]


class TestTheCapComesFromTheAccount:
    def test_session_tier_is_read_and_applied(self):
        body = function_body(page(), "function loadSession(")
        assert "tier = s.tier" in body
        assert "applyTierCap()" in body

    def test_the_cap_is_not_hardcoded_to_four(self):
        """The old control let anyone reach 4 regardless of plan — the cap
        now comes from whichever tier actually applies."""
        body = function_body(page(), "function applyTierCap(")
        assert "tier.variants" in body

    def test_more_button_stops_at_the_cap_not_at_a_literal_four(self):
        source = page()
        start = source.index("$('more').onclick")
        end = source.index("};", start)
        body = source[start:end]
        assert "tier.variants" in body
        assert "Math.min(4," not in body

    def test_hitting_the_cap_disables_the_button_with_a_reason(self):
        body = function_body(page(), "function applyTierCap(")
        assert "$('more').disabled" in body
        assert "$('tierNote').textContent" in body


class TestTheMarkupExists:
    def test_the_tier_note_element_is_present(self):
        assert 'id="tierNote"' in page()


class TestDocsCopyMatches:
    def test_the_same_mechanism_exists_there(self):
        source = page("docs/index.html")
        assert 'id="tierNote"' in source
        body = function_body(source, "function applyTierCap(")
        assert "tier.variants" in body
