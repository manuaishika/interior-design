"""Test for a live-reproduced screen bug: the dark pane's own centred empty
state ("Your designs appear here") sometimes never actually painted, though
the DOM and computed layout were correct — confirmed live via a real
browser: getBoundingClientRect and getComputedStyle both said the text was
there, visible, in the right place, light-on-dark — and the screen showed
solid black anyway. Scrolling one pixel fixed it instantly, which is the
signature of a compositor missing a repaint after a late layout shift
(most plausibly: webfonts finishing load and reflowing the paragraph above
the panel, changing the pane's height after its first paint).

Nothing here can stop the browser from missing that repaint. What this
pins is the cheap, safe mitigation: once the fonts actually finish loading,
force a real repaint of the one element that showed the bug.
"""

import pathlib


def page(path="static/showcase.html"):
    return pathlib.Path(path).read_text(encoding="utf-8")


class TestTheDarkPaneGetsASecondChanceToPaint:
    def test_a_repaint_is_forced_once_fonts_are_actually_ready(self):
        source = page()
        assert "document.fonts.ready.then(nudgeRepaint)" in source

    def test_the_nudge_is_scoped_to_the_pane_not_the_whole_page(self):
        """A global trick (toggling zoom, hiding <body>) risks a visible
        flicker or a scroll jump — this only touches the element that
        actually showed the bug, and leaves no visible trace."""
        source = page()
        start = source.index("function nudgeRepaint()")
        end = source.index("}", source.index("requestAnimationFrame", start))
        body = source[start:end]
        assert "$('pane')" in body
        assert "translateZ(0)" in body

    def test_docs_copy_has_the_same_fix(self):
        source = page("docs/index.html")
        assert "document.fonts.ready.then(nudgeRepaint)" in source
