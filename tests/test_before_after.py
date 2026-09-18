"""Tests for TODO.md #3: a draggable before/after, and a single download.

A range input sits transparent over two stacked images; dragging it moves a
CSS clip-path on the "after" image, which is the whole mechanism — no
library, as asked for. "What changed" reuses the reading's items and the
same mustStay state the chips already use, rather than inventing a second
source of truth for kept vs. replaced. Download draws both onto one
<canvas> and hands back one file, because a hand-stitched pair is the
friction that stops people sharing the work.

There is no JS test runner in this repo (see test_progressive_draw.py), so
this pins the source facts the way the rest of this project's frontend
fixes do.
"""

import pathlib
import re


def page(path="static/showcase.html"):
    return pathlib.Path(path).read_text(encoding="utf-8")


def function_body(source: str, signature: str) -> str:
    """Crude but sufficient for this file's flat, unnested top-level
    functions: from the signature to the next top-level `function` at the
    same two-space indent, or end of file."""
    start = source.index(signature)
    marker = "\n  function "
    async_marker = "\n  async function "
    end_candidates = [i for i in (
        source.find(marker, start + 1), source.find(async_marker, start + 1))
        if i != -1]
    end = min(end_candidates) if end_candidates else len(source)
    return source[start:end]


def style_block(source: str) -> str:
    match = re.search(r"<style>(.*?)</style>", source, re.S)
    assert match, "no <style> block found"
    return match.group(1)


IDS = ("compare", "compareBefore", "compareAfter", "compareRange",
      "compareLine", "compareDownload", "compareKept", "compareReplaced")


class TestTheMarkupExists:
    def test_every_id_is_present(self):
        source = page()
        for element_id in IDS:
            assert f'id="{element_id}"' in source, f"#{element_id} is missing"

    def test_the_slider_is_a_plain_range_input(self):
        """"Plain <input type=range>... No library" — not a custom widget,
        not a slider plugin."""
        source = page()
        start = source.index('id="compareRange"')
        tag_open = source.rindex("<", 0, start)
        tag = source[tag_open:source.index(">", start)]
        assert "<input" in tag
        assert 'type="range"' in tag


class TestTheRevealIsCSSClipPath:
    def test_the_after_image_is_clipped_not_a_second_canvas(self):
        css = style_block(page())
        assert "clip-path" in css
        assert "#compareAfter" in css

    def test_dragging_the_range_moves_the_clip(self):
        """The whole mechanism: the range's own value drives how much of the
        after-image's clip-path is revealed."""
        body = function_body(page(), "function setCompareSplit(")
        assert "clipPath" in body
        assert "value" in body

    def test_the_range_input_is_wired_to_it(self):
        source = page()
        assert "$('compareRange').addEventListener('input'" in source
        assert "setCompareSplit(" in source


class TestWhatChanged:
    def test_kept_and_replaced_come_from_the_same_state_the_chips_use(self):
        """A second, independent notion of "kept" would drift from what the
        chips in "What it found" already say — this pins that it reuses
        `mustStay` rather than inventing one."""
        body = function_body(page(), "function compareLists(")
        assert "mustStay[item.name]" in body
        assert "kept" in body and "replaced" in body

    def test_an_empty_list_says_so_rather_than_showing_nothing(self):
        source = page()
        assert "fillList(" in source
        assert "empty-note" in source


class TestDownloadBoth:
    def test_it_draws_both_onto_one_canvas(self):
        body = function_body(page(), "$('compareDownload').onclick = function")
        assert "createElement('canvas')" in body
        assert "drawImage(before" in body
        assert "drawImage(after" in body

    def test_it_gives_back_a_single_file_not_two(self):
        body = function_body(page(), "$('compareDownload').onclick = function")
        assert body.count("toDataURL(") == 1
        assert body.count(".download =") == 1

    def test_the_images_load_from_data_uris_so_the_canvas_is_never_tainted(self):
        """A remote image URL would taint the canvas and make toDataURL throw
        — the before/after images are always data: URIs (an object URL for
        the upload, a base64 PNG for the design), so this never applies, but
        pin that the download path still reads from the <img> elements
        rather than re-fetching anything over the network."""
        body = function_body(page(), "$('compareDownload').onclick = function")
        assert "before.src = $('compareBefore').src" in body
        assert "after.src = $('compareAfter').src" in body
        assert "fetch(" not in body


class TestItDoesNotLingerStale:
    def test_a_new_room_hides_the_old_comparison(self):
        body = function_body(page(), "function paint(")
        assert "$('compare').classList.add('hidden')" in body

    def test_a_new_batch_of_designs_hides_it_too(self):
        body = function_body(page(), "async function draw()")
        assert "$('compare').classList.add('hidden')" in body


class TestEveryDesignCanBeCompared:
    def test_each_design_gets_a_compare_button(self):
        body = function_body(page(), "function appendDesign(")
        assert "openCompare(" in body
        assert "'Compare'" in body


class TestDocsCopyMatches:
    def test_the_same_mechanism_exists_there(self):
        source = page("docs/index.html")
        for element_id in IDS:
            assert f'id="{element_id}"' in source
        css = style_block(source)
        assert "clip-path" in css
        body = function_body(source, "$('compareDownload').onclick = function")
        assert "createElement('canvas')" in body
