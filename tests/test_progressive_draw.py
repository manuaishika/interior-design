"""Tests for FIX-NEXT.md #5: do not wait for all designs before showing one.

asyncio.gather inside a single /api/generate call meant nothing appeared
until the slowest of two or three 20-40 second images finished — a minute
staring at an empty pane reads as the app having hung. The page now fires
one variants=1 request per design, concurrently, and appends each as it
lands; a real progress line ("Drawn 1 of 3…") replaces the old static
"Drawing 3…" message.

Verified live in a browser during development (not just pinned here): at
t=4.5s with delays of 3s/6s/9s on three mocked requests, exactly one design
had rendered and the line read "Drawn 1 of 3…" — this file pins the source
facts that make that true, since there is no JS test runner in this repo.
"""

import pathlib


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


class TestOneRequestPerDesign:
    def test_draw_sends_variants_equal_to_one(self):
        """The whole point — a batch request would ask for `variants=count`;
        one request per design asks for one each."""
        body = function_body(page(), "async function drawOne(")
        assert "f.append('variants', '1')" in body
        assert "f.append('variant_offset', String(i))" in body

    def test_the_requests_are_fired_together_not_awaited_one_by_one(self):
        """A for-loop that `await`s each request in turn would be no faster
        than the old batch — the fix is firing them all, then awaiting the
        whole set."""
        body = function_body(page(), "async function draw()")
        push_line = body.index("requests.push(drawOne(")
        await_line = body.index("await Promise.all(requests)")
        assert push_line < await_line

    def test_progress_is_reported_as_each_one_lands(self):
        body = function_body(page(), "async function drawOne(")
        assert "progress.done++" in body
        assert "'Drawn ' + progress.done + ' of ' + progress.total" in body

    def test_a_failed_design_does_not_block_the_others(self):
        """One bad request must not stop the rest from appearing, and only
        a clean sweep of failures should read as an error."""
        draw_body = function_body(page(), "async function draw()")
        assert "progress.done === 0" in draw_body
        one_body = function_body(page(), "async function drawOne(")
        assert "progress.failed++" in one_body

    def test_the_static_drawing_n_message_is_gone(self):
        """draw() now owns its own progress narrative — the caller must not
        set a message that a moment later gets silently overwritten."""
        source = page()
        assert "'Drawing ' + count + '" not in source


class TestDocsCopyMatches:
    def test_the_same_mechanism_exists_there(self):
        body = function_body(page("docs/index.html"), "async function drawOne(")
        assert "f.append('variants', '1')" in body
