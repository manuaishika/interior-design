"""Pins the invisible-buttons bug: near-black text on a near-black panel.

`.cta.ghost` sets `color: var(--ink)`, which is correct everywhere it is
used — except inside `.pane`, whose background *is* that same colour
(`--dark` and `--ink` are both #14130F). The border is light, so a Save or
Download button rendered there showed its outline and nothing else: not a
missing feature, an invisible one, and invisible to every other kind of
test this project has.

This does not run a browser. It does the small piece of the CSS cascade
that actually matters here — which declared `color` wins for an element
carrying classes {cta, ghost} inside an ancestor carrying class {pane} — so
this class of mistake gets a real regression test instead of a screenshot.
"""

from __future__ import annotations

import pathlib
import re


def page(path="static/showcase.html"):
    return pathlib.Path(path).read_text(encoding="utf-8")


def style_block(source: str) -> str:
    match = re.search(r"<style>(.*?)</style>", source, re.S)
    assert match, "no <style> block found"
    return match.group(1)


def custom_properties(css: str) -> dict[str, str]:
    root = re.search(r":root\s*\{([^}]*)\}", css, re.S)
    assert root, "no :root block found"
    return dict(re.findall(r"(--[\w-]+)\s*:\s*([^;]+);", root.group(1)))


def resolve(value: str, props: dict[str, str]) -> str:
    match = re.match(r"var\((--[\w-]+)\)", value.strip())
    return props.get(match.group(1), value).strip() if match else value.strip()


def rules(css: str) -> list[tuple[str, dict[str, str]]]:
    """[(selector, {property: value})], one entry per comma-separated selector.

    Deliberately naive about @media: the regex only ever completes a match on
    an innermost, single-brace-pair rule, so a selector wrapped in a media
    query is read the same as one that is not. That is exactly what is
    wanted here — none of the selectors this test cares about sit behind a
    breakpoint, and this project keeps its whole light palette in one root.
    """
    out = []
    for m in re.finditer(r"([^{}]+)\{([^{}]*)\}", css):
        selector = m.group(1).strip()
        if not selector or selector.startswith("@"):
            continue
        decls = dict(re.findall(r"([\w-]+)\s*:\s*([^;]+);?", m.group(2)))
        for one in selector.split(","):
            out.append((one.strip(), decls))
    return out


def _classes(compound: str) -> set[str]:
    return set(re.findall(r"\.([\w-]+)", compound))


def specificity(selector: str) -> int:
    """Class-token count across the whole selector — good enough here: none
    of the rules in play use an id, an attribute, or a bare type selector."""
    return len(re.findall(r"\.[\w-]+", selector))


def matches(selector: str, own_classes: set[str], ancestor_classes: set[str]) -> bool:
    """Does `selector` apply to an element with `own_classes`, somewhere
    inside an ancestor carrying `ancestor_classes`? Descendant combinators
    (whitespace) only — nothing here uses `>`, `+` or `~`."""
    parts = selector.split()
    if not parts:
        return False
    if not _classes(parts[-1]) <= own_classes:
        return False
    return all(_classes(part) <= ancestor_classes for part in parts[:-1])


def winning_color(css: str, own_classes: set[str], ancestor_classes: set[str],
                  props: dict[str, str]) -> str | None:
    """The `color` that actually applies, by CSS specificity — later rule
    wins a tie, same as the cascade does when nothing carries !important."""
    candidates = [
        (specificity(selector), i, decls["color"])
        for i, (selector, decls) in enumerate(rules(css))
        if "color" in decls and matches(selector, own_classes, ancestor_classes)
    ]
    if not candidates:
        return None
    _, _, color = max(candidates, key=lambda c: (c[0], c[1]))
    return resolve(color, props)


class TestTheSaveAndDownloadButtonsAreVisible:
    def test_the_dark_and_ink_tokens_really_are_the_same_colour(self):
        """The bug only exists because these two tokens happen to match —
        confirms the premise before testing the consequence."""
        css = style_block(page())
        props = custom_properties(css)
        assert props["--dark"] == props["--ink"]

    def test_a_ghost_button_inside_the_pane_is_not_ink_on_ink(self):
        css = style_block(page())
        props = custom_properties(css)
        color = winning_color(css, {"cta", "ghost", "keep"}, {"pane"}, props)
        assert color is not None, "no rule sets a colour for this button at all"
        assert color != props["--dark"]

    def test_the_same_button_outside_the_pane_is_unaffected(self):
        """The fix is scoped to `.pane .cta.ghost` on purpose — the plain
        version is correct everywhere else (the top bar, the sign-in
        dialog), so this pins that the general rule was not changed."""
        css = style_block(page())
        props = custom_properties(css)
        color = winning_color(css, {"cta", "ghost"}, set(), props)
        assert color == props["--ink"]

    def test_docs_copy_has_the_same_fix(self):
        """static/showcase.html and docs/index.html are kept identical bar
        one deliberate line — this fix is not that line."""
        css = style_block(page("docs/index.html"))
        props = custom_properties(css)
        color = winning_color(css, {"cta", "ghost", "keep"}, {"pane"}, props)
        assert color != props["--dark"]
