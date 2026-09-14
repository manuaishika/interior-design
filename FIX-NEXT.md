# Four fixes

Read `HANDOFF.md` for the layout of the project. Run
`.venv/bin/python -m pytest` before every commit — 340 pass right now.
`static/showcase.html` and `docs/index.html` are the same file: change one,
copy to the other, then check `git diff` for the single deliberate difference.

---

## 1. The invisible buttons under every design

**This is a bug, not a design choice.** Under each render there are two empty
outlined rectangles. They are the Save and Download buttons, and they cannot be
read or aimed at.

The cause, exactly:

```css
.cta.ghost { background: none; color: var(--ink); ... }   /* #14130F */
.pane      { background: var(--dark); ... }               /* #14130F */
```

Near-black text on a near-black panel. The border is light, so the outline
shows and the label does not.

**Fix:** give the ghost button a variant for use on the dark pane — light text,
light border. The buttons under a design live inside `.pane`, so
`.pane .cta.ghost` is enough; do not change `.cta.ghost` globally, it is
correct everywhere else (the top bar, the dialog).

While in there: those buttons are cramped. Give them room and make the label
plain — `Save` / `Saved` and `Download`.

**Pin it with a test.** Any element rendered inside `.pane` must not use a
foreground colour equal to `--dark`. A contrast assertion in the page tests is
enough; this class of mistake is invisible to every other kind of test.

---

## 2. A full redesign cannot change the ceiling

`app/generation.py`, the `PRESERVE` block, line reading:

```
- the walls, the ceiling and the floor plan, including ceiling height
```

That freezes the ceiling completely. So a full redesign in a bold style comes
back with the same flat ceiling it started with, and the room reads as barely
touched — which is the actual complaint behind "the vibe looks very similar".

**The distinction to draw is structure versus finish.**

Never changes, at any depth:
- room shape, wall positions, **ceiling height**
- window and door positions and sizes
- the camera

Changes on a full redesign:
- the **ceiling treatment** — a dropped or false ceiling, cove lighting, a
  different colour, coving, a change of light fitting
- wall finishes, flooring, all the loose furniture

Changes on a light restyle:
- paint, textiles, furniture finishes only. Ceiling stays as it is.

So `PRESERVE` should protect ceiling *height* and stop protecting ceiling
*appearance*, and `DEPTHS["renovate"]` should say the ceiling is fair game —
name false ceilings and cove lighting specifically, because in a lot of the
world that is what a redesign means and the model will not assume it.

`DEPTHS["restyle"]` should say the opposite in as many words, or the light
option stops being light.

Tests: a full redesign prompt mentions the ceiling as changeable; a light
restyle does not; both still protect ceiling height.

---

## 3. Take "Your chats" out of the top bar

It is in the navigation next to Studio, Explore, How it works and Pricing,
which says it is a section of the product. It is not. It is a drawer you open
occasionally.

- Remove `#navSaved` and the chats link from `<nav>`.
- Put both behind one control **inside the studio panel header**, next to
  "Second Draft engine" — a small `History` button opening a panel listing past
  threads and designs together, newest first, each row opening that thread.
- Keep every endpoint and table exactly as they are. This is presentation only.

The signed-in name and Sign out stay where they are.

---

## 4. The studio should be one panel, not a panel and three boxes

Right now: a controls rail, a dark results pane, then separately "Talk it
through", "What it found", and a cost table stacked underneath. It has been
moved once already and still reads as bolted on.

**Make the results pane the whole right-hand side, and put the conversation in
a bar pinned along the bottom of it** — an input that is always visible, with
the thread growing upward above it when there is one. That is the pattern from
the reference the project was built to, and it means the thing you do next is
never below the fold.

`What it found` and `What it would cost` become **tabs or collapsible sections
inside that same pane**, not separate cards down the page. Default to the
designs; the other two are there when wanted.

Keep every id the tests rely on — `#shots`, `#thread`, `#ask`, `#askGo`,
`#redraw`, `#chips`, `#cost`, `#costBlock`, `#savedBlock`. Move them, do not
rename them, or a lot of tests fail for no useful reason.

---

## 5. Do not wait for all three designs before showing any

Not asked for, but it is most of why the app feels slow.

`pipeline._run_openai` uses `asyncio.gather`, so nothing appears until the
slowest of three finishes. Each image is 20-40 seconds, so the person stares at
an empty panel for a minute and assumes it has hung.

Stream them instead: return each design as it lands. Either a streaming
response, or have the page ask for one variant at a time and append. One
picture in twenty seconds feels like a working product; three in a minute
feels broken.

If that is too big a change to make safely, at minimum put a real progress
line in the panel — "Drawing 1 of 3" — rather than a static message.
