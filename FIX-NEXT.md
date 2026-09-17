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

---

## 6. Two photographs minimum for a full redesign

**Do this after fix 4.** It touches the controls rail, which the layout rewrite
also moves, and doing them in the other order means merging the same file
twice.

### Why

One photograph shows one corner. A light restyle only changes the finishes on
things already in frame, so one is enough. A full redesign changes the ceiling,
the flooring and the furniture — and then has to invent everything the camera
could not see, which is where the invented doorways and impossible air
conditioners came from.

Two photographs from different corners give it most of the room. It also fixes
the reader missing things: a television at the edge of one frame is plainly
there in the other.

This is a real capability rather than a trick. `images.edit` takes **up to 16
reference images**. We are sending one.

### The rule

| depth | minimum |
| --- | --- |
| Light restyle | 1 photo |
| **Full redesign** | **2 photos** |

Enforce it, but never as a dead end. Someone with one photograph must be told
what to do about it:

> A full redesign changes things a single photograph cannot show — the ceiling,
> the far wall, the corner behind you. Add a second photo from another corner
> of the room, or switch to Light restyle.

Both escapes in that sentence must work: adding a photo, and switching depth.
If they switch to Light restyle the warning clears immediately.

### Build

**Page.** `#photo` takes `multiple`. The drop area becomes a strip of
thumbnails with an add tile and a remove control on each. Keep `#plus` working
as it does.

**The first photo is the one that gets redrawn** — the others are reference.
Say so on the strip ("This view is the one redrawn") and let people reorder, or
they will upload the best photo second and wonder why the output ignores it.

**`/api/generate`** takes `photos: list[UploadFile]` instead of one `photo`.
Accept a single `photo` as well so nothing already calling it breaks. Validate
every file through `_read_upload`, not just the first.

**`/api/read`** should read *all* of them — one survey across every view, so
the item list is the union. That is what stops the television going missing.

**`pipeline._run_openai`** passes the first as the image to edit and the rest
as references. `openai_images.redraw` grows a `references: list[bytes]`
parameter; the OpenAI client takes a list for `image`.

**Cost.** Each extra reference is more input to pay for on every variant. Not
much, but say so in the interface next to the count rather than letting a
surprise arrive on the bill.

### Tests

- A full redesign with one photo is refused, and the message names both ways out
- A light restyle with one photo still works
- Two photos reach the generator, first as subject and rest as reference
- A single `photo` field still works — old callers must not break
- Every uploaded file is validated, not only the first

---

## 7. Quality is the cost dial, and it is not wired up

**Do this first. It is two lines and it is where the money is going.**

`openai_images.redraw` never sends `quality`, so every render uses OpenAI's
default. On `gpt-image-2.5-sunburst` that resolves high, and the published
range per 1024x1024 image runs **$0.006 at `low` to $0.211 at `max`** — a
factor of about thirty-five. We are paying near the top of it and nobody chose
that.

Both models cost the same; the price is in the tokens, so `quality` and `size`
are the whole lever.

### Build

Add `image_quality` to `Settings` and pass it through. Then tie the allowance
to the tier — quality *and* count together, because that is exactly the ladder
the pricing page already sells:

| tier | quality | designs per run | rough cost per run |
| --- | --- | --- | --- |
| Free | `low` | 2 | about 1 cent |
| One Room / Whole Home | `high` | 3 | ~20-30 cents |
| Studio | `xhigh` | 4 | ~60-80 cents |

Those are estimates from the published per-image range at 1024x1024; our
landscape size is larger, so treat them as a floor and confirm against the
OpenAI usage page after a day of real use.

`max_variants` is currently 4 for everybody. It should come from the tier, and
the page's `−/+` control must not offer more than the tier allows — with a line
saying why, not a silently capped button.

**Sanity-check the cheap tier by eye before shipping it.** `low` may be too
rough to show a client; if it is, `medium` is the free tier and the free tier
gets 2 designs rather than 3.

---

## 8. A follow-up must edit the design, not start again

### Why it feels like nothing happens

Saying "change the wall colour" and pressing Draw again does not change the
wall colour of the design on screen. `#redraw` appends the conversation to the
brief and re-runs the whole pipeline **from the original photograph**. A new
room comes back, not an edit of the one being discussed — so the change is
lost among fifty other differences and the conversation reads as ignored.

### The fix

A follow-up should edit **the design that is being discussed**, with the
instruction as the whole prompt:

```
openai_images.redraw(<the generated design>, None, "Change the wall colour to
                     deep olive. Change nothing else.", settings)
```

Same endpoint, different subject. `gpt-image-2.5` is built for exactly this —
multi-turn editing is in its release notes.

Two paths, and the difference has to be visible in the interface:

- **Draw it again** — back to the photograph, new designs, everything may move
- **Apply this change** — edits the design on screen, one thing moves

Add the second as the primary button in the conversation, keep the first as
the secondary. Carry the design forward each time so a third instruction edits
the second result, not the first.

Append `PRESERVE` to the edit instruction as well, or the second edit
reintroduces the invented doorway the first one avoided.

### Tests

- A follow-up sends the generated design as the subject, not the room photo
- The instruction is the prompt; the original style preset is not re-applied
- A chain of three edits builds on each other

---

## 9. Before and after

A section under the designs: the original photograph and a chosen design, with
a **draggable divider** between them — drag left to see what was there, right
to see what it became. One control, no legend needed.

Underneath, what actually changed, from the reading: the kept items and the
replaced ones as two short lists. The picture shows the difference; the lists
say what it was.

Plain `<input type="range">` over two stacked images and a CSS `clip-path` is
enough. No library.

Also give it a **Download both** that writes a single side-by-side JPEG — that
is the thing that gets sent to a client or a builder, and stitching it by hand
is the friction that stops people sharing the work.
