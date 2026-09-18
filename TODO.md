# What is left

Everything still to build, in the order to build it. Each item says what is
wrong, where it lives, and how you know it is done.

Ground rules for all of it:
- `.venv/bin/python -m pytest` before every commit. **412 pass right now.**
- `static/showcase.html` and `docs/index.html` are the same file. Change one,
  copy to the other, `git diff` to check the one deliberate difference.
- Keep element ids. Move them, never rename them, or tests fail for no reason.
- Comments explain *why*, not *what*.

---

## 1 · The Save and Download buttons are invisible  — 10 minutes

Under every render sit two empty outlined rectangles. They are the buttons.

```css
.cta.ghost { color: var(--ink); }    /* #14130F */
.pane      { background: var(--dark); } /* #14130F */
```

Near-black on near-black. The border shows, the label does not.

**Do:** add `.pane .cta.ghost { color: <light token>; border-color: <light token>; }`.
Do **not** change `.cta.ghost` globally — it is correct in the top bar and the
dialog. Give the buttons more padding while you are there.

**Done when:** you can read both labels on the dark panel, and the top-bar
Sign in button still looks right.

**Test:** nothing rendered inside `.pane` may take a foreground colour equal to
`--dark`. This class of bug is invisible to every other kind of test.

---

## 2 · A follow-up must edit the design, not start again  — half a day

**The single worst thing in the product right now.** Typing "change the wall
colour" and pressing the button does not change the wall colour. `#redraw`
appends the conversation to the brief and **re-runs from the original
photograph**, so a different room comes back and the change is lost among fifty
other differences. It reads as though the conversation is ignored.

**Do:** a follow-up edits *the design on screen*, with the instruction as the
whole prompt:

```python
openai_images.redraw(<the generated design>, None,
                     instruction + "\n\n" + PRESERVE, settings)
```

Same endpoint, different subject. `gpt-image-2.5` is built for this.

Two buttons, and the difference must be visible:

- **Apply this change** — edits what is on screen, one thing moves. *Primary.*
- **Draw it again** — back to the photograph, everything may move. *Secondary.*

Carry the result forward so a third instruction edits the second result.
Append `PRESERVE` to every edit or the second one reintroduces the doorway the
first avoided.

**Done when:** "make the wall deep olive" returns the same room with a deep
olive wall, and the furniture is where it was.

**Tests:** a follow-up sends the generated design as the subject, not the room
photo; the style preset is not re-applied; three edits chain.

---

## 3 · Before and after  — half a day

A section under the designs: the original photograph and the chosen design with
a **draggable divider** between them. Plain `<input type="range">` over two
stacked images and a CSS `clip-path`. No library.

Underneath, what changed, from the reading: kept items and replaced items as
two short lists.

Plus **Download both** — one side-by-side JPEG drawn on a `<canvas>`. That is
the file that gets sent to a client or a builder, and stitching it by hand is
the friction that stops people sharing the work.

**Done when:** you can drag between the two, and one download gives a single
image with both in it.

---

## 4 · The tier has to come from the account  — half a day

`config.TIERS` and `tier_of()` exist and work; `Settings.tier` is one value for
the whole deployment. So everybody is on the same plan.

**Do:**
- `tier` column on `store.User`, defaulting to `"free"`
- resolve the tier from the signed-in user, not from `Settings`
- `/api/session` returns it so the page knows
- the page shows the tier and caps the `−/+` control to what it allows, with a
  line saying why rather than a silently dead button
- the Pricing page marks the current plan

No payments yet. Setting somebody's tier by hand in the database is fine, and
is what a demo needs anyway.

**Done when:** two accounts on different tiers get different quality and a
different number of designs from the same photograph.

---

## 5 · Finish the catalogue  — one to two days

`app/catalogue.py` and the importer exist. Close the loop.

- **Import:** `data/catalog.csv` in the shape of `catalogue-template.csv`. Must
  be re-runnable — upsert on `sku`, report rows added, updated and rejected,
  and say *why* each rejection failed. Reject the row, never the file.
- **Photos, both ways:** `image_url` when it is already online, `image_file`
  plus a folder at `data/catalog-images/` when it is not. A client with a hard
  drive full of JPEGs should not be told to build a website first.
- **Match:** `match(category, style, room, max_width_mm, budget)` ranked by
  style hit, then room hit, then fits the space, then in stock, then price. SQL
  with an ORDER BY. No embeddings.
- **Into the render:** for every item *not* ticked as must-stay, feed the
  matched product into `build_prompt` — "a walnut writing desk, 1200mm wide"
  rather than "a desk".
- **Show it:** matched products beside each design — photo, name, size, price,
  link. The costing table switches from the reader's estimate to the sum of
  real prices, keeping the estimate as a fallback for anything the catalogue
  does not sell.

**Label it honestly in the interface:** the render is steered towards these
pieces, it is not a photograph of them. That assumption ends in a complaint.

**Done when:** a CSV imports, re-imports cleanly, reports its rejections, a
wardrobe too wide for the wall is not suggested for it, and a render names real
products at their real prices.

---

## 6 · Payments  — after everything above

Razorpay, not Stripe — the customer is in India.

Blocked on a Razorpay account with business details, which is the client's to
open. When it exists: checkout for a tier, a webhook that sets `User.tier` on
success, and a way to cancel. **Verify the webhook signature** — an unverified
payment webhook is a free-upgrade button for anyone who finds the URL.

The tier system from item 4 is exactly what this plugs into, which is why it
comes after.

---

## Not to do

- Do not split the backend into a second repository.
- Do not add another engine. Three is one more than the product needs.
- Do not rewrite the frontend. One file is deliberate.
