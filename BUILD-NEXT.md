# Next two jobs

Read `HANDOFF.md` first for how the project is laid out. Do these in order —
job 1 is small and job 2 depends on nothing in it, but job 1 is what people
will notice missing tomorrow.

Run `.venv/bin/python -m pytest` before every commit. 309 pass right now.

---

# Job 1 — Chat history

## Why

Designs already persist against an account (`store.Design`). Conversations do
not. Close the tab and everything said about a room is gone, which is not how
anyone expects a chat-shaped thing to behave.

## Build

**`app/store.py`** — two tables, alongside the existing ones:

```python
class Conversation(Base):
    id, user_id (FK users, CASCADE, indexed)
    title       # auto, see below
    room, style # what was selected at the time
    photo       # LargeBinary: the original room photo, so a thread can be reopened
    created_at, updated_at (indexed)

class Message(Base):
    id, conversation_id (FK conversations, CASCADE, indexed)
    role        # "user" | "assistant"
    content
    created_at
```

Add `conversation_id` to `Design` (nullable FK), so the designs made in a
thread hang off it.

Helpers in the same style as the existing ones — `conversations_for(db, user_id)`,
`conversation_for(db, user_id, id)` scoped to the owner, `add_message(...)`,
`rename(...)`, `delete_conversation(...)`.

**Titles**: take the first 60 characters of the person's first message; if
there isn't one, `"{Room} · {Style}"` with the date. Never leave it blank —
a list of "Untitled" is not a history.

**`app/main.py`** — endpoints, all behind `auth.require_user` like `/api/designs`:

```
GET    /api/conversations            list, newest first
POST   /api/conversations            start one, returns id
GET    /api/conversations/{id}       messages + designs + the photo url
PATCH  /api/conversations/{id}       rename
DELETE /api/conversations/{id}
```

`/api/chat` takes an optional `conversation_id` and appends both turns to it.

**Ownership is the thing to get right.** Copy the pattern already in
`design_for`: every lookup is scoped to the owner and a miss returns **404, not
403** — somebody else's id must not be confirmed to exist by failing
differently. `tests/test_accounts.py::TestDesignsBelongToPeople` shows the
shape; mirror those tests exactly for conversations.

**Page** (`static/showcase.html`) — a thread list. There is already a
`#navSaved` button in the top bar for designs; put history beside it. Opening a
thread restores the photo, the messages and the designs. Keep it in the single
file.

## Done when

- A signed-in person reloads and their conversation is still there
- One account cannot see, rename or delete another's thread
- Signing out hides it all

---

# Job 2 — The catalogue

Read **`CATALOGUE.md`** first — it has the schema, the column meanings and the
honest limit. `catalogue-template.csv` is the shape of the data.

## Build in this order. Each step is useful on its own.

### 2a. Store it

`app/catalogue.py` + a `Product` table matching the CSV columns. Dimensions as
integer millimetres. `style_tags` and `room_tags` normalised into their own
tables or stored as arrays — not as comma strings to be `LIKE`-matched later.

An importer: `python -m app.import_catalogue products.csv`.

It must be **re-runnable**. Upsert on `sku`, report rows added, updated and
rejected, and say *why* each rejection failed. The client will send a v2 sheet
with typos in it, and a silent import is how a catalogue rots.

Validate: required columns present, price and dimensions numeric and positive,
`category` in the fixed list, `image_url` looks like a URL. Reject the row,
never the file — one bad row must not lose the other four hundred.

### 2b. Match against it

`match(category, style, room, max_width_mm=None, budget=None) -> [Product]`

Rank by: style tag hit, then room tag hit, then fits the space, then in stock,
then price ascending. Return the top few. No ML, no embeddings — a SQL query
with a sensible ORDER BY is correct here and will stay debuggable.

The size filter is the part that earns its keep: a 2400mm wardrobe does not go
in a 2000mm alcove, and that is the one thing a human eye misses on a moodboard.

### 2c. Put it in the render

In `pipeline._run_openai`, after the reading: for each item the person did
**not** tick as must-stay, find a match and feed the chosen product into
`build_prompt` as part of `contents` — "a walnut writing desk, 1200mm wide"
rather than "a desk".

Return the matched products in the response alongside `generations`.

### 2d. Show it, and price it

Beside each design: the matched products, with photo, name, size, price and a
link out. And the costing table (`#costBlock`, `drawCost`) switches from the
reader's estimate to the sum of real prices when a catalogue is loaded — keep
the estimate as the fallback, because most rooms will contain things the
catalogue does not sell.

**Label it honestly.** The render is steered towards these pieces; it is not a
photograph of them. `CATALOGUE.md` explains why. Put a line in the interface
saying so — that assumption ends in a complaint otherwise.

## Done when

- A CSV imports, re-imports cleanly, and reports its rejections
- A wardrobe too wide for the wall is not suggested for it
- A render names real products, and the total is their real prices
- With no catalogue loaded, everything still works exactly as it does today

---

## House rules

- Tests are behavioural — they pin what must stay true and why. Read a few first.
- Never commit a key. `.env` is gitignored.
- `static/showcase.html` and `docs/index.html` are the same file; change one,
  copy to the other, and check `git diff` for the one deliberate difference.
- Comments explain *why*, especially where something is deliberately odd.
