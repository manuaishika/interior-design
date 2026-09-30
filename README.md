# Second Draft

Upload a photo of a room, pick a look (or describe what you want), and get it
redrawn: same room, same view, same doors and windows, new finishes and
furniture. Built for Eleganza Interiors, Dubai.

Live studio: https://manuaishika.github.io/interior-design/#/studio
(the GitHub Pages copy is a preview of the interface; drawing runs on the
Render server, see [DEPLOY.md](DEPLOY.md)).

## What it does

- **Redesign from photos.** Six looks (Scandinavian, Mid-century, Industrial,
  Japandi, Bohemian, Modern luxury), or "stick to the brief" and write your own.
  Sixteen room types, each with what that room must contain (a nursery is not
  just a style).
- **Two depths.** *Light restyle* keeps the furniture and changes the finish
  (one photo). *Full redesign* replaces pieces, walls, floor, lighting and can
  add missing storage (two to four photos from different corners).
- **Every angle redrawn.** Each uploaded photo comes back redrawn, using the
  other photos as references so the angles match.
- **Structure stays put.** Doors, windows, built-ins, the AC and the *count* of
  things (one bed stays one bed) are protected in the instructions to the image
  model.
- **Pencil markup.** Draw freehand on a design, say what should change, and only
  that area is edited.
- **Before/after slider, download, follow-up edits, and a designer chat** that
  can see the design on screen.
- **Accounts and history.** Email sign-up or Google sign-in, saved designs and
  past rooms.
- **Plans and daily limits.** Free plan: 2 designs/day without an account, 5
  with one. Paid tiers change quality and designs per run (below).
- **"Talk to a designer"** hands the finished design to WhatsApp.
- **Feedback.** "Close to what you asked for?" is stored with the room, look and
  prompt so weak spots can be found.
- **Works on a phone.** Camera button, photos shrunk before upload, tap to
  preview a look, no sideways scrolling.

## How it works

```
photo(s) ─► reader (GPT-4o / Gemini): is it a room? what is in it, how many,
             three directions ─► shown to the person, who ticks what must stay
        ─► prompt: look + room brief + depth + what to keep + the person's words
        ─► image model edits the whole photo, original in front of it
        ─► designs appear one at a time
```

One key is enough. The engine is picked from whichever keys exist
(`resolve_backend` in `app/config.py`), or forced with `BACKEND=`:

| Engine | Needs | Reads with | Draws with |
|---|---|---|---|
| `openai` | `OPENAI_API_KEY` | GPT-4o | `gpt-image-2.5` (falls back to `gpt-image-2`, then `gpt-image-1`) |
| `free` | `GOOGLE_API_KEY` (no card) | Gemini | Gemini image model |
| `hosted` / `local` | Replicate + OpenAI, or nothing | SAM2 + GPT-4o / SegFormer | Stable Diffusion inpainting |

`openai` is the production path. `hosted` and `local` are the original
mask-and-inpaint pipeline (segment, label, lock, repaint); it still works and
backs `/api/analyze`, but it erased furniture inside the repaint zone, which is
why the default now protects structure in words rather than pixels.

### Plans

| Plan | Quality | Designs per run | Daily limit |
|---|---|---|---|
| Free | medium | 2 | 5 (2 without an account) |
| One Room | high | 3 | none |
| Whole Home | high | 4 | none |
| Studio | xhigh | 4 | none |

Paid plans are shown as "Coming soon" until payments are switched on.

## Run it

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env          # add OPENAI_API_KEY (or GOOGLE_API_KEY)
.venv/bin/uvicorn app.main:app --reload
```

Open http://localhost:8000. With no `DATABASE_URL` it uses a SQLite file.

Useful settings (all in `app/config.py`, all environment variables):

| Variable | What it does |
|---|---|
| `OPENAI_API_KEY` / `GOOGLE_API_KEY` | Turns on the engine |
| `DATABASE_URL` | Postgres in production; blank means SQLite |
| `SESSION_SECRET` | Keeps logins alive across restarts |
| `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET` | Shows "Continue with Google" |
| `STUDIO_ACCESS_CODE` | Closes the whole studio behind a phrase |
| `WHATSAPP_NUMBER` | Turns on "Talk to a designer" |
| `BUSINESS_NAME`, `CURRENCY`, `MARKET` | Branding and the region cost notes are written for |
| `ANON_DAILY_DESIGNS` | Daily allowance without an account (default 2) |
| `FEEDBACK_KEY` | Lets you read feedback at `/api/feedback?key=`; blank keeps it shut |

## Deploy

`render.yaml` describes the web service and the Postgres database; Render's
Blueprint button does the rest. Secrets are `sync: false` and are pasted in the
Render dashboard, never committed. Full walkthrough in [DEPLOY.md](DEPLOY.md).

The footer shows the deployed commit (`/api/health` returns it as `build`), so
you can tell whether a push is live.

## API

| Endpoint | Purpose |
|---|---|
| `POST /api/read` | Read a room: contents, three directions, cost lines |
| `POST /api/generate` | Photo(s) + look → design(s) |
| `POST /api/edit` | Change a design, optionally only inside a drawn area |
| `POST /api/chat` | Talk to the designer about a design |
| `POST /api/feedback` | Store a "close / not quite" answer |
| `GET /api/session`, `/api/health`, `/api/styles` | State, build and engine, looks |
| `/api/signup`, `/api/login`, `/api/auth/google` | Accounts |
| `/api/designs`, `/api/conversations` | Saved designs and history |

## Layout

```
app/       FastAPI backend: reading, generation, editing, limits, accounts
static/    the whole interface, one file (showcase.html)
docs/      GitHub Pages copy of the interface (kept identical, one line differs)
tests/     543 tests; the paid calls are stubbed, everything else is real
client/    client material
```

## Tests

```bash
.venv/bin/python -m pytest
```

After changing `static/showcase.html`, copy it to `docs/index.html`.
