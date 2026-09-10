# Putting this online

## First, the thing that keeps confusing people

**The website and the engine are two different things.**

The website is a shop window. It is a file. GitHub Pages will host it free,
forever, and it will always load instantly. It cannot think.

The engine is a workshop. It reads photographs and draws rooms. That takes a
computer that is switched on, running code, with a model behind it. A file
cannot do that, no matter how good the file is.

> **If a client asks "why doesn't the button work on the website?"**
>
> "That link is the shop window — it shows what the product looks like. The
> part that actually redesigns a room runs on a server, because it has to
> think. We're turning that on now; it's a separate address."

This is not a limitation of our build. Every AI product on earth works this
way. Nobody's website does the thinking.

So: **GitHub Pages for the window, a real host for the workshop.** Once the
workshop is live, its address serves the window too, and everything works at
one URL.

---

## The Google path — cheapest, about ten minutes

This is the one to use before anybody has committed to OpenAI + Replicate.

### 1. Get a key

Go to **[aistudio.google.com/apikey](https://aistudio.google.com/apikey)**,
sign in with any Google account, click **Create API key**. Copy it.

One key does both halves: reading the room and drawing it.

> **Reading a room is free. Drawing one is not — any more.**
>
> Google removed the free tier for image generation in 2026: every Gemini
> image model now returns HTTP 429 with `limit: 0` on a keyless project. To
> draw, open **[console.cloud.google.com](https://console.cloud.google.com)**
> → **Billing** → link a card to the project behind your key. It is
> pay-as-you-go, roughly **4 cents per generated image**, and — unlike
> `gpt-image-1` — there is **no ID / Persona check**, just the card. Set a
> budget alert at Billing → Budgets & alerts.
>
> Health will still say `can_draw: true` with an unbilled key, because the key
> is valid; the failure only shows on the first generate. `/api/generate` then
> returns a 502 whose message names billing.

### 2. Put the engine on Render

1. **[render.com](https://render.com)** → sign up (free) → **New → Blueprint**
2. Point it at this repository. It reads `render.yaml` and fills everything in.
3. Under environment variables, set:

   | Key | Value |
   | --- | --- |
   | `GOOGLE_API_KEY` | the key from step 1 |
   | `STUDIO_ACCESS_CODE` | any phrase you invent, e.g. `oak-lamp-42` |
   | `SESSION_SECRET` | any long random string |

4. Deploy. Two or three minutes.

You do **not** need to set `BACKEND`. The app works out which engine to run
from the keys you gave it — a Google key alone means the free engine.

### 3. Check it

Open `https://your-app.onrender.com/api/health`. You want:

```json
{ "engine": "free", "can_read": true, "can_draw": true,
  "locks_are_enforced": false }
```

Then open the site itself, press **Log in**, enter your access code, go to
**Studio**, and put a room photo in.

**Free tier caveat:** a free Render service sleeps after 15 minutes idle and
takes ~40 seconds to wake. Before a client demo, load the page once to wake
it. The $7/month plan removes this, and it is worth it for a real meeting.

---

## What the Google path actually gets you

Be straight with the client about this, because it is also the argument for
paying later.

| | Google | Paid (OpenAI + Replicate) |
| --- | --- | --- |
| Reads the room | yes, free | yes |
| Counts what is there | yes, free | yes |
| Three directions | yes, free | yes |
| Talks it through | yes, free | yes |
| Redraws the room | yes, **needs billing** (~4¢/image) | yes |
| **Holds doors and windows** | **asked for** | **enforced** |
| Card needed | yes, to draw | yes |
| ID check | **no** | yes, for `gpt-image-1` |

That bold row is the whole quality difference, and it is worth explaining plainly:

- **Free** tells the model, in detail, not to move the door. It usually
  listens. Sometimes it does not.
- **Paid** cuts the photo into regions, finds the door, and paints a mask over
  it. The generator is not *able* to touch it. It is geometry, not manners.

For showing someone the idea, free is genuinely fine. For a client's actual
flat, where the door is where the door is, the mask is what you are buying.

---

## The paid path

Same as above, but set these instead:

| Key | Where |
| --- | --- |
| `OPENAI_API_KEY` | platform.openai.com |
| `REPLICATE_API_TOKEN` | replicate.com |

Both present, and the app switches to the full engine on its own. Health then
reports `"engine": "hosted"` and `"locks_are_enforced": true`.

Keep both and it uses the paid path; the Google key stays as a fallback you
can switch to with `BACKEND=free`.

---

## The door

`STUDIO_ACCESS_CODE` is a shared code, checked by the server. Anyone with it
can use the studio; anyone without it gets the site but not the engine. That
stops a public link burning through your quota.

It is deliberately not a full account system — there is no sign-up, no email,
no password reset — because there are no users to put in a database yet. When
there are, the session cookie is already signed and expiring, so real accounts
slot in behind the same door.

Leave `STUDIO_ACCESS_CODE` unset and there is no door at all.

Set `SESSION_SECRET` or everyone gets signed out on every deploy.

---

## Never commit a key

`render.yaml` marks every key `sync: false`, which means Render asks you to
type it in and it never touches the repository. `.env` is gitignored. If a key
ever does reach a commit, treat it as burned: revoke it and issue a new one.

---

## Running it on your own machine

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
echo "GOOGLE_API_KEY=your-key-here" > .env
.venv/bin/uvicorn app.main:app --reload
```

http://localhost:8000. No access code needed locally unless you set one.

---

## Accounts

Signing in is how somebody keeps a design. That needs identity, so it needs a
place to put people.

### The database

Set `DATABASE_URL` and it uses Postgres. Leave it blank and it writes a SQLite
file next to the code — fine on your laptop, **wrong on Render**, where the
filesystem is wiped on every deploy and everyone's designs go with it.

`render.yaml` provisions a free Postgres and wires `DATABASE_URL` in for you,
so a Blueprint deploy gets this right without you doing anything. Render's
free database expires after 90 days; upgrade it before real people are using it.

### Continue with Google

Ten minutes, free, and it means nobody has to invent a password.

1. **[console.cloud.google.com](https://console.cloud.google.com)** → new project
2. **APIs & Services → OAuth consent screen** → External → fill in the app name
   and your email → add yourself as a test user
3. **Credentials → Create credentials → OAuth client ID → Web application**
4. Under **Authorised redirect URIs**, add — exactly, no trailing slash:

   ```
   https://your-app.onrender.com/api/auth/google/callback
   http://localhost:8000/api/auth/google/callback
   ```

5. Copy the client ID and secret into Render as `GOOGLE_CLIENT_ID` and
   `GOOGLE_CLIENT_SECRET`

Leave both blank and the button simply does not appear — email and password
still work. Get the redirect URI wrong and Google says `redirect_uri_mismatch`;
the app catches that one and tells you the exact URI to paste.

### Email and password

Works with no extra setup. Passwords are hashed with scrypt and a per-account
salt.

**There is no password reset yet**, because a reset means sending email, which
means an email provider and another key. Until that exists, someone who forgets
their password has to be helped by hand — which is fine at ten users and not at
a thousand. Google sign-in sidesteps it entirely.

### The access code is not the login

`STUDIO_ACCESS_CODE` is still there, and it is a different thing: it closes the
whole deployment to anyone who does not have the phrase. Useful for a private
demo, irrelevant otherwise. It admits you but makes you nobody in particular,
so it cannot save designs — only an account can.

Leave it unset and anyone can open the site and make an account, which is what
you want once it is public.

---

## One OpenAI key is enough

There is a third engine between free and full, and it is the one to use if you
have an OpenAI key and nothing else.

| | Free (Google) | **One key (OpenAI)** | Full (OpenAI + Replicate) |
| --- | --- | --- | --- |
| Reads the room | yes | yes | yes |
| Finds the doors | — | GPT-4o, as boxes | SAM2, as outlines |
| Holds them | **asked** | **masked** | **masked** |
| Accounts needed | 1, free | **1** | 2 |

Set `OPENAI_API_KEY` and leave `REPLICATE_API_TOKEN` blank. Health then reports:

```json
{ "engine": "openai", "can_read": true, "can_draw": true,
  "locks_are_enforced": true }
```

**How it works.** GPT-4o is asked where the doors, windows and walkways are, in
fractions of the frame. Those boxes are painted out, and `gpt-image-1` — whose
edit endpoint takes a mask — repaints only what is left. The generator is not
*able* to touch a door, which is the same guarantee the full path gives.

**What it gives up.** Boxes, not outlines. SAM2 returns the actual silhouette
of a door; this returns a rectangle around it, so a little wall near the frame
is protected too. That protects slightly more than it needs to, which is the
right direction to be wrong — destroying a doorway is the failure that matters,
under-editing the wall beside it is not.

Add `REPLICATE_API_TOKEN` later and it switches to outlines on its own.

### Cost

Two OpenAI calls per room plus one per design: a vision call to find the
structure, and an image edit per option. Set a spend limit at
platform.openai.com/settings/organization/limits before you point a demo at
it — that is the protection against a retry loop, not care.
