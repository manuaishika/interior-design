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

## The free path — no card, about ten minutes

This is the one to use before anybody has paid for anything.

### 1. Get a key

Go to **[aistudio.google.com/apikey](https://aistudio.google.com/apikey)**,
sign in with any Google account, click **Create API key**. No card, no
billing setup. Copy it.

One key does both halves: reading the room and drawing it.

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

## What free actually gets you

Be straight with the client about this, because it is also the argument for
paying later.

| | Free (Google) | Paid (OpenAI + Replicate) |
| --- | --- | --- |
| Reads the room | yes | yes |
| Counts what is there | yes | yes |
| Three directions | yes | yes |
| Talks it through | yes | yes |
| Redraws the room | yes | yes |
| **Holds doors and windows** | **asked for** | **enforced** |
| Cost | £0 | ~₹1–3 to read, ₹3–8 to draw |

That row in bold is the whole difference, and it is worth explaining plainly:

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
