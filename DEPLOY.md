# Putting it online

Four steps, about twenty minutes, mostly waiting.

## 1 — Replicate (draws the pictures)

1. Sign up at **replicate.com** with GitHub
2. Billing → add a card
3. **replicate.com/account/api-tokens** → copy the token (starts `r8_`)

Costs a few paise per image. A hundred rooms is a few hundred rupees.

## 2 — OpenAI (reads the room, does the chat)

1. Sign up at **platform.openai.com**
2. Billing → add a card, put $10 on it
3. API keys → create one → copy it (starts `sk-`)

This is **not** ChatGPT Plus. Different product, separate billing. A ChatGPT
subscription gives you nothing here.

## 3 — Render (runs the site)

1. Sign up at **render.com** with GitHub
2. **New → Blueprint**
3. Pick `manuaishika/interior-design`, branch
   `claude/interior-design-room-analysis-yj4fvg`
4. It reads `render.yaml` and fills everything in. It will ask for two values:

   | | |
   | --- | --- |
   | `OPENAI_API_KEY` | the `sk-...` from step 2 |
   | `REPLICATE_API_TOKEN` | the `r8_...` from step 1 |

5. **Apply**. First build takes about five minutes.

You get a URL like `https://second-draft.onrender.com`. That is the site.
Send it to anyone.

## 4 — Check it

Open `your-url/api/health`. Both of these must say `true`:

```json
{ "can_read": true, "can_draw": true }
```

If either is `false`, the key for that half is missing or wrong. Render →
Environment → fix it → Manual Deploy.

Then open the site, upload a room, press **Design**.

---

## Why Render and not Vercel

Drawing three images takes 30–60 seconds. Vercel's Hobby plan cuts requests
off at 10 seconds, so drawing would always fail there. Render has no request
timeout. `vercel.json` is still in the repo if you want Vercel for the reading
half on a Pro plan, but Render is the one that works.

**One thing about Render's free tier:** it sleeps after 15 minutes idle, and
the next visitor waits ~50 seconds while it wakes. Fine for testing, bad in
front of a client. The `starter` plan in `render.yaml` (about $7/month) stays
awake. Open the site a minute before any demo either way.

## Running it on your own machine

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env          # paste the same two keys
.venv/bin/uvicorn app.main:app
```

http://localhost:8000

## Free, with no keys at all

`notebooks/run_website_colab.py` runs this same site on Colab's free GPU with
open-weight models instead. No accounts, no card. Good enough to prove the
idea; not something to put in front of customers, because the link dies with
the notebook.
