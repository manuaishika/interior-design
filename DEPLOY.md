# Deploying it

The app is one FastAPI service. What it can do depends only on which keys it
has.

| Feature | Needs | GPU? |
| --- | --- | --- |
| Read the room, three directions | `OPENAI_API_KEY` | no |
| Talk it through | `OPENAI_API_KEY` | no |
| Draw the designs | `REPLICATE_API_TOKEN` | no — Replicate has the GPUs |

Nothing here needs a GPU of your own. That is the point of `BACKEND=hosted`.

## Vercel

1. Push this branch to GitHub (already done).
2. On vercel.com: **Add New → Project**, import `manuaishika/interior-design`.
3. Settings → Environment Variables:

   ```
   BACKEND            = hosted
   OPENAI_API_KEY     = sk-...
   REPLICATE_API_TOKEN = r8_...
   ```

4. Deploy. `vercel.json` and `api/index.py` are already here, so there is
   nothing to configure.

**One catch worth knowing before you pick a plan.** Generating three images
takes 30–60 seconds. Vercel's Hobby plan cuts requests off at 10 seconds, so
reading and chat will work but drawing will time out. `maxDuration` is set to
60, which needs a Pro plan. On Hobby, deploy the site for reading and run
drawing somewhere without that limit.

## Render / Railway / Fly — no timeout problem

```
Build:  pip install -r requirements.txt
Start:  uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

Same three environment variables. These have no request timeout, so drawing
works on the free tier.

## Your own machine

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env          # add your keys
.venv/bin/uvicorn app.main:app
```

Open http://localhost:8000.

## Running it free, with no keys at all

`BACKEND=local` runs open-weight models in the process instead. It needs a GPU
to be usable, which is what `notebooks/run_website_colab.py` is for — it starts
this same site on Colab's free GPU and prints a public link. Good for a demo,
not for a product.
