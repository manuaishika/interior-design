# AI Interior Design — room-aware generation

Upload a room photo, pick a style, get a restyled room back — with doors,
windows and walkways left untouched.

The flow is `upload → analyze → style → generate`. The analysis step segments
the room, labels each region with a vision-language model, and turns the locked
regions into an inpainting mask so the generator can only repaint the rest.

```
photo ──► SAM2 (Replicate)      ──► N masks
            │
            ├──► GPT-4o per mask ──► {category, name, confidence}
            │
            └──► structured JSON ──► {label, mask_id, bounding_box, locked}
                       │
                       └──► locked regions ──► inpainting mask ──► generation
```

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env          # add REPLICATE_API_TOKEN and OPENAI_API_KEY
.venv/bin/uvicorn app.main:app --reload
```

Open http://localhost:8000.

## API

| Endpoint | Purpose |
| --- | --- |
| `POST /api/generate` | upload + style → generated image **and** analysis JSON |
| `POST /api/analyze` | upload → analysis JSON only (inspect what was detected) |
| `GET /api/styles` | style presets and the active lock policy |
| `GET /api/health` | which credentials and models are configured |

```bash
curl -X POST localhost:8000/api/generate \
  -F photo=@room.jpg -F style=japandi | jq '.analysis'
```

`/api/generate` returns both halves:

```json
{
  "analysis": {
    "image_width": 1536, "image_height": 1024,
    "masks_returned": 47, "masks_labeled": 18,
    "objects": [
      { "label": "sofa",   "mask_id": "mask_003",
        "bounding_box": {"x": 120, "y": 540, "width": 480, "height": 300},
        "locked": false, "category": "furniture", "confidence": 0.94,
        "area_px": 118400, "area_frac": 0.075, "lock_source": "policy" },
      { "label": "window", "mask_id": "mask_007",
        "bounding_box": {"x": 980, "y": 180, "width": 300, "height": 520},
        "locked": true,  "category": "window",   "confidence": 0.97 }
    ]
  },
  "generation": {
    "image_base64": "iVBORw0…",
    "inpaint_mask_base64": "iVBORw0…",
    "image_url": "https://replicate.delivery/…",
    "prompt": "Interior design photograph of this room restyled in Japandi…"
  }
}
```

`inpaint_mask_base64` is the mask actually sent to the generator — white was
regenerated, black was preserved. The UI renders it next to the result, which
is the fastest way to see whether a door really got protected.

## How the locking works

Each labeled category maps to a lock decision in `app/config.py:LOCK_POLICY`:

| Category | Locked | Why |
| --- | --- | --- |
| `door`, `window`, `walkway` | **yes** | per spec — structure and circulation stay put |
| `furniture`, `floor` | no | per spec — the things being restyled |
| `wall` | no | *see note below* |
| `other` | **yes** | an unidentified region is safer preserved |

Two deliberate choices in `imaging.build_inpaint_mask`:

- **The canvas starts editable.** Pixels no SAM2 mask covered get regenerated —
  unsegmented background is open space, not something to protect.
- **Locked always wins overlaps.** Masks overlap constantly; where an unlocked
  sofa crosses a locked window, the window wins. The failure we care about is
  destroying a doorway, not under-editing a couch.

Locked regions are also dilated by `LOCKED_DILATION_PX` (default 12px) before
generation, because diffusion models bleed across mask boundaries and a hard
edge tends to eat into door frames and window reveals.

### Note on walls

The spec named doors, windows and walkways as locked, and furniture and open
floor as unlocked. Walls were not specified, so they default to **unlocked** —
repainting walls is a core interior-design edit and locking them would freeze
the largest surface in most rooms. If you would rather preserve room geometry
completely, flip `"wall": True` in `LOCK_POLICY`.

## Cost control

SAM2's automatic mode returns 50–150 masks on a busy room photo, and every
surviving mask is one VLM call. Before labeling, masks are filtered by area
(specks and whole-image masks dropped), deduplicated by IoU, and capped at
`MAX_MASKS` (default 24) largest-first. Labeling runs concurrently, bounded by
`LABEL_CONCURRENCY`.

## Keep vs. replace

Deciding *which* existing furniture to keep is out of scope, but the input path
is wired: pass `keep_mask_ids` / `replace_mask_ids` (comma-separated) to either
endpoint to override the policy per region. Overrides are reported as
`"lock_source": "user_override"` in the JSON. Run `/api/analyze` first to get
the mask ids.

## Tests

```bash
.venv/bin/python -m pytest
```

82 tests, no network calls. SAM2, the VLM, and the generator are stubbed;
everything between the upload bytes and the generator payload is real code —
image normalisation, mask filtering, bbox extraction, lock policy, and mask
composition.

## Model notes

- **SAM2** runs unpinned as `meta/sam-2`, so Replicate resolves the latest
  version. Pin `meta/sam-2:<version-sha>` for reproducibility. Output parsing
  is defensive (`segmentation._extract_mask_refs`) because Replicate output
  schemas drift between versions.
- **GPT-4V**: the original `gpt-4-vision-preview` checkpoint has been retired
  by OpenAI. `gpt-4o` is its vision-capable successor and speaks the same
  message format; override with `VLM_MODEL`.
- **Inpainting** defaults to `stability-ai/stable-diffusion-inpainting`. If you
  swap in an endpoint that treats black as "repaint this", set
  `INVERT_INPAINT_MASK=true`.

## Not implemented

Explicitly out of scope for this version: clearance/traffic-flow validation,
automatic keep-vs-replace decisions, and catalog matching.
