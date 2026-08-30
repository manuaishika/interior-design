# AI Interior Design — room-aware generation

Upload a room photo, pick a style, get back a few renovation options — with
doors, windows and walkways left untouched.

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

## Two backends

| | `hosted` (default) | `local` |
| --- | --- | --- |
| Segment | SAM2 on Replicate | SegFormer / ADE20K, in-process |
| Label | GPT-4o, one call per region | *same pass as segmentation* |
| Generate | SD inpainting on Replicate | SD inpainting via `diffusers` |
| Needs | two paid accounts | **nothing** |
| Speed | seconds | seconds on a GPU, minutes on CPU |

Set with `BACKEND=local` or `Settings(backend="local")`.

The keyless backend collapses the read and name passes into one: ADE20K's class
list already names walls, floors, doors, windows and furnishings, so a single
forward pass returns regions *already labelled* and the per-region
vision-language calls disappear entirely.

It is weaker in three specific ways, all of them worth knowing before you judge
the output:

- **Coarser labels.** `table`, never "reclaimed oak coffee table".
- **Semantic, not per-instance.** Two chairs merge into one `chair` region.
- **Walkways are inferred, not recognised.** ADE20K has no walkway class, so
  circulation is derived as floor-minus-furniture (`local_models.derive_walkway`).
  This is the one place the hosted backend is genuinely better.

### Run it free, in Colab

`notebooks/second_draft_colab.ipynb` runs the whole pipeline on Colab's free
GPU with no keys and no card. Upload a room photo, pick a style, get several
versions back. Open it from the repo with `File → Open notebook → GitHub`.

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
| `POST /api/generate` | upload + style → N design options **and** analysis JSON |
| `POST /api/analyze` | upload → analysis JSON only (inspect what was detected) |
| `GET /api/styles` | style presets and both lock profiles |
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
  "generations": [
    { "image_base64": "iVBORw0…", "seed": 88412, "variant_index": 0,
      "inpaint_mask_base64": "iVBORw0…",
      "prompt": "Interior design photograph of this room restyled in Japandi…" },
    { "image_base64": "iVBORw0…", "seed": 88413, "variant_index": 1,
      "inpaint_mask_base64": "iVBORw0…", "prompt": "…" }
  ]
}
```

### Options

`variants=N` (default 2, max 4) renders N design options. Analysis runs **once**
and is shared: segmentation and labeling are the slow, expensive part, and every
option is constrained by the same locked-region mask. The options differ only in
what the generator invents inside the editable area — never in which parts of
the room are allowed to change.

Options are rendered concurrently and differ by seed. Passing `seed=` anchors
the run so a set of options can be reproduced exactly. If one option fails, the
others are still returned; only a clean sweep of failures is an error.

`inpaint_mask_base64` is the mask actually sent to the generator — white was
regenerated, black was preserved. The UI renders it next to the result, which
is the fastest way to see whether a door really got protected.

## How the locking works

Each labeled category maps to a lock decision in `app/config.py:LOCK_PROFILES`:

| Category | `renovate` | `restyle` | Why |
| --- | --- | --- | --- |
| `door`, `window`, `walkway` | **locked** | **locked** | structure and circulation stay put |
| `furniture`, `floor` | editable | editable | the things being redesigned |
| `clutter` | editable | editable | laundry, bags, bottles — a renovation clears these away |
| `wall` | editable | editable | *see note below* |
| `other` | editable | **locked** | the only difference between the profiles |

Two profiles, selected per request with `profile=`:

- **`renovate`** (default) — full redesign. An unidentified region gets
  regenerated, because frozen islands scattered through a renovation look worse
  than re-imagined ones.
- **`restyle`** — conservative. An unidentified region is preserved. Use when
  the room should stay recognisably itself.

`clutter` is a category of its own rather than a flavour of `furniture` or
`other`. Real room photos are full of carrier bags, laundry and water bottles,
and a renovation render is supposed to make them disappear. Sweeping them into
`other` would have preserved them under `restyle`.

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
completely, flip `"wall": True` in the relevant profile in `LOCK_PROFILES`.

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

131 tests, no network calls. SAM2, the VLM, and the generator are stubbed;
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
- **Keyless models** default to `nvidia/segformer-b4-finetuned-ade-512-512` and
  `runwayml/stable-diffusion-inpainting`, both open weights. SD 1.x inpainting is
  trained at 512px, so the local backend generates at `LOCAL_GENERATION_SIZE` and
  resizes back — generating larger than the model's native size produces mush.
- **Inpainting** defaults to `stability-ai/stable-diffusion-inpainting`. If you
  swap in an endpoint that treats black as "repaint this", set
  `INVERT_INPAINT_MASK=true`.

## Where a product catalog plugs in

Catalog matching is not implemented. The seam is already there, though: every
unlocked `furniture` object in the analysis JSON carries exactly the fields a
product lookup needs.

```
{ "label": "sofa", "mask_id": "mask_003", "category": "furniture",
  "bounding_box": {...}, "area_frac": 0.075, "locked": false }
```

`label` is the query key, `bounding_box` gives the slot's position and rough
scale, and `locked: false` marks it as something being replaced. A matching
layer would sit between `analyze_room` and `compose_inpaint_mask` in
`pipeline.py`, resolving each unlocked furniture label to a catalog SKU and
feeding the chosen product names into `build_prompt`, so the render reflects
products that actually exist.

Note that this changes the generation strategy: text-prompted inpainting can be
steered *towards* a product ("a low walnut platform bed") but cannot reproduce a
specific SKU faithfully. Rendering the actual catalog item generally needs a
reference-image-conditioned model rather than a text prompt.

## Not implemented

Explicitly out of scope for this version: clearance/traffic-flow validation,
automatic keep-vs-replace decisions, and catalog matching.
