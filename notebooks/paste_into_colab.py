# =============================================================================
#  SECOND DRAFT — paste this whole thing into ONE Colab cell and press play.
#
#  1. colab.research.google.com  ->  New notebook
#  2. Runtime -> Change runtime type -> T4 GPU  -> Save
#  3. Paste this in, press the play button
#  4. It asks you for a photo. Click "Choose Files", pick your room picture.
#
#  No accounts. No keys. No card. Nothing to clone.
#  First run takes ~4 minutes (downloading the models). After that, seconds.
# =============================================================================

# ---- what you can change -----------------------------------------------------
STYLE   = "japandi"   # scandinavian | mid-century modern | industrial | japandi | bohemian | modern luxury
OPTIONS = 3           # how many different versions to make (1-4)
PROMPT  = ""          # optional, e.g. "add a reading chair by the window"
# ------------------------------------------------------------------------------

import subprocess, sys, io, base64, time

print("Installing (about a minute)...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "diffusers", "transformers", "accelerate"], check=False)

import numpy as np, torch
from PIL import Image
import matplotlib.pyplot as plt
from google.colab import files

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu":
    print("\n*** NO GPU. This will take ~10 minutes per option. ***")
    print("*** Runtime -> Change runtime type -> T4 GPU, then run again. ***\n")
else:
    print("GPU:", torch.cuda.get_device_name(0))

# ---- 1. your photo -----------------------------------------------------------
print("\nChoose your room photo:")
up = files.upload()
if not up:
    raise SystemExit("No photo chosen. Run the cell again.")

name = list(up)[0]
room = Image.open(io.BytesIO(up[name])).convert("RGB")

# Shrink to something the models are happy with, keeping the shape of the room.
W, H = room.size
scale = 768 / max(W, H)
if scale < 1:
    room = room.resize((int(W * scale) // 8 * 8, int(H * scale) // 8 * 8), Image.LANCZOS)
else:
    room = room.resize((W // 8 * 8, H // 8 * 8), Image.LANCZOS)
print(f"Loaded {name} at {room.size[0]}x{room.size[1]}")

# ---- 2. read the room --------------------------------------------------------
# One model that recognises walls, floors, doors, windows and furniture.
print("\nReading the room...")
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

SEG = "nvidia/segformer-b4-finetuned-ade-512-512"
proc = AutoImageProcessor.from_pretrained(SEG)
seg = SegformerForSemanticSegmentation.from_pretrained(SEG).to(DEVICE).eval()

inputs = {k: v.to(DEVICE) for k, v in proc(images=room, return_tensors="pt").items()}
with torch.no_grad():
    logits = seg(**inputs).logits
classes = torch.nn.functional.interpolate(
    logits, size=(room.size[1], room.size[0]), mode="bilinear", align_corners=False
).argmax(1)[0].cpu().numpy()

id2label = seg.config.id2label

# Things we must NOT touch: doors, windows, and the floor people walk on.
KEEP_WORDS = ("door", "window", "windowpane", "stairs", "stairway", "escalator")

keep = np.zeros(classes.shape, dtype=bool)
floor = np.zeros(classes.shape, dtype=bool)
furniture = np.zeros(classes.shape, dtype=bool)
found = []

for cid in np.unique(classes):
    region = classes == cid
    if region.mean() < 0.004:            # ignore specks
        continue
    label = str(id2label.get(int(cid), "")).lower()
    first = label.split(",")[0].strip()

    if any(w in label for w in KEEP_WORDS):
        keep |= region
        found.append((first, "KEPT"))
    elif "floor" in label or "rug" in label or "carpet" in label:
        floor |= region
        found.append((first, "redrawn"))
    elif "wall" in label or "ceiling" in label:
        found.append((first, "redrawn"))
    else:
        furniture |= region
        found.append((first, "redrawn"))

# The walking space = floor with nothing standing on it. Keep it clear.
walkway = floor & ~furniture
if walkway.mean() > 0.02:
    keep |= walkway
    found.append(("walking space", "KEPT"))

print(f"\nFound {len(found)} things:")
for what, how in found:
    print(f"   {what:<22} {how}")

# Grow the kept areas slightly so the redraw doesn't bleed into a doorway.
from PIL import ImageFilter
mask_img = Image.fromarray(np.where(keep, 0, 255).astype(np.uint8), "L")
mask_img = mask_img.filter(ImageFilter.MinFilter(9))

editable = (np.array(mask_img) > 127).mean()
print(f"\n{editable * 100:.0f}% of the picture will be redrawn, the rest is protected.")

# ---- 3. redraw it ------------------------------------------------------------
print("\nLoading the image generator (first time: ~3 min)...")
from diffusers import StableDiffusionInpaintPipeline

pipe = None
for candidate in ("stabilityai/stable-diffusion-2-inpainting",
                  "runwayml/stable-diffusion-inpainting",
                  "botp/stable-diffusion-v1-5-inpainting"):
    try:
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            candidate,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            safety_checker=None,
        ).to(DEVICE)
        print("Using:", candidate)
        break
    except Exception as e:
        print(f"  ({candidate} unavailable, trying another)")
if pipe is None:
    raise SystemExit("Could not load an image generator. Run the cell again.")
pipe.set_progress_bar_config(disable=True)

full = (f"interior design photograph of this room restyled in {STYLE}, "
        f"tidy and uncluttered, photorealistic, natural daylight, "
        f"same room shape and perspective. {PROMPT}").strip()
avoid = ("clutter, mess, laundry, clothes on the bed, bags, blurry, distorted, "
         "warped walls, extra doors, extra windows, low quality, watermark, text")

SIZE = 512
small_room = room.resize((SIZE, SIZE), Image.LANCZOS)
small_mask = mask_img.resize((SIZE, SIZE), Image.NEAREST)

results = []
for i in range(OPTIONS):
    print(f"Drawing option {i + 1} of {OPTIONS}...")
    t0 = time.time()
    out = pipe(
        prompt=full, negative_prompt=avoid,
        image=small_room, mask_image=small_mask,
        num_inference_steps=30, guidance_scale=7.5,
        generator=torch.Generator(device=DEVICE).manual_seed(1000 + i),
    ).images[0]
    results.append(out.resize(room.size, Image.LANCZOS))
    print(f"   done in {time.time() - t0:.0f}s")

# ---- 4. look at them ---------------------------------------------------------
panels = [(room, "Your room")]
panels += [(im, f"Option {i + 1}") for i, im in enumerate(results)]
panels.append((mask_img, "Protected areas\n(black = untouched)"))

cols = min(len(panels), 3)
rows = (len(panels) + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4.2 * rows))
flat = list(np.array(axes).flat)
for ax, (im, title) in zip(flat, panels):
    ax.imshow(im); ax.set_title(title, fontsize=13); ax.axis("off")
for ax in flat[len(panels):]:
    ax.axis("off")
plt.tight_layout(); plt.show()

# ---- 5. save them ------------------------------------------------------------
for i, im in enumerate(results):
    im.save(f"option_{i + 1}.png")
print("\nSaved:", ", ".join(f"option_{i + 1}.png" for i in range(len(results))))
print("Find them in the folder icon on the left. Right-click to download.")
