STYLE   = "japandi"   # scandinavian | mid-century modern | industrial | japandi | bohemian | modern luxury
OPTIONS = 3           # how many separate versions to make (1-4)
PROMPT  = ""          # optional extra, e.g. "add a reading chair by the window"

# Second Draft. Redraws a room from one photo, keeping doors, windows and the
# walking space exactly where they are. No accounts, no keys, no card.
# Needs a GPU: Runtime -> Change runtime type -> T4 GPU.

import subprocess, sys, io, time

print("Installing, about a minute...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "diffusers", "transformers", "accelerate"], check=False)

import numpy as np
import torch
from PIL import Image, ImageFilter
from IPython.display import display
from google.colab import files

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu":
    print("\n*** NO GPU — this will take about 10 minutes per option. ***")
    print("*** Runtime > Change runtime type > T4 GPU, then run again. ***\n")
else:
    print("GPU:", torch.cuda.get_device_name(0))

print("\nChoose your room photo:")
uploaded = files.upload()
if not uploaded:
    raise SystemExit("No photo chosen. Run the cell again.")

filename = list(uploaded)[0]
room = Image.open(io.BytesIO(uploaded[filename])).convert("RGB")

# Fit the photo to a size the models handle well, keeping the room's shape.
w, h = room.size
scale = 768 / max(w, h)
if scale < 1:
    w, h = int(w * scale), int(h * scale)
room = room.resize((max(8, w // 8 * 8), max(8, h // 8 * 8)), Image.LANCZOS)
print(f"Loaded {filename} at {room.size[0]}x{room.size[1]}")

print("\nReading the room...")
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

SEG = "nvidia/segformer-b4-finetuned-ade-512-512"
processor = AutoImageProcessor.from_pretrained(SEG)
segmenter = SegformerForSemanticSegmentation.from_pretrained(SEG).to(DEVICE).eval()

batch = {k: v.to(DEVICE) for k, v in processor(images=room, return_tensors="pt").items()}
with torch.no_grad():
    logits = segmenter(**batch).logits
classes = torch.nn.functional.interpolate(
    logits, size=(room.size[1], room.size[0]), mode="bilinear", align_corners=False
).argmax(1)[0].cpu().numpy()

id2label = segmenter.config.id2label
PROTECT = ("door", "window", "windowpane", "stairs", "stairway", "escalator")

protected = np.zeros(classes.shape, dtype=bool)
floor = np.zeros(classes.shape, dtype=bool)
furniture = np.zeros(classes.shape, dtype=bool)
found = []

for class_id in np.unique(classes):
    region = classes == class_id
    if region.mean() < 0.004:          # ignore specks
        continue
    label = str(id2label.get(int(class_id), "")).lower()
    name = label.split(",")[0].strip()

    if any(word in label for word in PROTECT):
        protected |= region
        found.append((name, "KEPT"))
    elif "floor" in label or "rug" in label or "carpet" in label:
        floor |= region
        found.append((name, "redrawn"))
    elif "wall" in label or "ceiling" in label:
        found.append((name, "redrawn"))
    else:
        furniture |= region
        found.append((name, "redrawn"))

# Walking space is floor with nothing standing on it. Keep it clear.
walkway = floor & ~furniture
if walkway.mean() > 0.02:
    protected |= walkway
    found.append(("walking space", "KEPT"))

print(f"\nFound {len(found)} things in the room:")
for name, treatment in found:
    print(f"   {name:<22} {treatment}")

# Grow the protected areas slightly so the redraw can't bleed into a doorway.
mask = Image.fromarray(np.where(protected, 0, 255).astype(np.uint8), "L")
mask = mask.filter(ImageFilter.MinFilter(9))
print(f"\n{(np.array(mask) > 127).mean() * 100:.0f}% of the picture will be redrawn.")

print("\nLoading the image generator, first time takes about 3 minutes...")
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
    except Exception:
        print(f"   ({candidate} unavailable, trying the next one)")

if pipe is None:
    raise SystemExit("Could not load an image generator. Run the cell again.")
pipe.set_progress_bar_config(disable=True)

want = (f"interior design photograph of this room restyled in {STYLE}, "
        f"tidy and uncluttered, bed neatly made, photorealistic, natural daylight, "
        f"same room shape and perspective. {PROMPT}").strip()
avoid = ("clutter, mess, laundry, clothes on the bed, bags, boxes, blurry, "
         "distorted, warped walls, extra doors, extra windows, low quality, "
         "watermark, text")

SIZE = 512
small_room = room.resize((SIZE, SIZE), Image.LANCZOS)
small_mask = mask.resize((SIZE, SIZE), Image.NEAREST)

results = []
for i in range(OPTIONS):
    print(f"\nDrawing option {i + 1} of {OPTIONS}...")
    started = time.time()
    drawn = pipe(
        prompt=want,
        negative_prompt=avoid,
        image=small_room,
        mask_image=small_mask,
        num_inference_steps=30,
        guidance_scale=7.5,
        generator=torch.Generator(device=DEVICE).manual_seed(1000 + i),
    ).images[0]
    results.append(drawn.resize(room.size, Image.LANCZOS))
    print(f"   done in {time.time() - started:.0f}s")

# Each picture on its own, full size. No collage.
print("\n\n=== YOUR ROOM ===")
display(room)

for i, picture in enumerate(results):
    print(f"\n\n=== OPTION {i + 1} ===")
    display(picture)
    picture.save(f"option_{i + 1}.png")

print("\n\n=== PROTECTED AREAS (black stayed untouched) ===")
display(mask)
mask.save("protected_areas.png")

print("\n\nSaved: " + ", ".join(f"option_{i + 1}.png" for i in range(len(results))))
print("Open the folder icon on the left to download them.")
