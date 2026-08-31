# Starts the actual website in Colab and gives you a link to open.
# This is what you show in the meeting: a real page, with a prompt box,
# where you type what you want and press a button.
#
# Paste into one Colab cell. Runtime > Change runtime type > T4 GPU first.

import subprocess, sys, os, time, threading, socket

REPO = "https://github.com/manuaishika/interior-design"
BRANCH = "claude/interior-design-room-analysis-yj4fvg"
PORT = 8000

print("Installing (2-3 minutes)...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "diffusers", "transformers", "accelerate", "scipy",
                "fastapi", "uvicorn", "python-multipart", "pydantic-settings"],
               check=False)

if not os.path.isdir("/content/interior-design"):
    subprocess.run(["git", "clone", "-q", "-b", BRANCH, REPO,
                    "/content/interior-design"], check=False)
else:
    subprocess.run(["git", "-C", "/content/interior-design", "pull", "-q"], check=False)

os.chdir("/content/interior-design")
sys.path.insert(0, "/content/interior-design")

# Keyless: no Replicate, no OpenAI, no card.
os.environ["BACKEND"] = "local"

import torch
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available()
      else "NONE — Runtime > Change runtime type > T4 GPU, then run again")

# Warm the models up now, so the first click in the browser isn't a 4-minute wait.
print("\nDownloading the models (first time only, ~4 minutes)...")
from app.config import Settings
from app.local_models import _load_segmenter, _load_inpainter

warm = Settings(backend="local")
_load_segmenter(warm.local_seg_model)
_load_inpainter(warm.local_inpaint_model)
print("Models ready.")

def serve():
    import uvicorn
    from app.main import app
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")

threading.Thread(target=serve, daemon=True).start()

for _ in range(40):
    try:
        socket.create_connection(("127.0.0.1", PORT), timeout=1).close()
        break
    except OSError:
        time.sleep(0.5)

from google.colab.output import eval_js
url = eval_js(f"google.colab.kernel.proxyPort({PORT})")

print("\n" + "=" * 62)
print("  YOUR WEBSITE IS LIVE — open this link:")
print(" ", url)
print("=" * 62)
print("\n  Scroll to Studio. Drop a room photo in, pick a style,")
print("  type anything you want in the box, press Redraw the room.")
print("\n  Keep this cell running. Closing it stops the site.")
