# Starts the real website inside Colab and gives you a public link.
# This is the demo: a page with a prompt box, that anyone can open.
#
# Paste into one Colab cell. Runtime > Change runtime type > T4 GPU first.
# Keep the cell running — closing it stops the site.

import os
import re
import socket
import subprocess
import sys
import threading
import time

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
os.environ["BACKEND"] = "local"          # keyless: no Replicate, no OpenAI

import torch
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available()
      else "NONE — Runtime > Change runtime type > T4 GPU, then run again")

# Load the models now, so the first click in front of a client is not a wait.
print("\nDownloading the models (first time only, about 4 minutes)...")
from app.config import Settings
from app.local_models import _load_inpainter, _load_segmenter

warm = Settings(backend="local")
_load_segmenter(warm.local_seg_model)
_load_inpainter(warm.local_inpaint_model)
print("Models ready.")

# --- start the server ---------------------------------------------------------
# uvicorn.run() cannot be used here: it installs signal handlers, which only
# works on the main thread, so in a notebook it dies before serving anything.
# Build the Server by hand and switch that off.
import uvicorn

from app.main import app as fastapi_app

config = uvicorn.Config(fastapi_app, host="0.0.0.0", port=PORT,
                        log_level="warning", loop="asyncio")
server = uvicorn.Server(config)
server.install_signal_handlers = lambda: None

threading.Thread(target=server.run, daemon=True).start()

up = False
for _ in range(60):
    try:
        socket.create_connection(("127.0.0.1", PORT), timeout=1).close()
        up = True
        break
    except OSError:
        time.sleep(0.5)

if not up:
    raise SystemExit("The server did not start. Re-run this cell.")
print(f"Server is up on port {PORT}.")

# --- put it on a public URL ---------------------------------------------------
# A free Cloudflare quick tunnel: no account, no token, and the link works in
# any browser — unlike Colab's own proxy links, which are tied to your session.
print("\nOpening a public link...")
if not os.path.exists("cloudflared"):
    subprocess.run(
        ["wget", "-q", "-O", "cloudflared",
         "https://github.com/cloudflare/cloudflared/releases/latest/download/"
         "cloudflared-linux-amd64"],
        check=False,
    )
    subprocess.run(["chmod", "+x", "cloudflared"], check=False)

tunnel = subprocess.Popen(
    ["./cloudflared", "tunnel", "--url", f"http://localhost:{PORT}",
     "--no-autoupdate"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
)

public = None
deadline = time.time() + 60
while time.time() < deadline:
    line = tunnel.stdout.readline()
    if not line:
        if tunnel.poll() is not None:
            break
        continue
    found = re.search(r"https://[-\w]+\.trycloudflare\.com", line)
    if found:
        public = found.group(0)
        break

print("\n" + "=" * 66)
if public:
    print("  YOUR WEBSITE IS LIVE. Open this in any browser:")
    print("   ", public)
else:
    print("  The public tunnel did not come up. Falling back to Colab's own")
    print("  link — it only opens in this browser, on this machine:")
    try:
        from google.colab.output import eval_js
        print("   ", eval_js(f"google.colab.kernel.proxyPort({PORT})"))
    except Exception as exc:
        print("    unavailable:", exc)
print("=" * 66)
print("\n  Scroll to Studio. Drop a room photo in, pick a style, type")
print("  whatever you want in the box, press Redraw the room.")
print("\n  Leave this cell running. Stopping it takes the site down.")

# Hold the cell open so the server and tunnel stay alive.
try:
    while True:
        time.sleep(5)
except KeyboardInterrupt:
    print("\nStopped.")
