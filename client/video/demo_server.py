"""Real app, real routes. Only the paid calls are stubbed, and they hand back stand-in pictures."""
import asyncio, io, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
V = os.path.dirname(os.path.abspath(__file__))
os.environ["OPENAI_API_KEY"] = "sk-fake"
os.environ["WHATSAPP_NUMBER"] = "971500000000"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{V}/demo.db"
from PIL import Image
import app.openai_images as oi, app.main as main

def png(name):
    b = io.BytesIO(); Image.open(f"{V}/{name}").convert("RGB").save(b, "PNG"); return b.getvalue()
async def find(image, s): return []
async def redraw(image, mask, prompt, s, **kw):
    await asyncio.sleep(1.8 if "marked area" not in prompt else 1.0)
    return png("edited.jpg" if "marked area" in prompt else "after.jpg")
async def read_room(photo, room_type, settings, currency="", market=""):
    return {"is_room": True, "room": "A bedroom with one bed, one door and one window.",
            "items": [{"name": "bed", "count": 1, "treatment": "redraw"}, {"name": "door", "count": 1, "treatment": "keep"},
                      {"name": "window", "count": 1, "treatment": "keep"}],
            "directions": [], "budget": {"currency": "AED", "lines": [], "assumes": ""}}
oi.find_structure = find; oi.redraw = redraw; main.read_room = read_room
import uvicorn
uvicorn.run(main.app, host="127.0.0.1", port=8767, log_level="warning")
