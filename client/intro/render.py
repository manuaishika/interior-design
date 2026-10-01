"""Render intro.html frame by frame (animations scrubbed, so motion is exact).

    python render.py 1920 1080 ../second-draft-intro-16x9.mp4
    python render.py 1080 1080 ../second-draft-intro-square.mp4

Swap before.jpg / after.jpg (3:4) for another room and re-run."""
import asyncio, os, sys, shutil, subprocess, httpx
from playwright.async_api import async_playwright
D = os.path.dirname(os.path.abspath(__file__))
FF = os.environ.get("FFMPEG", "ffmpeg")
CA = os.environ.get("CA_BUNDLE", True)
FPS, SECONDS = 30, float(sys.argv[4]) if len(sys.argv) > 4 else 16.5
W, H, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
only = None
async def fonts(route):
    async with httpx.AsyncClient(verify=CA, timeout=20) as c:
        r = await c.get(route.request.url, headers={"user-agent": route.request.headers.get("user-agent", "")})
    await route.fulfill(status=r.status_code, headers={"content-type": r.headers.get("content-type", "text/css"), "access-control-allow-origin": "*"}, body=r.content)
async def main():
    fr = f"{D}/frames_{W}x{H}"; shutil.rmtree(fr, ignore_errors=True); os.makedirs(fr)
    async with async_playwright() as p:
        b = await p.chromium.launch(executable_path="/opt/pw-browsers/chromium-1194/chrome-linux/chrome")
        pg = await b.new_page(viewport={"width": W, "height": H})
        await pg.route("**/fonts.googleapis.com/**", fonts); await pg.route("**/fonts.gstatic.com/**", fonts)
        await pg.goto(f"file://{D}/intro.html"); await pg.evaluate("document.fonts.ready")
        await pg.wait_for_timeout(500)
        print("fonts:", await pg.evaluate("[...document.fonts].filter(f=>f.status=='loaded').map(f=>f.family).join(',')"))
        await pg.evaluate("document.getAnimations().forEach(a => a.pause())")
        n = int(FPS * SECONDS)
        frames = range(n) if not os.environ.get("STILLS") else [int(float(t)*FPS) for t in os.environ["STILLS"].split(",")]
        for i in frames:
            await pg.evaluate("t => document.getAnimations().forEach(a => { a.currentTime = t; })", i * 1000 / FPS)
            await pg.screenshot(path=f"{fr}/f{i:05d}.jpg", type="jpeg", quality=94)
        await b.close()
    if os.environ.get("STILLS"): print("stills in", fr); return
    subprocess.run([FF, "-v", "error", "-y", "-framerate", str(FPS), "-i", f"{fr}/f%05d.jpg",
                    "-c:v", "libx264", "-crf", "18", "-preset", "slow", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", "-an", out], check=True)
    shutil.rmtree(fr)
    print("wrote", out)
asyncio.run(main())
