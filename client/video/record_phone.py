"""Records client/how-it-works-phone.mp4: the phone studio, start to finish.
Run demo_server.py first (port 8767; only the paid calls are stubbed and it
hands back after.jpg). Needs Chromium with --force-device-scale-factor=2, which
is what makes the screencast sharp. Frames are stitched with ffmpeg."""
import asyncio, time, httpx, glob, os, shutil
from playwright.async_api import async_playwright
V = os.path.dirname(os.path.abspath(__file__))
CA = "/root/.ccr/ca-bundle.crt"
OVERLAY = """
(() => {
  const css = document.createElement('style');
  css.textContent = `
   body{user-select:none!important;-webkit-user-select:none!important}
   #vcap{position:fixed;left:50%;bottom:22px;transform:translateX(-50%) translateY(8px);z-index:99999;
     background:rgba(20,18,14,.93);color:#fff;font:600 17px/1.2 system-ui,-apple-system,Segoe UI,sans-serif;
     padding:9px 16px 9px 10px;border-radius:999px;opacity:0;transition:opacity .2s,transform .2s;white-space:nowrap}
   #vcap.on{opacity:1;transform:translateX(-50%) translateY(0)}
   #vcap b{display:inline-block;background:#b8472a;border-radius:50%;width:26px;height:26px;line-height:26px;text-align:center;margin-right:9px;font-size:14px}
   #vdot{position:fixed;z-index:99998;width:30px;height:30px;border-radius:50%;background:rgba(184,71,42,.5);
     border:3px solid #fff;box-shadow:0 2px 10px rgba(0,0,0,.35);pointer-events:none;left:-60px;top:-60px;transform:translate(-50%,-50%);transition:transform .1s}
   #vdot.down{transform:translate(-50%,-50%) scale(.65)}
  `;
  document.head.appendChild(css);
  const cap = document.createElement('div'); cap.id='vcap'; document.body.appendChild(cap);
  const dot = document.createElement('div'); dot.id='vdot'; document.body.appendChild(dot);
  window.__cap = (n, t) => { if(!t){cap.classList.remove('on');return;} cap.innerHTML = '<b>'+n+'</b>'+t; cap.classList.add('on'); };
  addEventListener('mousemove', e => { dot.style.left=e.clientX+'px'; dot.style.top=e.clientY+'px'; }, true);
  addEventListener('mousedown', () => dot.classList.add('down'), true);
  addEventListener('mouseup', () => dot.classList.remove('down'), true);
})();
"""
async def fonts(route):
    try:
        async with httpx.AsyncClient(verify=CA, timeout=15) as c:
            r = await c.get(route.request.url, headers={"user-agent": route.request.headers.get("user-agent", "")})
        await route.fulfill(status=r.status_code, headers={"content-type": r.headers.get("content-type", "text/css"), "access-control-allow-origin": "*"}, body=r.content)
    except Exception:
        await route.abort()

async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(executable_path="/opt/pw-browsers/chromium-1194/chrome-linux/chrome", args=["--force-device-scale-factor=2"])
        shutil.rmtree(f"{V}/recp", ignore_errors=True)
        CTX0 = time.time()
        ctx = await b.new_context(viewport={"width": 390, "height": 900}, device_scale_factor=2, is_mobile=True, has_touch=True)
        await ctx.route("**/fonts.googleapis.com/**", fonts)
        await ctx.route("**/fonts.gstatic.com/**", fonts)
        page = await ctx.new_page()
        os.makedirs(f"{V}/recp", exist_ok=True)
        cdp = await ctx.new_cdp_session(page)
        frames = []
        async def on_frame(params):
            frames.append((params["metadata"]["timestamp"], params["data"]))
            await cdp.send("Page.screencastFrameAck", {"sessionId": params["sessionId"]})
        cdp.on("Page.screencastFrame", lambda ps: asyncio.ensure_future(on_frame(ps)))
        errs = []; page.on("pageerror", lambda e: errs.append(str(e)))
        await page.goto("http://127.0.0.1:8767/#/studio")
        await page.wait_for_load_state("networkidle")
        await page.evaluate(OVERLAY)
        await page.wait_for_timeout(300)
        await page.evaluate("document.getElementById('controls').scrollIntoView()")
        await page.mouse.move(300, 500)
        await page.wait_for_timeout(500)
        T0 = time.time(); T0ts = T0
        await cdp.send("Page.startScreencast", {"format": "jpeg", "quality": 90, "maxWidth": 780, "maxHeight": 1800})
        now = lambda: round(time.time() - T0, 2)
        async def cap(n, t): await page.evaluate("([n,t]) => window.__cap(n,t)", [n, t])
        async def glide(sel, steps=10):
            box = await (await page.query_selector(sel)).bounding_box()
            await page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2, steps=steps)
        async def tap(sel, steps=10):
            await glide(sel, steps); await page.wait_for_timeout(60)
            await page.mouse.down(); await page.wait_for_timeout(70); await page.mouse.up()
        step = lambda: page.evaluate("document.getElementById('controls').dataset.cur")

        await cap("1", "Home or business?")
        await page.wait_for_timeout(250)
        await tap("#settingPills [data-id='home']"); await page.wait_for_timeout(450); print("1", now(), await step())
        await cap("2", "How far should it go?")
        await page.wait_for_timeout(150)
        await tap("#depthPills [data-id='restyle']"); await page.wait_for_timeout(450); print("2", now(), await step())
        await cap("3", "Which space is it?")
        await tap("#roomSel"); await page.wait_for_timeout(150)
        await page.select_option("#roomSel", "bedroom"); await page.wait_for_timeout(200)
        await tap("#wizNext"); await page.wait_for_timeout(400); print("3", now(), await step())
        await cap("4", "Add a photo")
        await tap("#plus", 8)
        await page.set_input_files("#photo", [f"{V}/before.jpg"]); await page.wait_for_timeout(800)
        await tap("#wizNext"); await page.wait_for_timeout(400); print("4", now(), await step())
        await cap("5", "Pick a look")
        await tap("#stylePills [data-id='scandinavian']"); await page.wait_for_timeout(350)
        await tap("#wizNext"); await page.wait_for_timeout(400); print("5", now(), await step())
        await cap("6", "Tell it more, then Reimagine")
        await tap("#text", 8)
        await page.type("#text", "Warm light, add plants", delay=18); await page.wait_for_timeout(150)
        await page.evaluate("document.getElementById('less').click()")
        await tap("#go")
        await cap("7", "Your design lands on top")
        await page.wait_for_selector(".frame img", timeout=20000)
        await page.mouse.move(300, 760, steps=4)
        await page.wait_for_timeout(2300); print("end", now(), "errors:", errs)
        await cdp.send("Page.stopScreencast")
        await page.wait_for_timeout(200)
        import base64
        t_start = None
        # keep frames from T0 on; write them with their durations for ffmpeg
        shutil.rmtree(f"{V}/frames", ignore_errors=True); os.makedirs(f"{V}/frames")
        lines = []
        # screencast timestamps are epoch seconds
        keep = [(t, d) for (t, d) in frames if t >= T0ts - 0.0]
        pre = [(t, d) for (t, d) in frames if t < T0ts]
        if pre: keep.insert(0, (T0ts, pre[-1][1]))
        print("frames", len(frames), "kept", len(keep))
        for i, (t, d) in enumerate(keep):
            open(f"{V}/frames/f{i:04d}.jpg", "wb").write(base64.b64decode(d))
            dur = (keep[i + 1][0] - t) if i + 1 < len(keep) else 0.6
            lines.append(f"file 'f{i:04d}.jpg'\nduration {max(dur, 0.001):.4f}")
        lines.append(f"file 'f{len(keep)-1:04d}.jpg'")
        open(f"{V}/frames/list.txt", "w").write("\n".join(lines))
        await ctx.close(); await b.close()
asyncio.run(main())
