import asyncio, time, httpx, glob, os, shutil
from playwright.async_api import async_playwright
V = os.path.dirname(os.path.abspath(__file__))
CA = "/root/.ccr/ca-bundle.crt"

OVERLAY = """
(() => {
  const css = document.createElement('style');
  css.textContent = `
   #vcap{position:fixed;left:50%;bottom:34px;transform:translateX(-50%) translateY(14px);z-index:99999;
     background:rgba(20,18,14,.92);color:#fff;font:600 30px/1.2 system-ui,-apple-system,Segoe UI,sans-serif;
     padding:16px 30px;border-radius:999px;opacity:0;transition:opacity .25s,transform .25s;white-space:nowrap;letter-spacing:.2px}
   #vcap.on{opacity:1;transform:translateX(-50%) translateY(0)}
   #vcap b{display:inline-block;background:#b8472a;border-radius:50%;width:38px;height:38px;line-height:38px;text-align:center;margin-right:14px;font-size:22px}
   #vdot{position:fixed;z-index:99998;width:26px;height:26px;border-radius:50%;background:rgba(184,71,42,.55);
     border:3px solid #fff;box-shadow:0 2px 10px rgba(0,0,0,.35);pointer-events:none;left:-50px;top:-50px;transform:translate(-50%,-50%);transition:transform .12s}
   body{user-select:none!important;-webkit-user-select:none!important}
   #vdot.down{transform:translate(-50%,-50%) scale(.7)}
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
        b = await p.chromium.launch(executable_path="/opt/pw-browsers/chromium-1194/chrome-linux/chrome")
        shutil.rmtree(f"{V}/rec", ignore_errors=True)
        CTX0 = time.time()
        ctx = await b.new_context(viewport={"width": 1600, "height": 900},
                                  record_video_dir=f"{V}/rec", record_video_size={"width": 1280, "height": 720})
        await ctx.route("**/fonts.googleapis.com/**", fonts)
        await ctx.route("**/fonts.gstatic.com/**", fonts)
        page = await ctx.new_page()
        errs = []; page.on("pageerror", lambda e: errs.append(str(e)))
        await page.goto("http://127.0.0.1:8767/#/studio")
        await page.wait_for_load_state("networkidle")
        await page.evaluate(OVERLAY)
        await page.evaluate("window.scrollTo(0, 340)")
        await page.mouse.move(800, 200)
        await page.wait_for_timeout(700)
        T0 = time.time(); open(f'{V}/t0.txt','w').write(str(T0-CTX0))
        def now(): return round(time.time() - T0, 2)

        async def cap(n, t): await page.evaluate("([n,t]) => window.__cap(n,t)", [n, t])
        async def glide(sel, dx=0, dy=0, steps=14):
            box = await (await page.query_selector(sel)).bounding_box()
            await page.mouse.move(box["x"] + box["width"] / 2 + dx, box["y"] + box["height"] / 2 + dy, steps=steps)
        async def tap(sel):
            await glide(sel); await page.wait_for_timeout(120)
            await page.mouse.down(); await page.wait_for_timeout(90); await page.mouse.up()

        # 1 — how far
        await cap("1", "Choose how far to go")
        await page.wait_for_timeout(200)
        await tap("#depthPills [data-id='restyle']")
        await page.wait_for_timeout(350); print("1", now())

        # 2 — photo
        await cap("2", "Add a photo of your room")
        await glide("#plus", steps=8); await page.wait_for_timeout(100)
        await page.set_input_files("#photo", [f"{V}/before.jpg"])
        await page.wait_for_timeout(800); print("2", now())

        # 3 — comments, room, look
        await cap("3", "Say what you want, pick a look")
        await glide("#text"); await page.mouse.down(); await page.mouse.up()
        await page.type("#text", "Warm light, add a plant", delay=20)
        await page.wait_for_timeout(100)
        await page.evaluate("document.querySelector('#less').click()")
        await tap("#stylePills [data-id='scandinavian']")
        await page.wait_for_timeout(450); print("3", now())

        # 4 — go
        await cap("4", "Tap Reimagine")
        await page.evaluate("window.scrollTo({top: 640, behavior: 'smooth'})")
        await page.wait_for_timeout(550)
        await tap("#go")
        await page.wait_for_selector(".frame img", timeout=15000)
        await page.mouse.move(1150, 330, steps=6)
        await page.wait_for_timeout(300); print("4 done", now())

        # 5 — result
        await cap("5", "Same room, redrawn")
        await page.evaluate("document.querySelector('#shots').scrollIntoView({behavior:'smooth', block:'start'})")
        await page.wait_for_timeout(1000); print("5", now())

        # 6 — markup
        await cap("6", "Circle anything to change")
        await tap(".mark-toggle")
        await page.wait_for_timeout(200)
        cv = await (await page.query_selector(".mark-canvas")).bounding_box()
        pts = [(0.10,0.22),(0.16,0.12),(0.26,0.10),(0.34,0.20),(0.32,0.42),(0.22,0.50),(0.12,0.44),(0.10,0.30)]
        await page.mouse.move(cv["x"] + cv["width"] * pts[0][0], cv["y"] + cv["height"] * pts[0][1], steps=8)
        await page.mouse.down()
        for fx, fy in pts[1:]:
            await page.mouse.move(cv["x"] + cv["width"] * fx, cv["y"] + cv["height"] * fy, steps=6)
        await page.mouse.up(); await page.wait_for_timeout(200)
        await glide(".mark-toolbar input"); await page.mouse.down(); await page.mouse.up()
        await page.type(".mark-toolbar input", "Make this wall sage green", delay=20)
        await page.wait_for_timeout(250)
        await tap(".mark-toolbar .cta.clay")
        await page.wait_for_selector("#shots figure:nth-of-type(2) img", timeout=15000)
        await page.wait_for_timeout(400); print("6 done", now())

        # 7 — talk to a designer
        await cap("7", "Happy? Talk to a designer")
        await page.evaluate("document.querySelector('#contact').scrollIntoView({behavior:'smooth', block:'center'})")
        await page.wait_for_timeout(1000)
        await glide("#waGo", steps=10); await page.wait_for_timeout(450)
        print("end", now(), "errors:", errs)
        await ctx.close(); await b.close()
        vids = glob.glob(f"{V}/rec/*.webm"); print(vids)
asyncio.run(main())
