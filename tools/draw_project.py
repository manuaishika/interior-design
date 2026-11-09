"""Draw the "after" for every photograph in a project folder.

    OPENAI_API_KEY=sk-... python tools/draw_project.py static/projects/villa

Reads <folder>/project.json, runs each photo in before/ through the same
pipeline the studio uses (planner, drawing, self-check), and writes
after/NN.jpg. The page shows an after as soon as its number is listed in
project.json's "after", which this script updates as each one lands, so a
run that stops halfway can simply be started again: finished photos are
skipped. The same files are copied into docs/ for the GitHub Pages copy.

    --only 9,10   draw just these photos (redraws them if already done)
    --dry-run     print what would be drawn and the rough cost, draw nothing
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402

COST_PER_PHOTO = 0.30          # USD, one drawing + plan + check; a redraw adds ~0.25


def _jobs(project: dict, only: set[int] | None):
    done = set(project.get("after", []))
    for space in project["spaces"]:
        for n in space["shots"]:
            if only is not None:
                if n in only:
                    yield n, space
            elif n not in done:
                yield n, space


async def _draw(folder: Path, project: dict, n: int, space: dict, settings, lock) -> bool:
    from app.pipeline import run_pipeline

    before = (folder / "before" / f"{n:02d}.jpg").read_bytes()
    prompt = f"{project['brief']} {space['prompt']}"
    try:
        _, results = await run_pipeline(
            before, project.get("style", "modern-luxury"), settings,
            extra_prompt=prompt, room=space.get("room", ""),
            profile=space.get("profile", "renovate"), variants=1, tier="room")
    except Exception as exc:                       # noqa: BLE001 — report and move on
        print(f"  {n:02d}  failed: {exc}")
        return False
    if not results:
        print(f"  {n:02d}  no image came back")
        return False
    img = Image.open(io.BytesIO(base64.b64decode(results[0].image_base64))).convert("RGB")
    w, h = Image.open(io.BytesIO(before)).size
    img = img.resize((w, h), Image.LANCZOS)        # same frame as the before, so the slider lines up
    out = folder / "after" / f"{n:02d}.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, quality=86, optimize=True, progressive=True)
    async with lock:                               # one writer for project.json
        project["after"] = sorted(set(project.get("after", [])) | {n})
        (folder / "project.json").write_text(json.dumps(project, indent=2, ensure_ascii=False) + "\n")
        _mirror(folder, out)
    print(f"  {n:02d}  done  ({space['title']})")
    return True


def _mirror(folder: Path, after: Path) -> None:
    rel = folder.resolve().relative_to(ROOT / "static")
    docs = ROOT / "docs" / "static" / rel
    (docs / "after").mkdir(parents=True, exist_ok=True)
    shutil.copy2(after, docs / "after" / after.name)
    shutil.copy2(folder / "project.json", docs / "project.json")


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path)
    ap.add_argument("--only", default="")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--parallel", type=int, default=3)
    args = ap.parse_args()

    folder = args.folder.resolve()
    project = json.loads((folder / "project.json").read_text())
    only = {int(x) for x in args.only.split(",") if x.strip()} or None
    jobs = list(_jobs(project, only))
    print(f"{project['name']}: {len(jobs)} photo(s) to draw, "
          f"roughly ${len(jobs) * COST_PER_PHOTO:.0f}-{len(jobs) * (COST_PER_PHOTO + .25):.0f} of OpenAI usage")
    if args.dry_run or not jobs:
        return 0

    from app.config import get_settings, resolve_backend
    settings = get_settings()
    if resolve_backend(settings) != "openai" or not settings.openai_api_key:
        print("Set OPENAI_API_KEY first (in the shell or in .env).")
        return 2

    gate, lock = asyncio.Semaphore(max(1, args.parallel)), asyncio.Lock()

    async def one(n, space):
        async with gate:
            return await _draw(folder, project, n, space, settings, lock)

    ok = await asyncio.gather(*(one(n, s) for n, s in jobs))
    print(f"{sum(ok)} drawn, {len(ok) - sum(ok)} failed. Run again to retry the failures.")
    return 0 if all(ok) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
