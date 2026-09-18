"""Load a client's catalogue CSV into the database.

    python -m app.import_catalogue [path/to/catalogue.csv] [path/to/photos]

Both arguments are optional. With none, it reads `data/catalog.csv` — where
data/README.md tells you to put the client's returned file. The second
argument is a folder of JPEGs, for a client who sent photos rather than a
website: any `image_file` value in the sheet is copied from that folder
into `data/catalog-images/`, which the app serves at /catalog-images.

Re-runnable: upserts on `sku`, and reports what was added, updated and
rejected — and why, for each rejection — so a v2 sheet with typos in it
loses only the bad rows, not the other four hundred. See CATALOGUE.md for
the schema this expects.
"""

from __future__ import annotations

import asyncio
import csv
import shutil
import sys
from pathlib import Path

from . import store
from .catalogue import CATALOG_IMAGES_DIR, import_csv
from .config import get_settings

DEFAULT_CSV = Path(__file__).resolve().parent.parent / "data" / "catalog.csv"


def _copy_photos(rows: list[dict], photos_dir: Path) -> tuple[int, list[str]]:
    """Copy every row's `image_file` in from the folder that came with the
    sheet. Missing files are reported, not raised — a photo that has not
    arrived yet must not stop the rest of the catalogue from importing."""
    CATALOG_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    copied = 0
    missing: list[str] = []
    seen: set[str] = set()
    for row in rows:
        name = (row.get("image_file") or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        source = photos_dir / name
        if not source.is_file():
            missing.append(name)
            continue
        shutil.copy2(source, CATALOG_IMAGES_DIR / name)
        copied += 1
    return copied, missing


async def _run(csv_path: Path, photos_dir: Path | None) -> int:
    settings = get_settings()
    store.configure(settings.database_url)
    await store.create_tables()
    try:
        if not csv_path.is_file():
            print(f"{csv_path} does not exist.")
            return 1
        with open(csv_path, newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            print(f"{csv_path} has no rows.")
            return 1

        if photos_dir is not None:
            copied, missing = _copy_photos(rows, photos_dir)
            plural = "s" if copied != 1 else ""
            print(f"{copied} photo{plural} copied into {CATALOG_IMAGES_DIR}")
            if missing:
                print(f"{len(missing)} named in the sheet but not found in "
                     f"{photos_dir}: {', '.join(missing)}")

        async with store.session() as db:
            report = await import_csv(db, rows)
        print(report)
        return 0
    finally:
        await store.dispose()


def main() -> None:
    if len(sys.argv) > 3:
        print("Usage: python -m app.import_catalogue [path-to-csv] [path-to-photos]")
        raise SystemExit(2)
    csv_path = Path(sys.argv[1]) if len(sys.argv) >= 2 else DEFAULT_CSV
    photos_dir = Path(sys.argv[2]) if len(sys.argv) >= 3 else None
    raise SystemExit(asyncio.run(_run(csv_path, photos_dir)))


if __name__ == "__main__":
    main()
