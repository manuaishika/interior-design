"""Load a client's catalogue CSV into the database.

    python -m app.import_catalogue path/to/catalogue.csv

Re-runnable: upserts on `sku`, and reports what was added, updated and
rejected — and why, for each rejection — so a v2 sheet with typos in it
loses only the bad rows, not the other four hundred. See CATALOGUE.md for
the schema this expects.
"""

from __future__ import annotations

import asyncio
import csv
import sys

from . import store
from .catalogue import import_csv
from .config import get_settings


async def _run(path: str) -> int:
    settings = get_settings()
    store.configure(settings.database_url)
    await store.create_tables()
    try:
        with open(path, newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            print(f"{path} has no rows.")
            return 1
        async with store.session() as db:
            report = await import_csv(db, rows)
        print(report)
        return 0
    finally:
        await store.dispose()


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python -m app.import_catalogue <path-to-csv>")
        raise SystemExit(2)
    raise SystemExit(asyncio.run(_run(sys.argv[1])))


if __name__ == "__main__":
    main()
