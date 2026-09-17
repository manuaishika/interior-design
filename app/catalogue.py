"""The client's own furniture, stored and matched against.

Why this exists
----------------
Without a catalogue a render invents furniture and the costing table
estimates what it might cost. `CATALOGUE.md` has the schema and the honest
limit on what a render can promise; this module is the other half — storing
what a client sends, matching it against what a room needs, and reporting
plainly on what a re-import did.

Re-importable, on purpose
-------------------------
A client's spreadsheet gets typos fixed and rows added over months, and it
comes back as a whole file each time, not a diff. `import_csv` upserts on
`sku` so sending the same file twice is harmless, and it rejects a bad row
without losing the other four hundred — a catalogue that fails silently on
one bad row is a catalogue nobody trusts weeks later.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Iterable

from sqlalchemy import Boolean, DateTime, Float, Integer, JSON, String, Text, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from .store import Base, now

# The fixed list a category must come from — this is what matching keys off.
# A free-text category matches nothing, which is the whole point of the
# column being closed rather than open (see CATALOGUE.md).
CATEGORIES = (
    "bed", "sofa", "chair", "table", "desk", "wardrobe", "storage",
    "lighting", "rug", "soft-furnishing", "decor", "appliance",
)

# Columns that must have a value on every row, regardless of category.
# image_url and image_file are handled separately below: a row needs one of
# the two, not both.
REQUIRED_COLUMNS = (
    "sku", "name", "category", "price", "currency",
    "width_mm", "depth_mm", "height_mm",
    "colour", "material", "style_tags", "room_tags",
    "product_url", "in_stock",
)


class Product(Base):
    __tablename__ = "products"

    id: Mapped[int] = mapped_column(primary_key=True)
    # Their own code, unique. The one thing that must never change — it is
    # what a re-import matches an existing row against.
    sku: Mapped[str] = mapped_column(String(120), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(200))
    category: Mapped[str] = mapped_column(String(40), index=True)
    price: Mapped[float] = mapped_column(Float)
    currency: Mapped[str] = mapped_column(String(8))
    width_mm: Mapped[int] = mapped_column(Integer)
    depth_mm: Mapped[int] = mapped_column(Integer)
    height_mm: Mapped[int] = mapped_column(Integer)
    colour: Mapped[str] = mapped_column(String(80))
    material: Mapped[str] = mapped_column(String(120))
    # Lists, not comma strings to be LIKE-matched later — JSON is the one
    # column type SQLAlchemy can put through both SQLite and Postgres
    # unchanged, which matters because this project runs on both.
    style_tags: Mapped[list] = mapped_column(JSON, default=list)
    room_tags: Mapped[list] = mapped_column(JSON, default=list)
    image_url: Mapped[str | None] = mapped_column(String(500), default=None)
    image_file: Mapped[str | None] = mapped_column(String(300), default=None)
    product_url: Mapped[str] = mapped_column(String(500))
    in_stock: Mapped[bool] = mapped_column(Boolean, default=True)
    lead_time_days: Mapped[int | None] = mapped_column(Integer, default=None)
    notes: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True),
                                                    default=now)
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True),
                                                     default=now)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _tags(raw: str, sep: str) -> list[str]:
    return [part.strip() for part in (raw or "").split(sep) if part.strip()]


def _positive_int(raw: str, field_name: str, errors: list[str]) -> int:
    try:
        value = int(float(raw))
    except (TypeError, ValueError):
        errors.append(f"{field_name} is not a number: {raw!r}")
        return 0
    if value <= 0:
        errors.append(f"{field_name} must be a positive number, got {raw!r}")
    return value


def parse_row(row: dict[str, Any]) -> tuple[dict | None, list[str]]:
    """Turn one CSV row into `Product` kwargs, or a list of reasons it can't.

    Every check runs before returning, so a row with three problems is
    reported with three, not just the first one hit — the client fixes a
    sheet once, not once per re-import.
    """
    errors: list[str] = []
    row = {k: (v or "").strip() if isinstance(v, str) else v
          for k, v in row.items()}

    for column in REQUIRED_COLUMNS:
        if not row.get(column):
            errors.append(f"{column} is required")

    sku = row.get("sku") or ""
    category = (row.get("category") or "").strip().lower()
    if category and category not in CATEGORIES:
        errors.append(
            f"category {category!r} is not one of {', '.join(CATEGORIES)}")

    price = 0.0
    if row.get("price"):
        try:
            price = float(row["price"])
            if price <= 0:
                errors.append(f"price must be a positive number, got {row['price']!r}")
        except (TypeError, ValueError):
            errors.append(f"price is not a number: {row['price']!r}")

    width_mm = _positive_int(row.get("width_mm", ""), "width_mm", errors) if row.get("width_mm") else 0
    depth_mm = _positive_int(row.get("depth_mm", ""), "depth_mm", errors) if row.get("depth_mm") else 0
    height_mm = _positive_int(row.get("height_mm", ""), "height_mm", errors) if row.get("height_mm") else 0

    image_url = (row.get("image_url") or "").strip() or None
    image_file = (row.get("image_file") or "").strip() or None
    if not image_url and not image_file:
        errors.append("one of image_url or image_file is required")
    if image_url and not (image_url.startswith("http://") or image_url.startswith("https://")):
        errors.append(f"image_url does not look like a URL: {image_url!r}")

    in_stock_raw = (row.get("in_stock") or "").strip().lower()
    in_stock = in_stock_raw == "yes"
    if in_stock_raw not in ("yes", "no"):
        errors.append(f"in_stock must be yes or no, got {row.get('in_stock')!r}")

    lead_time_days = None
    if row.get("lead_time_days"):
        try:
            lead_time_days = int(float(row["lead_time_days"]))
        except (TypeError, ValueError):
            errors.append(f"lead_time_days is not a number: {row['lead_time_days']!r}")

    if errors:
        return None, errors

    return {
        "sku": sku,
        "name": row["name"],
        "category": category,
        "price": price,
        "currency": (row.get("currency") or "").strip().upper(),
        "width_mm": width_mm,
        "depth_mm": depth_mm,
        "height_mm": height_mm,
        "colour": row["colour"],
        "material": row["material"],
        "style_tags": _tags(row.get("style_tags", ""), ","),
        "room_tags": _tags(row.get("room_tags", ""), ";"),
        "image_url": image_url,
        "image_file": image_file,
        "product_url": row["product_url"],
        "in_stock": in_stock,
        "lead_time_days": lead_time_days,
        "notes": (row.get("notes") or "").strip(),
    }, []


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------

@dataclass
class Rejection:
    row_number: int   # 1-based, matching what a spreadsheet shows
    sku: str
    errors: list[str]


@dataclass
class ImportReport:
    added: int = 0
    updated: int = 0
    rejected: list[Rejection] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [f"{self.added} added, {self.updated} updated, "
                f"{len(self.rejected)} rejected"]
        for r in self.rejected:
            who = r.sku or "(no sku)"
            lines.append(f"  row {r.row_number} [{who}]: {'; '.join(r.errors)}")
        return "\n".join(lines)


async def upsert_product(db: AsyncSession, kwargs: dict) -> tuple[Product, bool]:
    """Insert, or update in place if the sku already exists.

    Returns the product and whether it was newly created, so a caller can
    report added vs. updated the way `import_csv` does.
    """
    existing = await db.scalar(select(Product).where(Product.sku == kwargs["sku"]))
    if existing is None:
        product = Product(**kwargs)
        db.add(product)
        return product, True
    for key, value in kwargs.items():
        setattr(existing, key, value)
    existing.updated_at = now()
    return existing, False


async def import_csv(db: AsyncSession, rows: Iterable[dict]) -> ImportReport:
    """Upsert every valid row; reject the row, never the file.

    One commit at the end — SQLite in these tests does not roll back a
    half-written import cleanly, and there is nothing here that needs each
    row visible to the next one.
    """
    report = ImportReport()
    for row_number, row in enumerate(rows, start=1):
        kwargs, errors = parse_row(row)
        if kwargs is None:
            report.rejected.append(
                Rejection(row_number=row_number,
                          sku=str(row.get("sku", "")).strip(), errors=errors))
            continue
        _, created = await upsert_product(db, kwargs)
        if created:
            report.added += 1
        else:
            report.updated += 1
    await db.commit()
    return report


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def _tag_hit(tags: list[str], wanted: str | None) -> bool:
    if not wanted:
        return False
    wanted = wanted.strip().lower()
    return any(wanted in tag.lower() or tag.lower() in wanted for tag in tags)


async def match(db: AsyncSession, category: str, *, style: str | None = None,
                room: str | None = None, max_width_mm: int | None = None,
                budget: float | None = None, limit: int = 5) -> list[Product]:
    """Find products for one item a room needs.

    The coarse cut — category, and a size that actually fits the alcove —
    happens in SQL; ranking within that is a handful of rows, so it is done
    in Python rather than reaching for a database-specific JSON operator
    that SQLite and Postgres would each need their own version of. No ML,
    no embeddings: a sensible ORDER BY, just applied in two stages instead
    of one.

    A 2400mm wardrobe does not go in a 2000mm alcove — that is the one
    check that earns this function its keep.
    """
    category = (category or "").strip().lower()
    stmt = select(Product).where(Product.category == category)
    if max_width_mm:
        stmt = stmt.where(Product.width_mm <= max_width_mm)
    if budget:
        stmt = stmt.where(Product.price <= budget)
    candidates = list(await db.scalars(stmt))

    def sort_key(p: Product):
        return (
            not _tag_hit(p.style_tags, style),     # style hit first
            not _tag_hit(p.room_tags, room),       # room hit first
            not p.in_stock,                        # in-stock first
            p.price,                               # cheaper first among ties
        )

    candidates.sort(key=sort_key)
    return candidates[:limit]


async def products_for_category(db: AsyncSession, category: str) -> list[Product]:
    rows = await db.scalars(
        select(Product).where(Product.category == category.strip().lower())
        .order_by(Product.price))
    return list(rows)


async def all_products(db: AsyncSession) -> list[Product]:
    rows = await db.scalars(select(Product).order_by(Product.category, Product.name))
    return list(rows)
