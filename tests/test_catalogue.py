"""Tests for the catalogue: import, re-import, and matching.

Job 2a-2b from BUILD-NEXT.md. The importer's whole job is to survive a real
client spreadsheet — typos included — without losing the rows that are fine,
so most of this pins rejection behaviour, not just the happy path.
"""

import csv
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy import select

from app import store
from app.catalogue import Product, import_csv, match
from app.config import Settings


@pytest.fixture
def settings(tmp_path):
    return Settings(database_url=f"sqlite+aiosqlite:///{tmp_path}/t.db")


@pytest_asyncio.fixture
async def db(settings):
    store.configure(settings.database_url)
    await store.create_tables()
    async with store.session() as session:
        yield session
    await store.dispose()


def row(**overrides):
    base = {
        "sku": "EL-DSK-007",
        "name": "Fenn Writing Desk 1200",
        "category": "desk",
        "price": "22500",
        "currency": "inr",
        "width_mm": "1200",
        "depth_mm": "600",
        "height_mm": "750",
        "colour": "walnut",
        "material": "solid oak legs",
        "style_tags": "mid-century-modern,scandinavian",
        "room_tags": "study or office;bedroom",
        "image_url": "https://eleganza.example/img/fenn-desk.jpg",
        "image_file": "",
        "product_url": "https://eleganza.example/p/fenn-desk",
        "in_stock": "yes",
        "lead_time_days": "14",
        "notes": "",
    }
    base.update(overrides)
    return base


class TestImporting:
    @pytest.mark.asyncio
    async def test_a_valid_row_is_added(self, db):
        report = await import_csv(db, [row()])
        assert report.added == 1
        assert report.updated == 0
        assert report.rejected == []

        product = await db.get(Product, 1)
        assert product.sku == "EL-DSK-007"
        assert product.currency == "INR"
        assert product.style_tags == ["mid-century-modern", "scandinavian"]
        assert product.room_tags == ["study or office", "bedroom"]
        assert product.in_stock is True

    @pytest.mark.asyncio
    async def test_reimporting_the_same_sku_updates_not_duplicates(self, db):
        await import_csv(db, [row(price="22500")])
        report = await import_csv(db, [row(price="19999")])
        assert report.added == 0
        assert report.updated == 1

        rows = list(await db.scalars(select(Product)))
        assert len(rows) == 1
        assert rows[0].price == 19999

    @pytest.mark.asyncio
    async def test_a_bad_row_is_rejected_the_rest_still_import(self, db):
        good = row(sku="EL-001")
        bad = row(sku="EL-002", category="not-a-real-category")
        report = await import_csv(db, [good, bad])
        assert report.added == 1
        assert len(report.rejected) == 1
        assert report.rejected[0].sku == "EL-002"
        assert "category" in report.rejected[0].errors[0]

    @pytest.mark.asyncio
    async def test_missing_required_column_is_rejected_with_a_reason(self, db):
        report = await import_csv(db, [row(colour="")])
        assert report.added == 0
        assert any("colour" in e for e in report.rejected[0].errors)

    @pytest.mark.asyncio
    async def test_non_positive_price_is_rejected(self, db):
        report = await import_csv(db, [row(price="0")])
        assert report.added == 0
        assert any("price" in e for e in report.rejected[0].errors)

    @pytest.mark.asyncio
    async def test_non_numeric_dimension_is_rejected(self, db):
        report = await import_csv(db, [row(width_mm="wide")])
        assert report.added == 0
        assert any("width_mm" in e for e in report.rejected[0].errors)

    @pytest.mark.asyncio
    async def test_needs_an_image_url_or_an_image_file(self, db):
        report = await import_csv(db, [row(image_url="", image_file="")])
        assert report.added == 0
        assert any("image_url" in e or "image_file" in e
                   for e in report.rejected[0].errors)

    @pytest.mark.asyncio
    async def test_image_file_alone_is_enough(self, db):
        report = await import_csv(db, [row(image_url="", image_file="desk.jpg")])
        assert report.added == 1

    @pytest.mark.asyncio
    async def test_a_url_that_does_not_look_like_one_is_rejected(self, db):
        report = await import_csv(db, [row(image_url="not-a-url")])
        assert report.added == 0
        assert any("image_url" in e for e in report.rejected[0].errors)

    @pytest.mark.asyncio
    async def test_in_stock_must_be_yes_or_no(self, db):
        report = await import_csv(db, [row(in_stock="maybe")])
        assert report.added == 0
        assert any("in_stock" in e for e in report.rejected[0].errors)

    @pytest.mark.asyncio
    async def test_the_shipped_template_imports_cleanly(self, db):
        path = Path(__file__).resolve().parent.parent / "catalogue-template.csv"
        with open(path, newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        report = await import_csv(db, rows)
        assert report.rejected == []
        assert report.added == len(rows)


class TestMatching:
    @pytest.mark.asyncio
    async def test_a_wardrobe_too_wide_for_the_alcove_is_not_suggested(self, db):
        await import_csv(db, [
            row(sku="WIDE", category="wardrobe", width_mm="2400",
                style_tags="mid-century-modern", room_tags="bedroom"),
            row(sku="NARROW", category="wardrobe", width_mm="1800",
                style_tags="mid-century-modern", room_tags="bedroom"),
        ])
        found = await match(db, "wardrobe", max_width_mm=2000)
        skus = [p.sku for p in found]
        assert "WIDE" not in skus
        assert "NARROW" in skus

    @pytest.mark.asyncio
    async def test_a_style_hit_ranks_above_a_style_miss(self, db):
        await import_csv(db, [
            row(sku="OFF-STYLE", category="desk", style_tags="industrial",
                room_tags="bedroom"),
            row(sku="ON-STYLE", category="desk", style_tags="japandi",
                room_tags="bedroom"),
        ])
        found = await match(db, "desk", style="japandi")
        assert found[0].sku == "ON-STYLE"

    @pytest.mark.asyncio
    async def test_out_of_stock_ranks_below_in_stock(self, db):
        await import_csv(db, [
            row(sku="OUT", category="chair", in_stock="no"),
            row(sku="IN", category="chair", in_stock="yes"),
        ])
        found = await match(db, "chair")
        assert found[0].sku == "IN"

    @pytest.mark.asyncio
    async def test_only_the_requested_category_comes_back(self, db):
        await import_csv(db, [
            row(sku="A-DESK", category="desk"),
            row(sku="A-SOFA", category="sofa"),
        ])
        found = await match(db, "sofa")
        assert [p.sku for p in found] == ["A-SOFA"]
