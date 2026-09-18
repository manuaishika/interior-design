"""Tests for TODO.md #5's photo half: "photos, both ways."

`image_url` already worked — it is just a link. `image_file` did not: the
column existed and validated, but nothing ever copied the file in or served
it, so a product imported with only a filename showed no photo at all. This
pins the three pieces that make it real: photo_url_for resolving either
column to one fetchable URL, the importer actually copying files in from
the folder that came with the sheet, and the app serving what it copied.
"""

import pathlib

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient

from app import store
from app.catalogue import CATALOG_IMAGES_DIR, Product, import_csv, photo_url_for
from app.config import Settings
from app.import_catalogue import _copy_photos
from app.main import app


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
        "sku": "EL-WRD-014", "name": "Hale Sliding Wardrobe 2400",
        "category": "wardrobe", "price": "96000", "currency": "INR",
        "width_mm": "2400", "depth_mm": "650", "height_mm": "2400",
        "colour": "walnut", "material": "laminate on ply",
        "style_tags": "mid-century-modern", "room_tags": "bedroom",
        "image_url": "", "image_file": "hale-wardrobe.jpg",
        "product_url": "https://eleganza.example/p/hale-wardrobe",
        "in_stock": "yes", "lead_time_days": "", "notes": "",
    }
    base.update(overrides)
    return base


class TestPhotoUrlFor:
    @pytest.mark.asyncio
    async def test_an_external_url_passes_through(self, db):
        await import_csv(db, [row(image_url="https://x.test/a.jpg", image_file="")])
        product = (await db.get(Product, 1))
        assert photo_url_for(product) == "https://x.test/a.jpg"

    @pytest.mark.asyncio
    async def test_a_local_filename_becomes_a_served_path(self, db):
        await import_csv(db, [row(image_file="hale-wardrobe.jpg")])
        product = await db.get(Product, 1)
        assert photo_url_for(product) == "/catalog-images/hale-wardrobe.jpg"

    @pytest.mark.asyncio
    async def test_url_wins_when_a_row_somehow_has_both(self, db):
        await import_csv(db, [row(image_url="https://x.test/a.jpg",
                                  image_file="also.jpg")])
        product = await db.get(Product, 1)
        assert photo_url_for(product) == "https://x.test/a.jpg"


class TestTheImporterCopiesPhotosIn:
    def test_a_present_file_is_copied(self, tmp_path, monkeypatch):
        monkeypatch.setattr("app.import_catalogue.CATALOG_IMAGES_DIR",
                            tmp_path / "dest")
        source = tmp_path / "photos"
        source.mkdir()
        (source / "hale-wardrobe.jpg").write_bytes(b"fake jpeg bytes")

        copied, missing = _copy_photos([row()], source)

        assert copied == 1
        assert missing == []
        assert (tmp_path / "dest" / "hale-wardrobe.jpg").read_bytes() == b"fake jpeg bytes"

    def test_a_missing_file_is_reported_not_raised(self, tmp_path, monkeypatch):
        monkeypatch.setattr("app.import_catalogue.CATALOG_IMAGES_DIR",
                            tmp_path / "dest")
        source = tmp_path / "photos"
        source.mkdir()

        copied, missing = _copy_photos([row()], source)

        assert copied == 0
        assert missing == ["hale-wardrobe.jpg"]

    def test_rows_with_no_image_file_are_skipped_quietly(self, tmp_path, monkeypatch):
        monkeypatch.setattr("app.import_catalogue.CATALOG_IMAGES_DIR",
                            tmp_path / "dest")
        source = tmp_path / "photos"
        source.mkdir()

        copied, missing = _copy_photos(
            [row(image_url="https://x.test/a.jpg", image_file="")], source)

        assert copied == 0
        assert missing == []

    def test_the_same_filename_twice_is_only_counted_once(self, tmp_path, monkeypatch):
        monkeypatch.setattr("app.import_catalogue.CATALOG_IMAGES_DIR",
                            tmp_path / "dest")
        source = tmp_path / "photos"
        source.mkdir()
        (source / "hale-wardrobe.jpg").write_bytes(b"x")

        copied, missing = _copy_photos(
            [row(sku="A"), row(sku="B")], source)

        assert copied == 1


class TestTheAppServesWhatItCopied:
    def test_a_copied_file_is_fetchable(self, tmp_path, monkeypatch):
        (CATALOG_IMAGES_DIR / "smoke-test-only.jpg").write_bytes(b"hello")
        try:
            with TestClient(app) as c:
                r = c.get("/catalog-images/smoke-test-only.jpg")
            assert r.status_code == 200
            assert r.content == b"hello"
        finally:
            (CATALOG_IMAGES_DIR / "smoke-test-only.jpg").unlink()

    def test_a_file_that_was_never_copied_is_a_plain_404(self):
        with TestClient(app) as c:
            r = c.get("/catalog-images/never-existed.jpg")
        assert r.status_code == 404


def page(path="static/showcase.html"):
    return pathlib.Path(path).read_text(encoding="utf-8")


def function_body(source: str, signature: str) -> str:
    start = source.index(signature)
    marker = "\n  function "
    async_marker = "\n  async function "
    end_candidates = [i for i in (
        source.find(marker, start + 1), source.find(async_marker, start + 1))
        if i != -1]
    end = min(end_candidates) if end_candidates else len(source)
    return source[start:end]


class TestTheFrontendRoutesRelativePhotosThroughApi:
    """A bare image_file becomes a path this same server serves
    (/catalog-images/...), but the studio page and the engine can be
    different hosts (see the CORS comment in main.py) — a relative path has
    to go through api() the way every other server-served image on this
    page already does, or it resolves against the wrong origin."""

    def test_a_relative_photo_url_is_routed_through_api(self):
        body = function_body(page(), "function drawProducts(")
        assert "api(photo)" in body

    def test_an_external_url_is_left_alone(self):
        body = function_body(page(), "function drawProducts(")
        assert "https?" in body

    def test_docs_copy_has_the_same_routing(self):
        body = function_body(page("docs/index.html"), "function drawProducts(")
        assert "api(photo)" in body
