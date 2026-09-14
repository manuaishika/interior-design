"""Tests for "where are the rest of my designs".

Somebody generated dozens and found two, because a design was only stored if
they pressed Keep. Nobody thinks of a picture they just made as unsaved, and
pressing Draw again threw away everything on screen.
"""

import io

import numpy as np
import pytest
from PIL import Image

from app import store


def photo_like(size=(600, 400)):
    """A flat colour compresses better as PNG than JPEG, so it would test the
    opposite of what a render actually is."""
    rng = np.random.default_rng(0)
    w, h = size
    base = np.linspace(0, 255, w)[None, :, None] * np.ones((h, 1, 3))
    arr = np.clip(base + rng.normal(0, 18, (h, w, 3)), 0, 255).astype("uint8")
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


class TestTheyKeepThemselves:
    def test_every_design_saves_without_being_asked(self):
        html = open("static/showcase.html", encoding="utf-8").read()
        assert "saving.push(keepDesign(g.image_base64, keep, true));" in html

    def test_the_button_is_a_status_not_a_chore(self):
        html = open("static/showcase.html", encoding="utf-8").read()
        assert "'Saving…'" in html and "'Saved'" in html

    def test_a_signed_out_visitor_is_not_nagged_mid_render(self):
        """Auto-saving must not throw a sign-in dialog over somebody's
        results the moment they appear."""
        html = open("static/showcase.html", encoding="utf-8").read()
        assert "if (!quiet) { openDialog(); }" in html

    def test_the_shelf_refreshes_once_not_per_design(self):
        html = open("static/showcase.html", encoding="utf-8").read()
        assert "Promise.all(saving).then(loadSaved);" in html


class TestStorage:
    def test_a_render_is_compressed(self):
        """Rooms are photographs. A free Postgres is a gigabyte and a PNG
        render is a couple of megabytes, which is a ceiling nobody wants to
        find by hitting it."""
        original = photo_like()
        smaller = store.compress(original)
        assert len(smaller) < len(original) / 2

    def test_it_is_still_a_readable_image(self):
        out = Image.open(io.BytesIO(store.compress(photo_like())))
        out.load()
        assert out.size == (600, 400)

    def test_transparency_does_not_break_it(self):
        """JPEG has no alpha channel, so an RGBA render has to be flattened
        rather than throwing. Photo-like content, or the size guard correctly
        keeps the PNG and this tests nothing."""
        rgba = Image.open(io.BytesIO(photo_like((400, 400)))).convert("RGBA")
        buf = io.BytesIO()
        rgba.save(buf, format="PNG")

        out = Image.open(io.BytesIO(store.compress(buf.getvalue())))
        out.load()
        assert out.mode == "RGB"

    def test_nonsense_is_returned_unchanged(self):
        """A design that saves imperfectly beats one that fails to save."""
        assert store.compress(b"not an image") == b"not an image"

    def test_compression_never_makes_it_bigger(self):
        """A flat colour is smaller as PNG, and the guard must notice."""
        buf = io.BytesIO()
        Image.new("RGB", (400, 400), (200, 190, 180)).save(buf, format="PNG")
        flat = buf.getvalue()
        assert len(store.compress(flat)) <= len(flat)


class TestServingThem:
    @pytest.mark.parametrize("make,expected", [
        (lambda: photo_like((32, 32)), "image/png"),
        (lambda: store.compress(photo_like((320, 320))), "image/jpeg"),
    ])
    def test_the_type_is_sniffed_not_assumed(self, make, expected):
        """Designs are JPEG now and older rows are still PNG. A PNG served as
        a JPEG downloads with the wrong extension and some viewers refuse it."""
        from app.main import _media_type

        assert _media_type(make()) == expected

    def test_junk_does_not_claim_to_be_an_image(self):
        from app.main import _media_type

        assert _media_type(b"\x00\x01\x02") == "application/octet-stream"


class TestTheListIsLongEnough:
    def test_it_holds_more_than_an_afternoon(self):
        """Four designs a run, and sixty was a couple of hours' work."""
        import inspect

        source = inspect.signature(store.designs_for)
        assert source.parameters["limit"].default >= 200
