"""End-to-end run of the full flow with only the model calls stubbed.

Everything between the upload bytes and the generator payload is real code:
image normalisation, mask filtering, bbox extraction, lock policy, and mask
composition.
"""

import base64
import io

import numpy as np
import pytest
from PIL import Image

from app.config import Settings
from app.imaging import Mask
from app.labeling import RegionLabel
from app.pipeline import run_pipeline


@pytest.fixture
def settings():
    return Settings(
        replicate_api_token="test",
        openai_api_key="test",
        min_mask_area_frac=0.001,
        max_mask_area_frac=0.99,
        locked_dilation_px=4,
        max_image_edge=256,
    )


def upload_bytes(size=(300, 200)):
    buf = io.BytesIO()
    Image.new("RGB", size, (210, 200, 190)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def stub_services(monkeypatch):
    """Stub SAM2 and the VLM; capture what the generator actually receives."""
    captured = {}

    labels = {
        "mask_000": RegionLabel("mask_000", "furniture", "sofa", 0.94),
        "mask_001": RegionLabel("mask_001", "window", "window", 0.91),
        "mask_002": RegionLabel("mask_002", "walkway", "walkway", 0.77),
        "mask_003": RegionLabel("mask_003", "wall", "wall", 0.85),
    }

    async def fake_read(image, settings):
        h, w = image.size[1], image.size[0]

        def band(y0, y1, x0, x1):
            arr = np.zeros((h, w), dtype=bool)
            arr[int(h * y0):int(h * y1), int(w * x0):int(w * x1)] = True
            return arr

        masks = [
            Mask("mask_000", band(0.55, 0.95, 0.02, 0.40)),  # sofa
            Mask("mask_001", band(0.10, 0.70, 0.80, 0.98)),  # window
            Mask("mask_002", band(0.45, 0.60, 0.00, 1.00)),  # walkway
            Mask("mask_003", band(0.00, 0.35, 0.05, 0.35)),  # wall
        ]
        return masks, [labels[m.mask_id] for m in masks]

    captured["seeds"] = []

    async def fake_generate(image, inpaint_mask, prompt, settings, seed=None):
        captured["image"] = image
        captured["mask"] = inpaint_mask
        captured["prompt"] = prompt
        captured["seeds"].append(seed)
        out = io.BytesIO()
        Image.new("RGB", image.size, (120, 130, 140)).save(out, format="PNG")
        return base64.b64encode(out.getvalue()).decode(), "https://example.test/o.png"

    monkeypatch.setattr("app.pipeline.read_room", fake_read)
    monkeypatch.setattr("app.pipeline.render", fake_generate)
    return captured


@pytest.mark.asyncio
class TestFullFlow:
    async def test_returns_both_image_and_json(self, settings, stub_services):
        analysis, generations = await run_pipeline(
            upload_bytes(), "japandi", settings, seed=7
        )
        assert len(analysis.objects) == 4
        assert len(generations) == settings.default_variants
        for generation in generations:
            assert base64.b64decode(generation.image_base64)[:4] == b"\x89PNG"
            assert base64.b64decode(generation.inpaint_mask_base64)[:4] == b"\x89PNG"
            assert "japandi" in generation.prompt.lower()

    async def test_mask_and_image_dimensions_agree(self, settings, stub_services):
        analysis, _ = await run_pipeline(upload_bytes(), "japandi", settings)
        assert stub_services["image"].size == stub_services["mask"].size
        assert stub_services["image"].size == (
            analysis.image_width,
            analysis.image_height,
        )

    async def test_image_is_normalised_before_analysis(self, settings, stub_services):
        analysis, _ = await run_pipeline(upload_bytes((1000, 700)), "japandi", settings)
        w, h = analysis.image_width, analysis.image_height
        assert max(w, h) <= settings.max_image_edge
        assert w % 8 == 0 and h % 8 == 0  # diffusion-safe dimensions

    async def test_locked_regions_are_black_in_the_generator_mask(
        self, settings, stub_services
    ):
        analysis, _ = await run_pipeline(upload_bytes(), "japandi", settings)
        arr = np.array(stub_services["mask"])
        h, w = arr.shape

        window = next(o for o in analysis.objects if o.mask_id == "mask_001")
        walkway = next(o for o in analysis.objects if o.mask_id == "mask_002")
        for obj in (window, walkway):
            bb = obj.bounding_box
            cy, cx = bb.y + bb.height // 2, bb.x + bb.width // 2
            assert arr[cy, cx] == 0, f"{obj.label} should be preserved"

    async def test_unlocked_regions_are_white_in_the_generator_mask(
        self, settings, stub_services
    ):
        analysis, _ = await run_pipeline(upload_bytes(), "japandi", settings)
        arr = np.array(stub_services["mask"])

        sofa = next(o for o in analysis.objects if o.mask_id == "mask_000")
        bb = sofa.bounding_box
        assert arr[bb.y + bb.height // 2, bb.x + bb.width // 2] == 255

    async def test_mask_leaves_room_to_edit(self, settings, stub_services):
        """Sanity check that we did not accidentally lock the whole frame."""
        await run_pipeline(upload_bytes(), "japandi", settings)
        editable = (np.array(stub_services["mask"]) == 255).mean()
        assert 0.2 < editable < 0.95

    async def test_seed_and_extra_prompt_reach_the_generator(
        self, settings, stub_services
    ):
        await run_pipeline(
            upload_bytes(), "industrial", settings,
            extra_prompt="add a tall bookshelf", seed=1234, variants=2,
        )
        # A caller-supplied seed anchors the run; each option steps off it.
        assert sorted(stub_services["seeds"]) == [1234, 1235]
        assert "tall bookshelf" in stub_services["prompt"]

    async def test_each_option_gets_a_distinct_seed(self, settings, stub_services):
        _, generations = await run_pipeline(
            upload_bytes(), "japandi", settings, variants=3
        )
        seeds = [g.seed for g in generations]
        assert len(set(seeds)) == 3
        assert [g.variant_index for g in generations] == [0, 1, 2]

    async def test_variants_are_capped(self, settings, stub_services):
        _, generations = await run_pipeline(
            upload_bytes(), "japandi", settings, variants=99
        )
        assert len(generations) == settings.max_variants

    async def test_every_option_shares_one_mask(self, settings, stub_services):
        """Options differ in content, never in which regions may change."""
        _, generations = await run_pipeline(
            upload_bytes(), "japandi", settings, variants=3
        )
        assert len({g.inpaint_mask_base64 for g in generations}) == 1

    async def test_one_failed_option_does_not_sink_the_batch(
        self, settings, stub_services, monkeypatch
    ):
        calls = {"n": 0}
        original = stub_services

        async def flaky(image, mask, prompt, settings, seed=None):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("generator hiccup")
            out = io.BytesIO()
            Image.new("RGB", image.size, (10, 20, 30)).save(out, format="PNG")
            return base64.b64encode(out.getvalue()).decode(), None

        monkeypatch.setattr("app.pipeline.render", flaky)
        _, generations = await run_pipeline(
            upload_bytes(), "japandi", settings, variants=3
        )
        assert len(generations) == 2

    async def test_user_keep_override_protects_the_sofa(self, settings, stub_services):
        analysis, _ = await run_pipeline(
            upload_bytes(), "japandi", settings, keep_mask_ids={"mask_000"}
        )
        arr = np.array(stub_services["mask"])
        sofa = next(o for o in analysis.objects if o.mask_id == "mask_000")
        bb = sofa.bounding_box

        assert sofa.locked is True
        assert arr[bb.y + bb.height // 2, bb.x + bb.width // 2] == 0
