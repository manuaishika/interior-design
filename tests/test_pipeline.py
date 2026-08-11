import numpy as np
import pytest
from PIL import Image

from app.config import LOCK_POLICY, Settings, is_locked
from app.imaging import Mask
from app.labeling import RegionLabel, _clean
from app.generation import STYLES, build_prompt, _extract_image_ref, GenerationError
from app.pipeline import analyze_room, compose_inpaint_mask
from app.segmentation import SegmentationError, _extract_mask_refs


def region(h, w, y0, y1, x0, x1):
    arr = np.zeros((h, w), dtype=bool)
    arr[y0:y1, x0:x1] = True
    return arr


class TestLockPolicy:
    @pytest.mark.parametrize("category", ["door", "window", "walkway"])
    def test_spec_locked_categories(self, category):
        assert is_locked(category) is True

    @pytest.mark.parametrize("category", ["furniture", "floor"])
    def test_spec_unlocked_categories(self, category):
        assert is_locked(category) is False

    def test_unknown_category_defaults_locked(self):
        assert is_locked("chandelier-of-mystery") is True

    def test_other_is_locked(self):
        """A region the VLM could not identify is preserved, not regenerated."""
        assert LOCK_POLICY["other"] is True


class TestLabelCleaning:
    def test_furniture_keeps_specific_name(self):
        label = _clean({"category": "furniture", "name": "coffee table",
                        "confidence": 0.9, "notes": ""}, "m1")
        assert label.category == "furniture" and label.name == "coffee table"

    def test_non_furniture_name_is_canonicalised(self):
        label = _clean({"category": "door", "name": "a wooden door thing",
                        "confidence": 0.8, "notes": ""}, "m1")
        assert label.name == "door"

    def test_unknown_category_falls_back_to_other(self):
        assert _clean({"category": "ceiling_fan", "name": "fan"}, "m1").category == "other"

    def test_confidence_is_clamped(self):
        assert _clean({"category": "wall", "confidence": 5}, "m1").confidence == 1.0
        assert _clean({"category": "wall", "confidence": -2}, "m1").confidence == 0.0

    def test_garbage_confidence_becomes_zero(self):
        assert _clean({"category": "wall", "confidence": "high"}, "m1").confidence == 0.0

    def test_missing_name_falls_back_to_category(self):
        assert _clean({"category": "furniture"}, "m1").name == "furniture"

    def test_unknown_fallback_is_locked_category(self):
        assert is_locked(RegionLabel.unknown("m1", "boom").category) is True


class TestOutputNormalisers:
    def test_sam2_dict_output(self):
        assert _extract_mask_refs({"individual_masks": ["a", "b"]}) == ["a", "b"]

    def test_sam2_list_output(self):
        assert _extract_mask_refs(["a"]) == ["a"]

    def test_sam2_combined_only_is_an_error(self):
        with pytest.raises(SegmentationError, match="combined mask"):
            _extract_mask_refs({"combined_mask": "u"})

    def test_sam2_none_is_an_error(self):
        with pytest.raises(SegmentationError):
            _extract_mask_refs(None)

    def test_generation_list_output(self):
        assert _extract_image_ref(["first", "second"]) == "first"

    def test_generation_empty_list_is_an_error(self):
        with pytest.raises(GenerationError):
            _extract_image_ref([])


class TestPrompt:
    def test_known_style_expands(self):
        assert "pale oak" in build_prompt("scandinavian")

    def test_style_lookup_is_case_insensitive(self):
        assert build_prompt("Scandinavian") == build_prompt("scandinavian")

    def test_freeform_style_passes_through(self):
        assert "art deco with teal" in build_prompt("art deco with teal")

    def test_extra_prompt_appended(self):
        assert "reading nook" in build_prompt("japandi", "add a reading nook")

    def test_every_style_builds(self):
        assert all(build_prompt(s) for s in STYLES)


class FakeAnalysis:
    """Drives analyze_room with stubbed SAM2 + VLM calls."""

    def __init__(self, monkeypatch, masks, labels):
        async def fake_segment(image, settings):
            return masks

        async def fake_label(image, ms, boxes, settings):
            return [labels[m.mask_id] for m in ms]

        monkeypatch.setattr("app.pipeline.segment_room", fake_segment)
        monkeypatch.setattr("app.pipeline.label_masks", fake_label)


@pytest.fixture
def settings():
    return Settings(
        replicate_api_token="test", openai_api_key="test",
        min_mask_area_frac=0.0, max_mask_area_frac=1.0, locked_dilation_px=0,
    )


@pytest.fixture
def scene(monkeypatch):
    """A 100x100 room: a sofa, a door, and a stretch of floor."""
    masks = [
        Mask("mask_000", region(100, 100, 60, 90, 5, 45)),    # sofa
        Mask("mask_001", region(100, 100, 20, 80, 80, 95)),   # door
        Mask("mask_002", region(100, 100, 85, 100, 50, 100)), # floor
    ]
    labels = {
        "mask_000": RegionLabel("mask_000", "furniture", "sofa", 0.93),
        "mask_001": RegionLabel("mask_001", "door", "door", 0.88),
        "mask_002": RegionLabel("mask_002", "floor", "floor", 0.7),
    }
    return masks, labels


@pytest.mark.asyncio
class TestAnalyzeRoom:
    async def test_builds_the_required_json_shape(self, monkeypatch, settings, scene):
        masks, labels = scene
        FakeAnalysis(monkeypatch, masks, labels)
        analysis, _ = await analyze_room(Image.new("RGB", (100, 100)), settings)

        assert len(analysis.objects) == 3
        for obj in analysis.objects:
            payload = obj.model_dump()
            assert {"label", "mask_id", "bounding_box", "locked"} <= payload.keys()
            assert {"x", "y", "width", "height"} == payload["bounding_box"].keys()

    async def test_lock_flags_follow_the_policy(self, monkeypatch, settings, scene):
        masks, labels = scene
        FakeAnalysis(monkeypatch, masks, labels)
        analysis, _ = await analyze_room(Image.new("RGB", (100, 100)), settings)
        by_id = {o.mask_id: o for o in analysis.objects}

        assert by_id["mask_001"].locked is True    # door
        assert by_id["mask_000"].locked is False   # sofa
        assert by_id["mask_002"].locked is False   # floor

    async def test_labels_carry_specific_furniture_names(
        self, monkeypatch, settings, scene
    ):
        masks, labels = scene
        FakeAnalysis(monkeypatch, masks, labels)
        analysis, _ = await analyze_room(Image.new("RGB", (100, 100)), settings)
        assert {o.label for o in analysis.objects} == {"sofa", "door", "floor"}

    async def test_bounding_box_matches_the_mask(self, monkeypatch, settings, scene):
        masks, labels = scene
        FakeAnalysis(monkeypatch, masks, labels)
        analysis, _ = await analyze_room(Image.new("RGB", (100, 100)), settings)
        door = next(o for o in analysis.objects if o.mask_id == "mask_001")
        bb = door.bounding_box
        assert (bb.x, bb.y, bb.width, bb.height) == (80, 20, 15, 60)

    async def test_user_override_can_keep_furniture(self, monkeypatch, settings, scene):
        masks, labels = scene
        FakeAnalysis(monkeypatch, masks, labels)
        analysis, _ = await analyze_room(
            Image.new("RGB", (100, 100)), settings, keep_mask_ids={"mask_000"}
        )
        sofa = next(o for o in analysis.objects if o.mask_id == "mask_000")
        assert sofa.locked is True and sofa.lock_source == "user_override"

    async def test_user_override_cannot_be_contradictory(
        self, monkeypatch, settings, scene
    ):
        masks, labels = scene
        FakeAnalysis(monkeypatch, masks, labels)
        with pytest.raises(ValueError, match="both keep and replace"):
            await analyze_room(
                Image.new("RGB", (100, 100)), settings,
                keep_mask_ids={"mask_000"}, replace_mask_ids={"mask_000"},
            )

    async def test_objects_sorted_largest_first(self, monkeypatch, settings, scene):
        masks, labels = scene
        FakeAnalysis(monkeypatch, masks, labels)
        analysis, _ = await analyze_room(Image.new("RGB", (100, 100)), settings)
        areas = [o.area_px for o in analysis.objects]
        assert areas == sorted(areas, reverse=True)

    async def test_inpaint_mask_protects_only_locked_regions(
        self, monkeypatch, settings, scene
    ):
        masks, labels = scene
        FakeAnalysis(monkeypatch, masks, labels)
        image = Image.new("RGB", (100, 100))
        analysis, mask_map = await analyze_room(image, settings)
        arr = np.array(compose_inpaint_mask(analysis, mask_map, settings))

        assert arr[50, 85] == 0     # door -> preserved
        assert arr[70, 20] == 255   # sofa -> regenerated
        assert arr[90, 70] == 255   # floor -> regenerated
        assert arr[5, 5] == 255     # unsegmented -> regenerated

    async def test_walkway_is_locked_end_to_end(self, monkeypatch, settings):
        masks = [Mask("mask_000", region(100, 100, 40, 60, 0, 100))]
        labels = {"mask_000": RegionLabel("mask_000", "walkway", "walkway", 0.8)}
        FakeAnalysis(monkeypatch, masks, labels)
        image = Image.new("RGB", (100, 100))
        analysis, mask_map = await analyze_room(image, settings)
        arr = np.array(compose_inpaint_mask(analysis, mask_map, settings))

        assert analysis.objects[0].locked is True
        assert arr[50, 50] == 0

    async def test_failed_label_region_is_locked(self, monkeypatch, settings):
        masks = [Mask("mask_000", region(100, 100, 10, 40, 10, 40))]
        labels = {"mask_000": RegionLabel.unknown("mask_000", "api error")}
        FakeAnalysis(monkeypatch, masks, labels)
        image = Image.new("RGB", (100, 100))
        analysis, mask_map = await analyze_room(image, settings)
        arr = np.array(compose_inpaint_mask(analysis, mask_map, settings))

        assert analysis.objects[0].locked is True
        assert arr[25, 25] == 0

    async def test_counters_report_filtering(self, monkeypatch, settings, scene):
        masks, labels = scene
        FakeAnalysis(monkeypatch, masks, labels)
        analysis, _ = await analyze_room(Image.new("RGB", (100, 100)), settings)
        assert analysis.masks_returned == 3 and analysis.masks_labeled == 3
