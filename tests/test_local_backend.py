"""Tests for the keyless backend's pure logic.

The models themselves need weights this environment cannot reach, but the two
pieces most likely to be wrong are pure functions: mapping ADE20K's class names
onto our categories, and inferring circulation space that ADE20K has no class
for. Both are tested here.
"""

import numpy as np
import pytest

from app.config import CATEGORIES, is_locked
from app.local_models import ade_category, ade_label, derive_walkway


class TestCategoryMapping:
    @pytest.mark.parametrize("name,expected", [
        ("door", "door"),
        ("screen door, screen", "door"),
        ("windowpane, window", "window"),
        ("wall", "wall"),
        ("floor, flooring", "floor"),
        ("rug, carpet, carpeting", "floor"),
        ("ceiling", "other"),
        ("bed", "furniture"),
        ("sofa, couch, lounge", "furniture"),
        ("chest of drawers, chest, bureau, dresser", "furniture"),
        ("coffee table, cocktail table", "furniture"),
        ("wardrobe, closet, press", "furniture"),
        ("bag", "clutter"),
        ("clothes", "clutter"),
        ("book", "clutter"),
        ("bottle", "clutter"),
    ])
    def test_known_classes(self, name, expected):
        assert ade_category(name) == expected

    def test_everything_maps_into_the_taxonomy(self):
        for name in ("door", "windowpane", "wall", "floor", "bed", "bag", "zebra", ""):
            assert ade_category(name) in CATEGORIES

    def test_unknown_class_is_other(self):
        assert ade_category("escalator, moving staircase") == "other"

    def test_empty_name_is_other(self):
        assert ade_category("") == "other"

    def test_structure_wins_over_furnishing(self):
        """A door must never fall through into furniture — it is load-bearing
        for the whole lock policy."""
        assert ade_category("door") == "door"
        assert is_locked(ade_category("door")) is True

    def test_ceiling_does_not_become_a_wall(self):
        assert ade_category("ceiling") != "wall"

    def test_structural_classes_end_up_locked(self):
        for name in ("door", "windowpane, window"):
            assert is_locked(ade_category(name)) is True

    def test_furnishings_and_mess_end_up_open(self):
        for name in ("sofa, couch", "bed", "bag", "clothes"):
            assert is_locked(ade_category(name)) is False


class TestLabels:
    def test_furniture_keeps_its_first_synonym(self):
        assert ade_label("sofa, couch, lounge", "furniture") == "sofa"

    def test_clutter_keeps_its_first_synonym(self):
        assert ade_label("bag, handbag", "clutter") == "bag"

    def test_structure_is_canonicalised(self):
        assert ade_label("windowpane, window", "window") == "window"
        assert ade_label("door", "door") == "door"


class TestWalkway:
    def floor(self, h=100, w=100, top=60):
        arr = np.zeros((h, w), dtype=bool)
        arr[top:, :] = True
        return arr

    def test_floor_minus_furniture(self):
        floor = self.floor()
        sofa = np.zeros((100, 100), dtype=bool)
        sofa[60:100, 0:40] = True

        walkway = derive_walkway(floor, [sofa])
        assert walkway is not None
        assert walkway[80, 70]          # open floor is circulation
        assert not walkway[80, 20]      # under the sofa is not

    def test_no_floor_gives_nothing(self):
        assert derive_walkway(np.zeros((50, 50), dtype=bool), []) is None

    def test_fully_covered_floor_gives_nothing(self):
        floor = self.floor()
        assert derive_walkway(floor, [floor]) is None

    def test_slivers_are_rejected(self):
        """A few loose pixels between furniture legs is not a route."""
        floor = self.floor()
        almost_all = floor.copy()
        almost_all[65:, :] = True       # leaves a 5px band, well under 2%
        assert derive_walkway(floor, [almost_all]) is None

    def test_derived_walkway_is_locked(self):
        floor = self.floor()
        sofa = np.zeros((100, 100), dtype=bool)
        sofa[60:100, 0:30] = True
        assert derive_walkway(floor, [sofa]) is not None
        assert is_locked("walkway") is True

    def test_input_floor_is_not_mutated(self):
        floor = self.floor()
        before = floor.copy()
        sofa = np.zeros((100, 100), dtype=bool)
        sofa[60:100, 0:30] = True
        derive_walkway(floor, [sofa])
        assert np.array_equal(floor, before)


class TestBackendSwitch:
    def test_default_backend_is_hosted(self):
        """The field is now blank by default and resolved from the keys that
        exist, so the guarantee worth pinning is the behaviour: configure
        nothing and you are still on the hosted path."""
        from app.config import Settings, resolve_backend

        assert resolve_backend(Settings()) == "hosted"

    def test_a_free_key_alone_switches_the_engine(self):
        """The commonest way to deploy this wrong is to paste a key and forget
        the flag that turns its engine on, so the key is enough by itself."""
        from app.config import Settings, resolve_backend

        assert resolve_backend(Settings(google_api_key="k")) == "free"
        # An explicit choice still wins over what the keys imply.
        assert resolve_backend(
            Settings(google_api_key="k", backend="hosted")) == "hosted"
        # And a paid pair is preferred, because its locks are real.
        assert resolve_backend(Settings(
            google_api_key="k", openai_api_key="o",
            replicate_api_token="r")) == "hosted"

    def test_local_backend_is_recognised(self):
        from app.config import Settings
        from app.pipeline import _is_local

        assert _is_local(Settings(backend="local")) is True
        assert _is_local(Settings(backend="LOCAL")) is True
        assert _is_local(Settings(backend="hosted")) is False

    def test_local_backend_needs_no_credentials(self):
        """The whole point: a keyless run must not require the paid accounts."""
        from app.config import Settings

        s = Settings(backend="local")
        assert not s.replicate_api_token
        assert not s.openai_api_key
