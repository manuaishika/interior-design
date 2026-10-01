"""Tests for the multi-angle redesign fix.

Uploading two or three photos for a full redesign used to redraw only the
first one — the rest were sent as `extras` and used purely as reference
context, so a person who walked the room and shot three corners got back
three near-identical takes on the one corner they photographed first. The
page said as much: "Designs come back from the first photo."

Now every uploaded photo beyond the first isn't just context: it is its own
request, with the *other* shots as its references, so a three-photo upload
comes back as three photos of the *same* design, each drawn from where that
picture was actually taken. A single photo still behaves exactly as before
— the "how many designs" stepper still produces that many independent
takes on the one view.

There is no JS test runner in this repo (see test_progressive_draw.py), so
this pins the source facts, the same way that file does.
"""

import pathlib


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


class TestEveryAngleIsItsOwnRequest:
    def test_multi_photo_draws_each_shot_not_just_the_first(self):
        body = function_body(page(), "async function draw()")
        assert "var primary = shots[i];" in body
        assert "j !== i" in body  # every other shot becomes that one's refs

    def test_the_offset_holds_still_across_angles(self):
        """variant_offset picks a different wording nudge per index in the
        single-photo path — applied across angles it would make each corner
        a different design instead of the same one, so the multi-angle call
        pins it at 0."""
        body = function_body(page(), "async function draw()")
        assert "drawOne(i, saving, progress, primary, refs, 0," in body

    def test_single_photo_path_is_unchanged_in_effect(self):
        """One photo still means `count` independent takes on that one view,
        offset by index, exactly as before the fix."""
        body = function_body(page(), "async function draw()")
        assert "drawOne(i, saving, progress, currentPhoto()," in body
        assert "shots.slice(1), i)" in body


class TestTheDesignCountIsThePersonsChoice:
    """One to three designs, whatever the number of photographs. Two photos
    used to force two designs; now someone can ask for one."""

    def test_three_is_the_ceiling_whatever_the_photo_count(self):
        source = page()
        assert "var DESIGN_CAP = 3;" in source
        body = function_body(source, "function designMax(")
        assert "Math.min(DESIGN_CAP," in body
        assert "tier.variants" in body

    def test_it_never_exceeds_the_photos_that_could_be_drawn(self):
        body = function_body(page(), "function designMax(")
        assert "shots.length > 1 ? Math.min(m, shots.length) : m" in body

    def test_the_stepper_stays_usable_with_several_photos(self):
        body = function_body(page(), "function updateCountUI(")
        assert "$('less').disabled = n <= 1;" in body
        assert "$('more').disabled = n >= designMax();" in body
        assert "multi" not in body

    def test_each_design_is_drawn_from_one_photo_with_the_rest_as_references(self):
        body = function_body(page(), "async function draw()")
        assert "var primary = shots[i];" in body
        assert "var n = optionCount()" in body.replace("multi = shots.length > 1, n", "var n")


class TestCompareUsesTheRightBeforePhoto:
    """Compare used to always reopen shot 0 — correct for a single photo,
    wrong once each figure was drawn from a different one."""

    def test_each_design_remembers_which_photo_it_came_from(self):
        body = function_body(page(), "function appendDesign(")
        assert "figure.beforePhoto = beforeBlob || currentPhoto();" in body

    def test_compare_reads_that_photo_back(self):
        body = function_body(page(), "function openCompare(")
        assert "figure.beforePhoto" in body
        assert "currentPhoto()" in body  # still the fallback


class TestEveryShotIsRedrawnNoneIsJustAReference:
    """A photo tagged "Reference" was never asked for as a feature — it was
    a plan cap wearing a label that made it sound deliberate. Every shot in
    the strip is redrawn now, so every tag says the one true thing."""

    def test_the_old_first_photo_only_claim_is_gone(self):
        assert "Designs come back from the first photo" not in page()

    def test_photos_not_drawn_are_labelled_and_explained(self):
        """Asking for fewer designs than photos leaves some as references.
        That is said on the photo and in the line under it, not left silent."""
        source = page()
        body = function_body(source, "function updateCountUI(")
        assert "tag.textContent = i < n ? 'Redrawn' : 'Reference';" in body
        assert "The others tell" in body

    def test_the_ceiling_follows_the_mode_and_says_so_upfront(self):
        """One photo is all a light restyle uses; a full redesign stops at
        four — a room has that many walls. Enforced as photos are added,
        with a disabled button and a straight sentence, never by quietly
        leaving an accepted photo out of the redraw."""
        assert "function maxShots() { return depth.id === 'renovate' ? 4 : 1; }" in page()
        body = function_body(page(), "function drawShots(")
        assert "$('plus').disabled = atMax;" in body
        assert "Up to ' + maxShots() + ' angles for a full redesign" in body

    def test_a_light_restyle_swaps_its_photo_rather_than_refusing(self):
        body = function_body(page(), "function addPhotos(")
        assert "files.slice(0, 1)" in body       # only the first is kept...
        assert "shots = ready;" in body          # ...and it replaces, not appends

    def test_photos_are_shrunk_to_what_the_server_uses_before_upload(self):
        """A phone photo is 3-12 MB; the server works at 1536px anyway, and
        every design request re-sent every angle at full size — the
        sluggishness on a phone, and a request that fails on weak signal."""
        source = page()
        assert "var PHOTO_EDGE = 1536;" in source
        assert "keep.map(shrinkPhoto)" in function_body(source, "function addPhotos(")
        assert "imageOrientation: 'from-image'" in source   # portraits stay upright


class TestDocsCopyMatches:
    def test_the_same_mechanism_exists_there(self):
        source = page("docs/index.html")
        assert "var primary = shots[i];" in source
        assert "figure.beforePhoto" in source
        assert "Designs come back from the first photo" not in source
