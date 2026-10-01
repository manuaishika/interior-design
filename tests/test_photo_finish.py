"""Designs should look like photographs, not renders — and the client's own
words must never be the part that gets cut off."""

from app.generation import PHOTO_FINISH, PRESERVE, build_prompt


class TestThePhotographicFinish:
    def test_every_prompt_asks_for_a_real_photograph(self):
        for style in ("japandi", "none", "brief"):
            prompt = build_prompt(style, room="bedroom", depth="restyle")
            assert PHOTO_FINISH in prompt
        assert "not a 3D render" in PHOTO_FINISH
        assert "full-frame camera" in PHOTO_FINISH
        assert "late-afternoon window light" in PHOTO_FINISH

    def test_the_old_architectural_visualisation_wording_is_gone(self):
        assert "architectural photography" not in build_prompt("japandi")

    def test_the_finish_never_moves_the_camera(self):
        """The lens and eye-level are about the look; the viewpoint is the
        original photograph's, and both clauses say so."""
        assert "never changes the camera" in PHOTO_FINISH
        assert "same spot" in PRESERVE

    def test_lived_in_does_not_become_clutter(self):
        assert "never clutter" in PHOTO_FINISH

    def test_the_clients_words_stay_last(self):
        prompt = build_prompt("japandi", "add a reading corner", room="bedroom",
                              depth="renovate")
        assert prompt.index(PHOTO_FINISH) < prompt.index("add a reading corner")
        assert prompt.rstrip().endswith("add a reading corner")


class TestNothingIsCutOff:
    def test_a_long_prompt_still_ends_with_the_clients_words(self, monkeypatch):
        """It was clipped to 4,000 characters, and the client's words are the
        last thing in the prompt. A full redesign with a long room description
        and a long request reached about 4,700."""
        import asyncio
        import io

        from PIL import Image

        from app import openai_images
        from app.config import Settings

        long_prompt = build_prompt(
            "japandi", "the last thing the client wrote " + "x" * 400,
            contents="y" * 500, keep="z" * 200, room="nursery", depth="renovate")
        assert len(long_prompt) > 4000

        seen = {}

        class Images:
            async def edit(self, **call):
                seen.update(call)
                raise RuntimeError("stop here")

        class Client:
            images = Images()

        monkeypatch.setattr(openai_images, "_client", lambda s: Client())
        image = Image.new("RGB", (64, 64))
        try:
            asyncio.run(openai_images.redraw(
                image, None, long_prompt, Settings(openai_api_key="sk-x")))
        except Exception:
            pass
        assert seen["prompt"].endswith("x" * 400)


class TestNotAJungle:
    """Scandinavian and friends came back with plants on every surface."""

    def test_every_prompt_limits_plants(self):
        from app.generation import DECOR_RESTRAINT
        for style in ("scandinavian", "bohemian", "japandi", "none"):
            assert DECOR_RESTRAINT in build_prompt(style, room="living")
        assert "at most one or two plants" in DECOR_RESTRAINT
        assert "no cut flowers" in DECOR_RESTRAINT

    def test_the_client_can_still_ask_for_a_jungle(self):
        from app.generation import DECOR_RESTRAINT
        prompt = build_prompt("bohemian", "lots of plants please")
        assert prompt.index(DECOR_RESTRAINT) < prompt.index("lots of plants please")

    def test_bohemian_no_longer_asks_for_abundant_plants(self):
        from app.generation import STYLES
        assert "abundant houseplants" not in STYLES["bohemian"]
