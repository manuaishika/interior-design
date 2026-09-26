"""A finished design hands people to the business on WhatsApp, and the cost
estimate is priced where the business actually works.

The page never says "Dubai" — the currency and the market only steer the
figures, and the WhatsApp button only appears once a number is configured.
"""

import pathlib

from app.config import Settings
from app.main import _contact
from app.reading import SURVEY


def page(path="static/showcase.html"):
    return pathlib.Path(path).read_text(encoding="utf-8")


class TestContactCard:
    def test_no_number_means_no_button(self):
        assert _contact(Settings(whatsapp_number="")) is None

    def test_the_number_is_reduced_to_digits_for_wa_me(self):
        card = _contact(Settings(whatsapp_number="+971 50 123 4567"))
        assert card["whatsapp"] == "971501234567"

    def test_the_business_is_named(self):
        card = _contact(Settings(whatsapp_number="971501234567",
                                 business_name="Eleganza Interiors"))
        assert card["business"] == "Eleganza Interiors"


class TestTheMarket:
    def test_defaults_are_aed_and_dubai(self):
        s = Settings()
        assert s.currency == "AED"
        assert s.market == "Dubai, UAE"

    def test_the_example_json_does_not_contradict_the_currency(self):
        """The example used to say "INR" whatever the instruction said, and a
        model copies an example before it follows a sentence."""
        prompt = SURVEY.format(room="bedroom", currency="AED")
        assert '"currency": "AED"' in prompt
        assert "INR" not in prompt


class TestThePage:
    def test_the_button_links_to_wa_me_with_a_prefilled_message(self):
        source = page()
        assert "'https://wa.me/' + contact.whatsapp" in source
        assert "encodeURIComponent(text)" in source

    def test_it_appears_only_after_a_design_lands(self):
        source = page()
        draw = source[source.index("async function draw()"):]
        draw = draw[:draw.index("\n  function ")]
        assert draw.index("progress.done === 0") < draw.index("showContact()")

    def test_the_page_never_names_the_city(self):
        assert "Dubai" not in page()

    def test_docs_copy_matches(self):
        assert 'id="waGo"' in page("docs/index.html")
