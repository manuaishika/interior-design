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


class TestTheBuildMarker:
    """The one thing that ends "did it actually deploy" as a guessing game:
    the footer reads back whichever commit Render says it built, straight
    from /api/health, no separate lookup."""

    def test_the_footer_has_somewhere_to_put_it(self):
        assert 'id="build"' in page()

    def test_the_health_check_writes_it_there(self):
        source = page()
        start = source.index("function check()")
        end = source.index("\n  }", start)
        body = source[start:end]
        assert "$('build').textContent" in body
        assert "h.build" in body


class TestLooksOnATouchScreen:
    """A phone has no hover: the look popup opened on a tap and died 180ms
    later, so "Mid-century" never said what it meant. On a device that
    cannot hover, the look is shown in place under the pills."""

    def test_touch_is_detected_by_hover_capability_not_width(self):
        assert "matchMedia('(hover: none)')" in page()

    def test_a_tap_fills_the_card_instead_of_the_popup(self):
        source = page()
        pick = source[source.index("function pickLook("):]
        pick = pick[:pick.index("\n  }")]
        assert "if (TOUCH) { showLookCard(x); } else { peek(x); }" in pick
        assert 'id="lookCard"' in source

    def test_the_hint_says_tap_on_a_phone(self):
        assert "Tap one to see it." in page()


class TestThePricingPageSellsWhatExists:
    """It used to sell Sketch / Studio / Practice — 100 designs a day,
    shareable boards, a priority queue — none of which existed."""

    def test_the_plans_are_the_ones_config_runs_on(self):
        from app.config import TIERS
        pricing = page()[page().index('id="v-pricing"'):page().index("</main>", page().index('id="v-pricing"'))]
        for tier in TIERS.values():
            assert f"<h3>{tier['label']}</h3>" in pricing
        for gone in ("Sketch", "Practice", "100 designs a day", "Unlimited designs", "Priority queue"):
            assert f">{gone}<" not in pricing and f"<li>{gone}</li>" not in pricing

    def test_paid_plans_do_not_pretend_to_be_buyable(self):
        pricing = page()[page().index('id="v-pricing"'):]
        assert pricing.count("disabled>Coming soon</button>") == 3

    def test_the_draft_note_is_not_shown_to_customers(self):
        assert "Set your own before this page goes in front of a customer" not in page()
