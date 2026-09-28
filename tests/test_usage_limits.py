"""Tests for the daily design cap.

auth.guard only answers "is this deployment open at all" — with no access
code set (the live deployment's own configuration), that is anyone, no
account required, which is the point of a free demo. Nothing capped how
many designs one visitor could draw in a day, so "free demo" and "an
unmetered OpenAI key" were the same sentence.

A stranger with no account is now capped at settings.anon_daily_designs
(2), identified by IP, the only thing an anonymous request carries. A
signed-in account is capped at its own tier's TIERS[...]["daily"] — 5 for
Free, matching the number already on the pricing page; uncapped for the
paid tiers, whose cost is already bounded by quality and variants per run.
The count is credited only after a design actually comes back, so a
request that fails before drawing anything does not cost the person any
of their quota.
"""

import io

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from PIL import Image

import app.main as main
from app import store
from app.config import Settings
from app.main import app


def settings(**kw):
    kw.setdefault("openai_api_key", "sk-test")
    kw.setdefault("session_secret", "s3cret")
    kw.setdefault("anon_daily_designs", 2)
    return Settings(**kw)


@pytest.fixture
def db_settings(tmp_path):
    return settings(database_url=f"sqlite+aiosqlite:///{tmp_path}/t.db")


@pytest_asyncio.fixture
async def db(db_settings):
    store.configure(db_settings.database_url)
    await store.create_tables()
    yield
    await store.dispose()


@pytest.fixture
def client(monkeypatch, db_settings, db):
    monkeypatch.setattr(main, "get_settings", lambda: db_settings)
    with TestClient(app) as c:
        yield c


def png():
    buf = io.BytesIO()
    Image.new("RGB", (64, 64)).save(buf, "PNG")
    return buf.getvalue()


def join(client, email, password="a-good-password"):
    return client.post("/api/signup", data={"email": email, "password": password})


def draw_stub(monkeypatch, fail_times=0):
    """No real OpenAI calls. fail_times lets a test see that a failed
    attempt does not spend the person's quota."""
    calls = {"n": 0}

    async def find(image, s):
        return []

    async def redraw(image, mask, prompt, s, **kw):
        calls["n"] += 1
        if calls["n"] <= fail_times:
            raise RuntimeError("upstream is down")
        buf = io.BytesIO()
        Image.new("RGB", (8, 8)).save(buf, "PNG")
        return buf.getvalue()

    monkeypatch.setattr("app.openai_images.find_structure", find)
    monkeypatch.setattr("app.openai_images.redraw", redraw)
    return calls


def draw(client, variants="1", ip=None):
    headers = {"x-forwarded-for": ip} if ip else {}
    return client.post(
        "/api/generate",
        files={"photo": ("r.png", png(), "image/png")},
        data={"style": "japandi", "variants": variants},
        headers=headers,
    )


class TestAnonymousVisitors:
    def test_the_configured_number_go_through(self, client, monkeypatch):
        draw_stub(monkeypatch)
        for _ in range(2):
            assert draw(client, ip="1.1.1.1").status_code == 200

    def test_the_next_one_is_refused_with_a_clear_reason(self, client, monkeypatch):
        draw_stub(monkeypatch)
        for _ in range(2):
            draw(client, ip="1.1.1.1")
        r = draw(client, ip="1.1.1.1")
        assert r.status_code == 429
        body = r.json()["detail"]
        assert "2" in body
        assert "account" in body.lower()

    def test_a_different_visitor_has_their_own_quota(self, client, monkeypatch):
        """Keyed by IP, not by the process — someone else hitting their
        cap must not lock everyone else out too."""
        draw_stub(monkeypatch)
        for _ in range(2):
            draw(client, ip="1.1.1.1")
        assert draw(client, ip="1.1.1.1").status_code == 429
        assert draw(client, ip="2.2.2.2").status_code == 200

    def test_a_failed_attempt_does_not_spend_the_quota(self, client, monkeypatch):
        draw_stub(monkeypatch, fail_times=1)
        # First call: the only redraw attempt raises, so nothing comes back.
        first = draw(client, ip="3.3.3.3")
        assert first.status_code == 502
        # Both of the real two are still available.
        assert draw(client, ip="3.3.3.3").status_code == 200
        assert draw(client, ip="3.3.3.3").status_code == 200
        assert draw(client, ip="3.3.3.3").status_code == 429

    def test_forwarded_for_takes_the_first_address_in_the_chain(self, client, monkeypatch):
        """A request that has passed through more than one proxy lists each
        hop; the first is the visitor, not whichever proxy appended itself
        last."""
        draw_stub(monkeypatch)
        headers = {"x-forwarded-for": "4.4.4.4, 10.0.0.1"}
        for _ in range(2):
            client.post("/api/generate", files={"photo": ("r.png", png(), "image/png")},
                        data={"style": "japandi", "variants": "1"}, headers=headers)
        r = client.post("/api/generate", files={"photo": ("r.png", png(), "image/png")},
                        data={"style": "japandi", "variants": "1"}, headers=headers)
        assert r.status_code == 429
        assert draw(client, ip="4.4.4.4").status_code == 429   # same visitor


class TestSignedInFreeTier:
    def test_five_a_day_matches_the_pricing_page(self, client, monkeypatch):
        draw_stub(monkeypatch)
        join(client, "anna@example.test")
        for _ in range(5):
            assert draw(client).status_code == 200
        r = draw(client)
        assert r.status_code == 429
        assert "5" in r.json()["detail"]

    def test_two_accounts_do_not_share_a_quota(self, client, monkeypatch):
        draw_stub(monkeypatch)
        join(client, "anna@example.test")
        for _ in range(5):
            draw(client)
        assert draw(client).status_code == 429
        client.post("/api/logout")

        join(client, "ben@example.test")
        assert draw(client).status_code == 200

    def test_signing_in_is_its_own_reason_over_staying_anonymous(self, client, monkeypatch):
        """The two limits are 2 and 5 — an anonymous visitor who signs in
        gets a real increase, not a relabelled version of the same cap."""
        from app.config import TIERS
        assert TIERS["free"]["daily"] == 5
        assert settings().anon_daily_designs == 2


class TestPaidTiersAreNotCapped:
    def test_a_paid_tier_keeps_drawing_past_the_free_number(self, client, db_settings, monkeypatch):
        import sqlite3
        draw_stub(monkeypatch)
        join(client, "priya@example.test")
        path = db_settings.database_url.split("///")[-1]
        conn = sqlite3.connect(path)
        conn.execute("UPDATE users SET tier = ? WHERE email = ?", ("studio", "priya@example.test"))
        conn.commit()
        conn.close()

        for _ in range(7):     # more than Free's 5
            assert draw(client).status_code == 200


class TestFailingOpenRatherThanClosed:
    def test_no_store_configured_does_not_block_generation(self, monkeypatch):
        """The meter itself being unreachable must not turn a soft limit
        into a hard outage. The app's own startup event would otherwise
        configure a store the moment TestClient enters, same as it does in
        production — stubbed out here so this test actually exercises the
        unconfigured path instead of a real one."""
        async def no_op(*a, **kw):
            return None
        monkeypatch.setattr(store, "configure", lambda *a, **kw: None)
        monkeypatch.setattr(store, "create_tables", no_op)
        store._sessions = None
        draw_stub(monkeypatch)
        s = settings()
        monkeypatch.setattr(main, "get_settings", lambda: s)
        with TestClient(app) as c:
            for _ in range(4):   # more than anon_daily_designs=2
                assert draw(c, ip="5.5.5.5").status_code == 200


class TestEditAlsoSpendsTheQuota:
    """/api/edit costs the same real money as /api/generate — it was not
    gated at all before this, which meant hitting the daily cap on
    /api/generate was never actually a cap: the same person could keep
    editing instead."""

    def _png(self):
        buf = io.BytesIO()
        Image.new("RGB", (32, 32)).save(buf, "PNG")
        return buf.getvalue()

    def test_an_edit_counts_against_the_same_anonymous_quota(self, client, monkeypatch):
        draw_stub(monkeypatch)
        for _ in range(2):
            assert draw(client, ip="6.6.6.6").status_code == 200
        r = client.post(
            "/api/edit",
            files={"design": ("d.png", self._png(), "image/png")},
            data={"instruction": "make it warmer"},
            headers={"x-forwarded-for": "6.6.6.6"},
        )
        assert r.status_code == 429

    def test_an_edit_can_itself_spend_the_last_of_the_quota(self, client, monkeypatch):
        draw_stub(monkeypatch)
        r = client.post(
            "/api/edit",
            files={"design": ("d.png", self._png(), "image/png")},
            data={"instruction": "make it warmer"},
            headers={"x-forwarded-for": "7.7.7.7"},
        )
        assert r.status_code == 200
        assert draw(client, ip="7.7.7.7").status_code == 200        # 2nd
        assert draw(client, ip="7.7.7.7").status_code == 429        # 3rd, over
