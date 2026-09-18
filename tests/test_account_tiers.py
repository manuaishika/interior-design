"""Tests for TODO.md #4: the tier comes from the account, not the deployment.

config.TIERS and tier_of() already worked; Settings.tier was one value for
the whole deployment, so two people signed into the same studio always got
the same quality and the same number of designs regardless of which plan
either of them was meant to be on. store.User.tier closes that gap. This
pins the actual acceptance test TODO.md names — two accounts on different
tiers draw differently from the same photograph — plus the migration onto
an existing `users` table, and that Settings.tier survives only as the
fallback for a session with no account at all.

Setting a tier by hand in the database is what the product actually does
right now (no payments yet, see TODO.md #6) — `_set_tier` does that with a
plain synchronous sqlite3 connection against the same file, deliberately
bypassing the app's own async engine so this never has to reason about
which event loop a pooled aiosqlite connection was opened under.
"""

import io
import sqlite3

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


def set_tier(db_settings, email, tier):
    path = db_settings.database_url.split("///")[-1]
    conn = sqlite3.connect(path)
    conn.execute("UPDATE users SET tier = ? WHERE email = ?", (tier, email))
    conn.commit()
    conn.close()


def draw_stub(monkeypatch):
    """No real OpenAI calls — this file is about which quality and how many
    variants get asked for, not about what comes back."""
    seen = {"calls": []}

    async def find(image, s):
        return []

    async def redraw(image, mask, prompt, s, **kw):
        seen["calls"].append(kw.get("quality"))
        buf = io.BytesIO()
        Image.new("RGB", (8, 8)).save(buf, "PNG")
        return buf.getvalue()

    monkeypatch.setattr("app.openai_images.find_structure", find)
    monkeypatch.setattr("app.openai_images.redraw", redraw)
    return seen


class TestTheMigrationOntoAnExistingDatabase:
    """create_all only creates tables that do not exist yet — it never
    alters one that does. `users` already existed, with real accounts in
    it, by the time tier was added to the model."""

    @pytest.mark.asyncio
    async def test_an_existing_users_table_gets_the_new_column(self, tmp_path):
        from sqlalchemy import Column, DateTime, Integer, MetaData, String, Table

        url = f"sqlite+aiosqlite:///{tmp_path}/old.db"
        store.configure(url)
        try:
            async with store._engine.begin() as conn:
                def make_old_shape(sync_conn):
                    meta = MetaData()
                    Table(
                        "users", meta,
                        Column("id", Integer, primary_key=True),
                        Column("email", String(320), unique=True),
                        Column("name", String(120)),
                        Column("password_hash", String(255)),
                        Column("google_id", String(64)),
                        Column("created_at", DateTime),
                    )
                    meta.create_all(sync_conn)
                await conn.run_sync(make_old_shape)

            await store.create_tables()          # the migration under test

            async with store.session() as db:
                user = await store.sign_up(db, "a@b.test", "a-good-password")
                assert user.tier == "free"        # the column's own default
        finally:
            await store.dispose()


class TestSessionReportsTheAccountsTier:
    def test_a_fresh_account_is_on_the_free_tier(self, client):
        join(client, "anna@example.test")
        body = client.get("/api/session").json()
        assert body["account"]["tier"] == "free"
        assert body["tier"]["id"] == "free"
        assert body["tier"]["variants"] == 2

    def test_a_tier_set_by_hand_is_reported_back(self, client, db_settings):
        join(client, "anna@example.test")
        set_tier(db_settings, "anna@example.test", "studio")

        body = client.get("/api/session").json()
        assert body["account"]["tier"] == "studio"
        assert body["tier"]["id"] == "studio"
        assert body["tier"]["variants"] == 4
        assert body["tier"]["quality"] == "xhigh"

    def test_settings_tier_is_only_the_fallback_for_no_account(self, client):
        """A code-only session (no signup, no login) has nobody's tier to
        read — it falls back to the deployment default, not to a signed-in
        account that happens to share the process."""
        body = client.get("/api/session").json()
        assert body["account"] is None
        assert body["tier"]["id"] == "free"


class TestTwoAccountsTwoTiers:
    """The acceptance test TODO.md #4 actually names: two accounts on
    different tiers get different quality and a different number of
    designs from the same photograph."""

    def test_quality_differs_by_account(self, client, db_settings, monkeypatch):
        seen = draw_stub(monkeypatch)

        join(client, "anna@example.test")
        set_tier(db_settings, "anna@example.test", "free")
        r = client.post("/api/generate",
                        files={"photo": ("r.png", png(), "image/png")},
                        data={"style": "japandi", "variants": "1"})
        assert r.status_code == 200
        client.post("/api/logout")

        join(client, "ben@example.test")
        set_tier(db_settings, "ben@example.test", "studio")
        r = client.post("/api/generate",
                        files={"photo": ("r.png", png(), "image/png")},
                        data={"style": "japandi", "variants": "1"})
        assert r.status_code == 200

        assert seen["calls"] == ["medium", "xhigh"]

    def test_the_design_count_cap_differs_by_account(self, client, db_settings, monkeypatch):
        seen = draw_stub(monkeypatch)

        join(client, "anna@example.test")
        set_tier(db_settings, "anna@example.test", "free")
        client.post("/api/generate",
                    files={"photo": ("r.png", png(), "image/png")},
                    data={"style": "japandi", "variants": "4"})
        free_count = len(seen["calls"])
        client.post("/api/logout")

        seen["calls"] = []
        join(client, "ben@example.test")
        set_tier(db_settings, "ben@example.test", "studio")
        client.post("/api/generate",
                    files={"photo": ("r.png", png(), "image/png")},
                    data={"style": "japandi", "variants": "4"})
        studio_count = len(seen["calls"])

        assert free_count == 2      # config.TIERS["free"]["variants"]
        assert studio_count == 4    # config.TIERS["studio"]["variants"]


class TestEditRespectsTheAccountsTierToo:
    def test_applying_a_change_uses_the_accounts_quality(self, client, db_settings, monkeypatch):
        seen = draw_stub(monkeypatch)

        join(client, "anna@example.test")
        set_tier(db_settings, "anna@example.test", "home")
        r = client.post("/api/edit",
                        files={"design": ("d.png", png(), "image/png")},
                        data={"instruction": "make the wall deep olive"})
        assert r.status_code == 200
        assert seen["calls"] == ["high"]      # config.TIERS["home"]["quality"]
