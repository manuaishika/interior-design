"""Tests for /api/feedback: collecting real opinions on real designs, tied
to the exact prompt that produced them.

"Why isn't the agent getting better" has no honest answer from inside one
session — one person's read of one result is a data point, not a verdict.
This is the mechanism for gathering more than one: a short "was this
close, and if not, why not" under each design, saved with the prompt that
made it, readable later at /api/feedback?key=... rather than reconstructed
from memory.
"""

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient

import app.main as main
from app import store
from app.config import Settings
from app.main import app


def settings(**kw):
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


def send(client, good=True, **kw):
    data = {"good": str(good).lower(), "note": "", "room": "bedroom",
            "style": "brief", "depth": "renovate", "prompt": "Redesign..."}
    data.update(kw)
    return client.post("/api/feedback", data=data)


class TestSubmitting:
    def test_a_good_verdict_is_saved(self, client):
        r = send(client, good=True)
        assert r.status_code == 200
        assert r.json() == {"saved": True}

    def test_a_not_quite_verdict_with_a_note_is_saved(self, client):
        r = send(client, good=False, note="wanted more storage, got none")
        assert r.status_code == 200

    def test_no_sign_in_required(self, client):
        """Feedback from a visitor who never made an account is still real
        feedback — same door as /api/generate, not a stricter one."""
        r = send(client, good=False)
        assert r.status_code == 200

    def test_the_prompt_that_actually_produced_it_is_what_gets_saved(self, client, monkeypatch, db_settings):
        s = settings(database_url=db_settings.database_url, feedback_key="k")
        monkeypatch.setattr(main, "get_settings", lambda: s)

        send(client, good=False, note="ceiling turned into a pentagon",
            prompt="Redesign this room as the interior described...")

        rows = client.get("/api/feedback", params={"key": "k"}).json()["feedback"]
        assert rows[0]["prompt"] == "Redesign this room as the interior described..."
        assert rows[0]["note"] == "ceiling turned into a pentagon"


class TestReviewIsLockedByASeparateKey:
    def test_with_no_key_configured_the_endpoint_always_404s(self, client):
        r = client.get("/api/feedback", params={"key": "anything"})
        assert r.status_code == 404

    def test_the_wrong_key_404s_too(self, client, monkeypatch, db_settings):
        s = settings(database_url=db_settings.database_url, feedback_key="right-one")
        monkeypatch.setattr(main, "get_settings", lambda: s)
        r = client.get("/api/feedback", params={"key": "wrong-one"})
        assert r.status_code == 404

    def test_the_right_key_reads_it_back_newest_first(self, client, monkeypatch, db_settings):
        s = settings(database_url=db_settings.database_url, feedback_key="right-one")
        monkeypatch.setattr(main, "get_settings", lambda: s)

        send(client, good=True, room="bedroom")
        send(client, good=False, room="kitchen", note="too plain")

        r = client.get("/api/feedback", params={"key": "right-one"})
        assert r.status_code == 200
        rows = r.json()["feedback"]
        assert len(rows) == 2
        assert rows[0]["room"] == "kitchen"        # most recent first
        assert rows[0]["good"] is False
        assert rows[0]["note"] == "too plain"
        assert rows[1]["room"] == "bedroom"

    def test_a_deployment_that_never_set_the_key_cannot_be_tricked_by_an_empty_one(self, client):
        """settings.feedback_key defaults to "" — an empty ?key= must not
        match an empty configured key, or every unconfigured deployment
        would be readable by anyone who sent nothing."""
        r = client.get("/api/feedback", params={"key": ""})
        assert r.status_code == 404
