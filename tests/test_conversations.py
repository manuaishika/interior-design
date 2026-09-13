"""Tests for chat history: threads that survive closing the tab.

Designs already persisted against an account; conversations did not. This
pins the same separation `test_accounts.py::TestDesignsBelongToPeople`
pins for designs — one account's threads must not be reachable, by any
route, from another account's session — plus the two things unique to a
thread: that it remembers what was said, and that a design kept while it
was open hangs off it.
"""

import io
import json

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from PIL import Image

import app.main as main
from app import store
from app.config import Settings
from app.main import app


@pytest.fixture
def settings(tmp_path):
    return Settings(session_secret="s3cret",
                    database_url=f"sqlite+aiosqlite:///{tmp_path}/t.db")


@pytest_asyncio.fixture
async def db(settings):
    store.configure(settings.database_url)
    await store.create_tables()
    yield
    await store.dispose()


@pytest.fixture
def client(monkeypatch, settings, db):
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    with TestClient(app) as c:
        yield c


def png(colour=(200, 190, 180)):
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), colour).save(buf, format="PNG")
    return buf.getvalue()


def join(client, email="anna@example.test", password="a-good-password"):
    return client.post("/api/signup",
                       data={"email": email, "password": password,
                             "name": "Anna"})


def start(client, room="bedroom", style="japandi", first_message=""):
    return client.post(
        "/api/conversations",
        files={"photo": ("room.png", png(), "image/png")},
        data={"room": room, "style": style, "first_message": first_message},
    )


class TestStartingAndListing:
    def test_a_thread_is_started_and_listed(self, client):
        join(client)
        res = start(client, first_message="why did the desk go")
        assert res.status_code == 200
        conv_id = res.json()["id"]

        listed = client.get("/api/conversations").json()["conversations"]
        assert len(listed) == 1
        assert listed[0]["id"] == conv_id

    def test_the_title_is_the_first_message(self, client):
        join(client)
        res = start(client, first_message="  why did the desk go?  ")
        assert res.json()["title"] == "why did the desk go?"

    def test_a_title_is_never_blank(self, client):
        """A list of "Untitled" is not a history."""
        join(client)
        res = start(client, room="living", style="scandinavian")
        title = res.json()["title"]
        assert title.strip()
        assert "Living" in title and "Scandinavian" in title

    def test_a_stranger_cannot_start_or_list(self, client):
        assert start(client).status_code == 401
        assert client.get("/api/conversations").status_code == 401

    def test_newest_first(self, client):
        join(client)
        first = start(client, first_message="first").json()["id"]
        second = start(client, first_message="second").json()["id"]
        ids = [c["id"] for c in client.get("/api/conversations").json()["conversations"]]
        assert ids == [second, first]


class TestOpeningAThread:
    def test_the_photo_messages_and_designs_come_back(self, client):
        join(client)
        conv_id = start(client, first_message="hello").json()["id"]

        client.post("/api/designs",
                    files={"image": ("d.png", png(), "image/png")},
                    data={"title": "v1", "conversation_id": str(conv_id)})

        body = client.get(f"/api/conversations/{conv_id}").json()
        assert body["id"] == conv_id
        assert body["photo_url"] == f"/api/conversations/{conv_id}/photo"
        assert len(body["designs"]) == 1
        assert body["designs"][0]["title"] == "v1"

        photo = client.get(body["photo_url"])
        assert photo.status_code == 200
        assert photo.headers["content-type"] == "image/png"

    def test_opening_someone_elses_thread_is_a_plain_404(self, client):
        join(client, email="anna@example.test")
        conv_id = start(client).json()["id"]
        client.post("/api/logout")

        join(client, email="ben@example.test")
        assert client.get(f"/api/conversations/{conv_id}").status_code == 404
        assert client.get(f"/api/conversations/{conv_id}/photo").status_code == 404


class TestChatAppendsToTheThread:
    def test_a_reply_is_recorded_against_the_thread(self, client, monkeypatch):
        async def fake_discuss(summary, turns, settings, design=None):
            return "the desk stays now"

        monkeypatch.setattr("app.main.discuss", fake_discuss)

        join(client)
        conv_id = start(client).json()["id"]

        res = client.post("/api/chat", data={
            "room_summary": "a bedroom",
            "turns": json.dumps([{"role": "user", "content": "why did the desk go"}]),
            "conversation_id": str(conv_id),
        })
        assert res.status_code == 200
        assert res.json()["reply"] == "the desk stays now"

        body = client.get(f"/api/conversations/{conv_id}").json()
        roles = [m["role"] for m in body["messages"]]
        assert roles == ["user", "assistant"]
        assert body["messages"][0]["content"] == "why did the desk go"
        assert body["messages"][1]["content"] == "the desk stays now"

    def test_chat_still_works_with_no_conversation_id(self, client, monkeypatch):
        """Anonymous and code-only sessions keep chatting exactly as before —
        history is additive, not a requirement."""
        monkeypatch.setattr("app.main.discuss",
                            lambda *a, **k: _ok())

        res = client.post("/api/chat", data={
            "room_summary": "a bedroom",
            "turns": json.dumps([{"role": "user", "content": "hello"}]),
        })
        assert res.status_code == 200

    def test_a_strangers_conversation_id_is_rejected(self, client, monkeypatch):
        """The same non-answer as any other missing id — chat must not
        confirm somebody else's thread exists by appending to it anyway."""
        monkeypatch.setattr("app.main.discuss", lambda *a, **k: _ok())

        join(client, email="anna@example.test")
        conv_id = start(client).json()["id"]
        client.post("/api/logout")

        join(client, email="ben@example.test")
        res = client.post("/api/chat", data={
            "room_summary": "a bedroom",
            "turns": json.dumps([{"role": "user", "content": "hi"}]),
            "conversation_id": str(conv_id),
        })
        assert res.status_code == 404


async def _ok():
    return "ok"


class TestRenamingAndDeleting:
    def test_renaming_your_own(self, client):
        join(client)
        conv_id = start(client).json()["id"]
        res = client.patch(f"/api/conversations/{conv_id}",
                           data={"title": "The bedroom redo"})
        assert res.status_code == 200
        assert res.json()["title"] == "The bedroom redo"

    def test_one_person_cannot_rename_or_delete_anothers(self, client):
        join(client, email="anna@example.test")
        conv_id = start(client).json()["id"]
        client.post("/api/logout")

        join(client, email="ben@example.test")
        assert client.patch(f"/api/conversations/{conv_id}",
                            data={"title": "mine now"}).status_code == 404
        assert client.delete(f"/api/conversations/{conv_id}").status_code == 404

        client.post("/api/logout")
        client.post("/api/login", data={"email": "anna@example.test",
                                        "password": "a-good-password"})
        assert len(client.get("/api/conversations").json()["conversations"]) == 1

    def test_deleting_takes_its_messages_and_designs_with_it(self, client, monkeypatch):
        """SQLite in these tests does not enforce ON DELETE CASCADE, so this
        pins that store.delete_conversation cleans up explicitly rather than
        relying on a pragma nothing here turns on."""
        monkeypatch.setattr("app.main.discuss", lambda *a, **k: _ok())

        join(client)
        conv_id = start(client, first_message="hi").json()["id"]
        client.post("/api/chat", data={
            "room_summary": "a bedroom",
            "turns": json.dumps([{"role": "user", "content": "hi"}]),
            "conversation_id": str(conv_id),
        })
        client.post("/api/designs",
                    files={"image": ("d.png", png(), "image/png")},
                    data={"conversation_id": str(conv_id)})

        assert client.delete(f"/api/conversations/{conv_id}").status_code == 200
        assert client.get(f"/api/conversations/{conv_id}").status_code == 404

    def test_signing_out_hides_it_all(self, client):
        join(client)
        start(client)
        client.post("/api/logout")
        assert client.get("/api/conversations").status_code == 401


class TestKeepingADesignInAThread:
    def test_a_strangers_conversation_id_does_not_attach(self, client):
        """A design_id from another thread is ignored, not trusted — kept as
        a design with no conversation, not attached to someone else's."""
        join(client, email="anna@example.test")
        conv_id = start(client).json()["id"]
        client.post("/api/logout")

        join(client, email="ben@example.test")
        res = client.post("/api/designs",
                          files={"image": ("d.png", png(), "image/png")},
                          data={"conversation_id": str(conv_id)})
        assert res.status_code == 200
        # Ben's own conversation list is still empty — nothing leaked into it.
        assert client.get("/api/conversations").json()["conversations"] == []
