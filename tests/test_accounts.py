"""Tests for accounts, and for the designs that belong to them.

An access code answers "are you allowed in". It cannot answer "where are my
designs", because everybody holding it is the same nobody. So what is pinned
here is mostly separation: that one person's collection is not reachable from
another person's session, by any route.

The database is a throwaway SQLite file per test.
"""

import io

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from PIL import Image

import app.main as main
from app import auth, store
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


class TestSigningUp:
    def test_an_account_is_made_and_signed_in(self, client):
        res = join(client)
        assert res.status_code == 200
        assert res.json()["account"]["email"] == "anna@example.test"
        assert client.get("/api/session").json()["account"]["name"] == "Anna"

    def test_a_short_password_is_refused(self, client):
        res = client.post("/api/signup",
                          data={"email": "a@b.test", "password": "short"})
        assert res.status_code == 400

    def test_a_nonsense_email_is_refused(self, client):
        res = client.post("/api/signup",
                          data={"email": "anna", "password": "a-good-password"})
        assert res.status_code == 400

    def test_the_same_email_twice_is_refused(self, client):
        join(client)
        client.post("/api/logout")
        assert join(client).status_code == 409

    def test_case_does_not_make_a_second_account(self, client):
        """Somebody who signs up as Anna@ and returns as anna@ is one person,
        and would otherwise find their designs gone."""
        join(client, email="Anna@Example.test")
        client.post("/api/logout")
        again = client.post("/api/signup",
                            data={"email": "anna@example.test",
                                  "password": "another-password"})
        assert again.status_code == 409


class TestSigningIn:
    def test_the_right_password_works(self, client):
        join(client)
        client.post("/api/logout")
        res = client.post("/api/login", data={"email": "anna@example.test",
                                              "password": "a-good-password"})
        assert res.status_code == 200
        assert client.get("/api/session").json()["account"]["email"] == \
            "anna@example.test"

    def test_the_wrong_password_does_not(self, client):
        join(client)
        client.post("/api/logout")
        assert client.post("/api/login",
                           data={"email": "anna@example.test",
                                 "password": "guessing"}).status_code == 401

    def test_an_unknown_email_says_the_same_thing(self, client):
        """The same refusal either way, so this cannot be used to find out
        which addresses have accounts."""
        missing = client.post("/api/login", data={"email": "nobody@example.test",
                                                  "password": "guessing"})
        join(client)
        client.post("/api/logout")
        wrong = client.post("/api/login", data={"email": "anna@example.test",
                                                "password": "guessing"})
        assert missing.status_code == wrong.status_code == 401
        assert missing.json()["detail"] == wrong.json()["detail"]

    def test_the_session_survives_a_reload(self, client):
        join(client)
        assert client.get("/api/session").json()["signed_in"] is True

    def test_signing_out_ends_it(self, client):
        join(client)
        client.post("/api/logout")
        assert client.get("/api/session").json()["account"] is None


class TestPasswords:
    def test_the_password_is_never_stored(self):
        stored = store.hash_password("a-good-password")
        assert "a-good-password" not in stored
        assert stored.startswith("scrypt$")

    def test_the_same_password_hashes_differently_each_time(self):
        """Salted, so two people with the same password do not share a hash
        and one leaked hash does not crack both."""
        assert store.hash_password("same") != store.hash_password("same")

    def test_verification_works_and_fails(self):
        stored = store.hash_password("a-good-password")
        assert store.verify_password("a-good-password", stored) is True
        assert store.verify_password("a-good-passwore", stored) is False

    @pytest.mark.parametrize("junk", ["", "notahash", "scrypt$zz$zz", "a$b"])
    def test_a_corrupt_hash_is_rejected_not_crashed_on(self, junk):
        assert store.verify_password("anything", junk) is False


class TestDesignsBelongToPeople:
    def keep(self, client, colour=(10, 20, 30)):
        return client.post("/api/designs",
                           files={"image": ("d.png", png(colour), "image/png")},
                           data={"title": "Front room", "room": "living",
                                 "style": "japandi"})

    def test_a_design_is_kept_and_listed(self, client):
        join(client)
        assert self.keep(client).status_code == 200
        designs = client.get("/api/designs").json()["designs"]
        assert len(designs) == 1
        assert designs[0]["title"] == "Front room"

    def test_the_picture_comes_back(self, client):
        join(client)
        url = self.keep(client).json()["url"]
        res = client.get(url)
        assert res.status_code == 200
        assert res.headers["content-type"] == "image/png"

    def test_a_stranger_cannot_keep_anything(self, client):
        """This is the whole reason accounts exist rather than a shared code:
        without a person, there is nowhere to put it."""
        assert self.keep(client).status_code == 401
        assert client.get("/api/designs").status_code == 401

    def test_one_persons_designs_are_not_anothers(self, client):
        join(client, email="anna@example.test")
        anna_url = self.keep(client, (255, 0, 0)).json()["url"]
        client.post("/api/logout")

        join(client, email="ben@example.test")
        assert client.get("/api/designs").json()["designs"] == []
        # 404, not 403 — Ben should not learn that Anna's design exists.
        assert client.get(anna_url).status_code == 404

    def test_one_person_cannot_delete_anothers(self, client):
        join(client, email="anna@example.test")
        anna_id = self.keep(client).json()["id"]
        client.post("/api/logout")

        join(client, email="ben@example.test")
        assert client.delete(f"/api/designs/{anna_id}").status_code == 404

        client.post("/api/logout")
        client.post("/api/login", data={"email": "anna@example.test",
                                        "password": "a-good-password"})
        assert len(client.get("/api/designs").json()["designs"]) == 1

    def test_deleting_your_own_works(self, client):
        join(client)
        design_id = self.keep(client).json()["id"]
        assert client.delete(f"/api/designs/{design_id}").status_code == 200
        assert client.get("/api/designs").json()["designs"] == []

    def test_newest_first(self, client):
        join(client)
        first = self.keep(client).json()["id"]
        second = self.keep(client).json()["id"]
        ids = [d["id"] for d in client.get("/api/designs").json()["designs"]]
        assert ids[0] == second and first in ids


class TestTheCodeIsNoLongerTheLogin:
    def test_free_visitors_need_no_code_to_sign_up(self, client):
        """The complaint that started this: an access code gated the studio,
        so somebody on the free tier could not make an account at all."""
        assert client.get("/api/session").json()["required"] is False
        assert join(client).status_code == 200

    def test_a_code_only_session_is_admitted_but_is_nobody(
            self, monkeypatch, tmp_path, db):
        """It opens the studio and cannot save a thing, which is exactly the
        difference between being let in and being someone."""
        gated = Settings(session_secret="s", studio_access_code="hello",
                         database_url=f"sqlite+aiosqlite:///{tmp_path}/t.db")
        monkeypatch.setattr(main, "get_settings", lambda: gated)
        with TestClient(app) as c:
            assert c.post("/api/login", data={"code": "hello"}).status_code == 200
            state = c.get("/api/session").json()
            assert state["signed_in"] is True
            assert state["account"] is None
            assert c.get("/api/designs").status_code == 401


class TestGoogle:
    def test_the_button_is_hidden_until_it_is_set_up(self, client):
        assert client.get("/api/session").json()["google_available"] is False

    def test_starting_without_credentials_is_refused(self, client):
        assert client.get("/api/auth/google",
                          follow_redirects=False).status_code == 503

    def test_a_mismatched_state_never_signs_anyone_in(self, monkeypatch,
                                                      tmp_path, db):
        """The check that stops somebody else's sign-in being replayed at you
        to land you in their account."""
        configured = Settings(session_secret="s", google_client_id="id",
                              google_client_secret="secret",
                              database_url=f"sqlite+aiosqlite:///{tmp_path}/t.db")
        monkeypatch.setattr(main, "get_settings", lambda: configured)
        with TestClient(app) as c:
            res = c.get("/api/auth/google/callback?code=x&state=forged",
                        follow_redirects=False)
            assert res.status_code == 303
            assert c.get("/api/session").json()["account"] is None

    def test_the_start_url_asks_which_account(self, settings):
        """Somebody signed into three Google accounts should be asked, not
        silently given whichever they used last."""
        from app import google_login

        s = Settings(google_client_id="id", google_client_secret="secret")
        url = google_login.start_url(s, "https://x.test/cb", "state123")
        assert "accounts.google.com" in url
        assert "prompt=select_account" in url
        assert "state=state123" in url
        assert "scope=openid+email+profile" in url

    @pytest.mark.asyncio
    async def test_google_and_password_land_in_one_account(self, db):
        """Somebody who signed up with a password and later clicks Continue
        with Google must land in their own account, not a second empty one."""
        async with store.session() as s:
            first = await store.sign_up(s, "anna@example.test", "a-good-password")
            again = await store.from_google(s, "google-123",
                                            "Anna@Example.test", "Anna")
            assert again.id == first.id
            assert again.google_id == "google-123"

    @pytest.mark.asyncio
    async def test_a_google_only_account_has_no_password_to_guess(self, db):
        async with store.session() as s:
            user = await store.from_google(s, "google-9", "new@example.test")
            assert user.password_hash is None
            assert await store.sign_in(s, "new@example.test", "") is None
