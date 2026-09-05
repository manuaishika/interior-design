"""Tests for the door on the studio.

The point of the door is that nobody can spend your quota without the code.
So what is pinned here is refusal: not that a right code works, but that a
wrong one, a missing one, a forged one and an expired one all fail.
"""

import io
import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import app.main as main
from app import auth
from app.config import Settings
from app.main import app

CODE = "open-sesame"


def locked(**kw):
    kw.setdefault("session_secret", "s3cret")
    return Settings(studio_access_code=CODE, **kw)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main, "get_settings", locked)
    return TestClient(app)


def photo():
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (200, 190, 180)).save(buf, format="PNG")
    return {"photo": ("room.png", buf.getvalue(), "image/png")}


class TestTheDoorExistsOnlyWhenAsked:
    def test_no_code_means_no_door(self, monkeypatch):
        """Local development, and any deployment that wants to stay open, must
        be unaffected by this whole file."""
        monkeypatch.setattr(main, "get_settings", Settings)
        body = TestClient(app).get("/api/session").json()
        assert body["required"] is False and body["signed_in"] is True

    def test_a_code_raises_the_door(self, client):
        body = client.get("/api/session").json()
        assert body["required"] is True and body["signed_in"] is False


class TestSigningIn:
    def test_the_right_code_gets_you_in(self, client):
        assert client.post("/api/login", data={"code": CODE}).status_code == 200
        assert client.get("/api/session").json()["signed_in"] is True

    def test_surrounding_space_is_forgiven(self, client):
        """People paste codes out of messages."""
        assert client.post("/api/login",
                           data={"code": "  " + CODE + " "}).status_code == 200

    def test_the_wrong_code_is_refused(self, client):
        res = client.post("/api/login", data={"code": "guess"})
        assert res.status_code == 401
        assert client.get("/api/session").json()["signed_in"] is False

    def test_signing_out_closes_the_door_again(self, client):
        client.post("/api/login", data={"code": CODE})
        client.post("/api/logout")
        assert client.get("/api/session").json()["signed_in"] is False

    def test_the_cookie_is_not_readable_by_scripts(self, client):
        """HttpOnly, so a stray script on the page cannot lift the session."""
        header = client.post("/api/login",
                             data={"code": CODE}).headers["set-cookie"]
        assert "httponly" in header.lower()

    def test_the_cookie_is_secure_over_https(self, client):
        """Marking it Secure on plain HTTP would make signing in impossible
        rather than safer, so it follows the actual connection."""
        plain = client.post("/api/login", data={"code": CODE})
        assert "secure" not in plain.headers["set-cookie"].lower()

        behind_tls = client.post("/api/login", data={"code": CODE},
                                 headers={"x-forwarded-proto": "https"})
        assert "secure" in behind_tls.headers["set-cookie"].lower()


class TestNothingCostsMoneyUnsigned:
    @pytest.mark.parametrize("path", ["/api/read", "/api/generate"])
    def test_uploads_are_refused(self, client, path):
        res = client.post(path, files=photo(), data={"style": "japandi"})
        assert res.status_code == 401

    def test_chat_is_refused(self, client):
        res = client.post("/api/chat",
                          data={"room_summary": "x", "turns": "[]"})
        assert res.status_code == 401

    def test_the_door_opens_the_endpoints(self, client, monkeypatch):
        async def fake_read(*a, **k):
            return {"room": "A bedroom.", "items": [], "directions": []}

        monkeypatch.setattr(main, "read_room", fake_read)
        client.post("/api/login", data={"code": CODE})
        assert client.post("/api/read", files=photo()).status_code == 200

    def test_looking_at_the_site_never_needs_a_code(self, client):
        """The shop window stays open. Only the workshop is locked."""
        for path in ("/", "/api/health", "/api/styles", "/api/session"):
            assert client.get(path).status_code == 200


class TestTokens:
    def test_a_forged_signature_is_rejected(self):
        s = locked()
        token, _ = auth.issue(s)
        stamp, _, _ = token.partition(".")
        assert auth.valid(stamp + "." + "0" * 64, s) is False

    def test_an_expired_token_is_rejected(self):
        s = locked()
        expired = f"{int(time.time()) - 10}.{auth._sign(int(time.time()) - 10, s)}"
        assert auth.valid(expired, s) is False

    def test_a_token_from_another_deployment_is_rejected(self):
        """Two servers, two secrets: a session minted on one must not open the
        other."""
        token, _ = auth.issue(locked())
        assert auth.valid(token, locked(session_secret="different")) is False

    @pytest.mark.parametrize("junk", ["", None, "nonsense", "abc.def", "..",
                                      "9999999999.", "."])
    def test_malformed_tokens_are_rejected_not_crashed_on(self, junk):
        assert auth.valid(junk, locked()) is False

    def test_a_fresh_token_is_accepted(self):
        s = locked()
        token, max_age = auth.issue(s)
        assert auth.valid(token, s) is True
        assert max_age > 0

    def test_sessions_do_not_survive_a_restart_without_a_secret(self):
        """No configured secret means a random one per process. That logs
        everyone out on deploy, which is the safe direction to fail — the
        alternative is a secret in the source that opens every deployment."""
        s = Settings(studio_access_code=CODE)
        token, _ = auth.issue(s)
        assert auth.valid(token, s) is True

        original = auth._FALLBACK_SECRET
        try:
            auth._FALLBACK_SECRET = "a-different-process"
            assert auth.valid(token, s) is False
        finally:
            auth._FALLBACK_SECRET = original
