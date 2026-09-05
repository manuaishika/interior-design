"""A door on the studio.

What this is
------------
A shared access code, checked by the server, that gates the endpoints which
cost money. You hand the code to whoever should be able to try it.

What this is not
----------------
Accounts. There is no sign-up, no email, no password reset and no per-person
history, because there are no people in a database yet. Building all of that
before anyone has logged in once would be inventing requirements.

The upgrade path is real, though: swap `_matches` for a user lookup and the
cookie already carries a signed, expiring session.

Why a cookie and not a header
-----------------------------
So a reload keeps you in. HttpOnly so page scripts — ours or anyone's —
cannot read it back out.

No code set means no door at all, so local development and any deployment
that wants to stay open are unaffected.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time

from fastapi import HTTPException, Request, Response

from .config import Settings

COOKIE = "sd_session"
LIFETIME_S = 14 * 24 * 60 * 60          # a fortnight, then sign in again

# Regenerated whenever the process restarts unless one is configured. That
# logs everyone out on deploy, which is the safe direction to fail: a
# hardcoded fallback secret would mean anyone holding this source could mint
# a session for any deployment of it.
_FALLBACK_SECRET = secrets.token_hex(32)


def _secret(settings: Settings) -> bytes:
    return (settings.session_secret or _FALLBACK_SECRET).encode()


def required(settings: Settings) -> bool:
    """A door exists only if a code was set for it."""
    return bool(settings.studio_access_code)


def _sign(expiry: int, settings: Settings) -> str:
    return hmac.new(_secret(settings), str(expiry).encode(),
                    hashlib.sha256).hexdigest()


def issue(settings: Settings) -> tuple[str, int]:
    expiry = int(time.time()) + LIFETIME_S
    return f"{expiry}.{_sign(expiry, settings)}", LIFETIME_S


def valid(token: str | None, settings: Settings) -> bool:
    if not token or "." not in token:
        return False
    stamp, _, signature = token.partition(".")
    try:
        expiry = int(stamp)
    except ValueError:
        return False
    if expiry < time.time():
        return False
    # compare_digest, not ==, so a wrong signature cannot be found one
    # character at a time by watching how long the answer takes.
    return hmac.compare_digest(signature, _sign(expiry, settings))


def matches(code: str, settings: Settings) -> bool:
    return hmac.compare_digest((code or "").strip(),
                               settings.studio_access_code)


def signed_in(request: Request, settings: Settings) -> bool:
    return not required(settings) or valid(request.cookies.get(COOKIE), settings)


def guard(request: Request, settings: Settings) -> None:
    """Refuse anything that costs money to an unsigned visitor."""
    if not signed_in(request, settings):
        raise HTTPException(401, "Sign in with the studio access code first.")


def over_https(request: Request) -> bool:
    """Whether the visitor's connection is encrypted.

    A host like Render terminates TLS at its proxy and forwards plain HTTP, so
    the app's own scheme says "http" on a site that is emphatically HTTPS. The
    forwarded header is the one that describes the real connection.
    """
    forwarded = request.headers.get("x-forwarded-proto", "")
    scheme = forwarded.split(",")[0].strip() or request.url.scheme
    return scheme == "https"


def set_cookie(response: Response, token: str, max_age: int,
               secure: bool = True) -> None:
    response.set_cookie(
        COOKIE, token,
        max_age=max_age,
        httponly=True,      # scripts cannot read it, so a stray one cannot leak it
        samesite="lax",
        # Secure over HTTPS, which is everywhere real. Marking it Secure on a
        # plain-HTTP connection does not harden anything — the browser simply
        # discards the cookie and nobody can sign in at all.
        secure=secure,
        path="/",
    )


def clear_cookie(response: Response) -> None:
    response.delete_cookie(COOKIE, path="/")
