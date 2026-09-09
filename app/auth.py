"""Who is signed in.

A session cookie carrying a signed, expiring claim about *which account* is
using the studio. That identity is what makes saved designs possible: you
cannot answer "where are my designs" from a shared password.

Two ways in, both landing in the same account:
  - email and password
  - Continue with Google

The access code is still here but demoted. It answers "is this deployment
open to the public yet", which is a different question from "who are you",
and it is off unless somebody sets one. A free-tier visitor signs in with an
account like anyone else.

Why a cookie and not a header
-----------------------------
So a reload keeps you in, and so a Google redirect can land back on the site
already signed in. HttpOnly, so page scripts — ours or anyone else's — cannot
read it back out.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time

from fastapi import HTTPException, Request, Response

from .config import Settings

COOKIE = "sd_session"
LIFETIME_S = 30 * 24 * 60 * 60          # a month, then sign in again

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


def issue(settings: Settings, user_id: int = 0) -> tuple[str, int]:
    """A token says who, and until when. Both are signed together, so neither
    the account nor the expiry can be edited without invalidating it."""
    expiry = int(time.time()) + LIFETIME_S
    body = f"{user_id}.{expiry}"
    return f"{body}.{_sign(body, settings)}", LIFETIME_S


def read(token: str | None, settings: Settings) -> int | None:
    """The account id inside a valid token, or None. 0 means a code-only
    session, which is signed in but is nobody in particular."""
    if not token:
        return None
    parts = token.split(".")
    if len(parts) != 3:
        return None
    user_id, expiry, signature = parts
    try:
        if int(expiry) < time.time():
            return None
        who = int(user_id)
    except ValueError:
        return None
    # compare_digest, not ==, so a wrong signature cannot be found one
    # character at a time by watching how long the answer takes.
    if not hmac.compare_digest(signature, _sign(f"{user_id}.{expiry}", settings)):
        return None
    return who


def valid(token: str | None, settings: Settings) -> bool:
    return read(token, settings) is not None


def matches(code: str, settings: Settings) -> bool:
    return hmac.compare_digest((code or "").strip(),
                               settings.studio_access_code)


def current_user_id(request: Request, settings: Settings) -> int | None:
    """The signed-in account, if there is one. None for a visitor, and None
    for a code-only session, which is admitted but is nobody."""
    who = read(request.cookies.get(COOKIE), settings)
    return who if who else None


def signed_in(request: Request, settings: Settings) -> bool:
    if valid(request.cookies.get(COOKIE), settings):
        return True
    # With no access code set the studio is open to anyone; signing in is then
    # about keeping your work, not about getting in.
    return not required(settings)


def guard(request: Request, settings: Settings) -> None:
    """Refuse anything that costs money to an unsigned visitor."""
    if not signed_in(request, settings):
        raise HTTPException(401, "Sign in to use the studio.")


def require_user(request: Request, settings: Settings) -> int:
    """For anything belonging to a person. A code-only session is not one."""
    who = current_user_id(request, settings)
    if who is None:
        raise HTTPException(401, "Sign in to keep your designs.")
    return who


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
