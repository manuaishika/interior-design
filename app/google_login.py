"""Continue with Google.

The standard server-side authorization-code flow, in about eighty lines and
with no new dependency.

Why there is no JWT library here
--------------------------------
Google's token response contains an `id_token`, and verifying one properly
means fetching Google's signing keys, checking the algorithm, the issuer, the
audience and the expiry — a library's worth of work that is easy to get subtly
and silently wrong.

None of it is necessary for this flow. The code is exchanged over TLS directly
with Google's token endpoint, so the response did not travel through the
browser and there is nothing to forge. We then ask Google's userinfo endpoint
who the access token belongs to. Google answers. That is the identity.

This reasoning holds *only* for the server-side code flow. An id_token that
arrives from a browser is a different matter entirely and must be verified.
"""

from __future__ import annotations

import secrets
from urllib.parse import urlencode

import httpx

AUTH = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN = "https://oauth2.googleapis.com/token"
USERINFO = "https://openidconnect.googleapis.com/v1/userinfo"

STATE_COOKIE = "sd_oauth_state"


class GoogleLoginError(RuntimeError):
    pass


def configured(settings) -> bool:
    return bool(settings.google_client_id and settings.google_client_secret)


def new_state() -> str:
    return secrets.token_urlsafe(24)


def start_url(settings, redirect_uri: str, state: str) -> str:
    return AUTH + "?" + urlencode({
        "client_id": settings.google_client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": "openid email profile",
        "state": state,
        # So somebody signed into three Google accounts is asked which one,
        # rather than silently getting whichever they used last.
        "prompt": "select_account",
    })


async def exchange(settings, code: str, redirect_uri: str) -> dict:
    """Swap the one-time code for an access token, then ask who it belongs to."""
    async with httpx.AsyncClient(timeout=20.0) as http:
        token_response = await http.post(TOKEN, data={
            "code": code,
            "client_id": settings.google_client_id,
            "client_secret": settings.google_client_secret,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code",
        })
        if token_response.status_code >= 400:
            raise GoogleLoginError(_explain(token_response, redirect_uri))

        access_token = token_response.json().get("access_token")
        if not access_token:
            raise GoogleLoginError("Google returned no access token.")

        who = await http.get(USERINFO,
                             headers={"Authorization": f"Bearer {access_token}"})
        if who.status_code >= 400:
            raise GoogleLoginError("Google would not say who signed in.")

    profile = who.json()
    if not profile.get("email"):
        raise GoogleLoginError("That Google account has no email address on it.")
    # An unverified address must not be trusted: it would let someone claim an
    # account belonging to whoever really owns that mailbox.
    if profile.get("email_verified") is False:
        raise GoogleLoginError("That Google account's email is not verified.")

    return {
        "google_id": str(profile.get("sub") or ""),
        "email": profile["email"],
        "name": profile.get("name") or "",
    }


def _explain(response: httpx.Response, redirect_uri: str) -> str:
    """redirect_uri_mismatch is the error everyone hits, and Google's own
    message does not say what to do about it."""
    try:
        error = response.json().get("error", "")
    except Exception:
        error = response.text[:200]
    if error == "redirect_uri_mismatch":
        return (
            "Google rejected the redirect address. Add exactly this URI to "
            f"your OAuth client's authorised redirect URIs: {redirect_uri}"
        )
    if error == "invalid_client":
        return "Google rejected the client id or secret. Check both."
    return f"Google refused the sign-in ({error})."
