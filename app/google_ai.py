"""The free path: one Google AI Studio key, no card, both halves of the job.

Why this exists
---------------
The paid path needs two accounts — OpenAI to read the room, Replicate to
redraw it — and until somebody pays for both, the product does nothing at
all. That is a bad place to demo from.

Google AI Studio hands out a key with no card attached, and one key covers
both halves: a vision model that reads the photograph, and an image model
that edits it. So the whole thing runs, free, on a key you can make yourself
in about two minutes.

What you give up
----------------
The paid path cuts the photo into regions with SAM2 and turns the doors and
windows into an actual inpainting mask — the generator is not *able* to
repaint them. This path has no mask. It asks the model to leave the
architecture alone, firmly and specifically, and mostly it obliges.

Asking nicely is weaker than a mask, and that difference is the honest answer
to "why would we pay for the other one".

Talking to it
-------------
Plain HTTPS via httpx rather than another SDK: two endpoints, one shape, and
one less dependency to install on a free dyno.
"""

from __future__ import annotations

import base64
import json
import logging

import httpx

from .config import Settings

log = logging.getLogger(__name__)

BASE = "https://generativelanguage.googleapis.com/v1beta/models"


class GoogleError(RuntimeError):
    pass


def _key(settings: Settings) -> str:
    if not settings.google_api_key:
        raise GoogleError(
            "No GOOGLE_API_KEY set. Make one free at aistudio.google.com/apikey "
            "— no card needed — and add it to your environment."
        )
    return settings.google_api_key


def _part_image(data: bytes, mime: str = "image/jpeg") -> dict:
    return {"inline_data": {"mime_type": mime,
                            "data": base64.b64encode(data).decode("ascii")}}


async def _call(model: str, body: dict, settings: Settings) -> dict:
    url = f"{BASE}/{model}:generateContent"
    async with httpx.AsyncClient(timeout=settings.request_timeout_s) as http:
        response = await http.post(
            url, params={"key": _key(settings)}, json=body,
            headers={"Content-Type": "application/json"},
        )
    if response.status_code == 429:
        raise GoogleError(
            "The free tier's rate limit was hit. Wait a minute and try again, "
            "or move to the paid path for room-after-room use."
        )
    if response.status_code >= 400:
        raise GoogleError(_explain(response))
    return response.json()


def _explain(response: httpx.Response) -> str:
    """Google's errors are nested and long; the useful sentence is buried."""
    try:
        message = response.json()["error"]["message"]
    except Exception:
        message = response.text[:300]
    if response.status_code in (401, 403):
        return f"Google rejected the key ({message}). Check GOOGLE_API_KEY."
    return f"Google returned {response.status_code}: {message}"


def _parts(payload: dict) -> list[dict]:
    """Every part of the first candidate, or nothing if it was refused."""
    for candidate in payload.get("candidates") or []:
        content = candidate.get("content") or {}
        return content.get("parts") or []
    return []


def _inline(part: dict) -> dict | None:
    """Responses come back camelCase, requests go out snake_case. Accept both
    rather than depending on which side of that line this API is on today."""
    return part.get("inlineData") or part.get("inline_data")


# ---------------------------------------------------------------------------
# Reading the room
# ---------------------------------------------------------------------------

async def read_room(photo: bytes, room_type: str, prompt: str,
                    settings: Settings) -> dict:
    """Survey the photograph and return the same JSON the paid reader returns.

    The prompt is passed in rather than duplicated here, so both readers are
    held to one description of the job and cannot drift apart.
    """
    payload = await _call(
        settings.google_vision_model,
        {
            "contents": [{"parts": [{"text": prompt}, _part_image(photo)]}],
            "generationConfig": {
                "response_mime_type": "application/json",
                "temperature": 0.4,
                "maxOutputTokens": 1600,
            },
        },
        settings,
    )

    text = "".join(part.get("text", "") for part in _parts(payload)).strip()
    if not text:
        raise GoogleError("The reader returned nothing to read.")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise GoogleError("The reader returned something unreadable") from exc


async def discuss(system: str, turns: list[dict], settings: Settings) -> str:
    """Keep talking about a room that has already been read."""
    contents = []
    for turn in turns[-12:]:
        role = "model" if turn.get("role") == "assistant" else "user"
        text = str(turn.get("content", ""))[:4000]
        if text:
            contents.append({"role": role, "parts": [{"text": text}]})
    if not contents:
        raise GoogleError("Nothing was asked")

    payload = await _call(
        settings.google_vision_model,
        {
            "contents": contents,
            "system_instruction": {"parts": [{"text": system}]},
            "generationConfig": {"temperature": 0.7, "maxOutputTokens": 700},
        },
        settings,
    )
    return "".join(part.get("text", "") for part in _parts(payload)).strip()


# ---------------------------------------------------------------------------
# Redrawing the room
# ---------------------------------------------------------------------------

# Without a mask, the lock has to live in the sentence. Spelling out what
# "structure" means beats one word the model is free to interpret loosely.
KEEP_THE_BUILDING = (
    "Keep the architecture exactly as photographed. Do not move, resize, "
    "remove or add any door, doorway, window, wall or opening, and do not "
    "change the ceiling height or the shape of the room. Keep the camera in "
    "the same position with the same perspective and the same view through "
    "any window. Change only the furniture, soft furnishings, surface "
    "finishes, lighting fixtures and decoration."
)

# One image model, one photograph, no seed — so options have to differ by
# instruction. Each nudge asks for a genuinely different room, not a re-roll.
VARIATIONS = (
    "",
    " Take a warmer, softer reading of this style, with more textile and "
    "more layered lighting.",
    " Take a cooler, more pared-back reading of this style, with fewer "
    "pieces and more empty floor.",
    " Take a bolder reading of this style, with one strong colour and one "
    "sculptural piece as the focus.",
)


async def redraw(photo: bytes, prompt: str, settings: Settings,
                 variant: int = 0) -> bytes:
    """Return a redrawn room as PNG/JPEG bytes."""
    instruction = (
        f"{prompt}\n\n{KEEP_THE_BUILDING}"
        f"{VARIATIONS[variant % len(VARIATIONS)]}\n\n"
        "Return the edited photograph as an image."
    )
    payload = await _call(
        settings.google_image_model,
        {"contents": [{"parts": [{"text": instruction}, _part_image(photo)]}]},
        settings,
    )

    for part in _parts(payload):
        inline = _inline(part)
        if inline and inline.get("data"):
            try:
                return base64.b64decode(inline["data"])
            except Exception as exc:
                raise GoogleError("The drawing came back corrupt") from exc

    # A refusal comes back as prose where the picture should be. Show it:
    # "I can't edit photographs of people" is worth reading, not swallowing.
    said = "".join(part.get("text", "") for part in _parts(payload)).strip()
    raise GoogleError(
        f"No picture came back. The model said: {said[:300]}" if said
        else "No picture came back."
    )
