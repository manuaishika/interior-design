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

import asyncio
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
            "No GOOGLE_API_KEY set. Make one at aistudio.google.com/apikey and "
            "add it to your environment. Reading the room is free; image "
            "generation now needs billing enabled on the Cloud project."
        )
    return settings.google_api_key


def _part_image(data: bytes, mime: str = "image/jpeg") -> dict:
    return {"inline_data": {"mime_type": mime,
                            "data": base64.b64encode(data).decode("ascii")}}


# 503 ("high demand") and a bare 429 rate-limit are both transient on the free
# tier — the same request succeeds a few seconds later. A free-tier *image*
# quota of 0, though, comes back 429 every time and must not be retried into a
# long stall, so that one is told apart by its quota metric and raised at once.
_RETRY_STATUSES = {503, 429}
_MAX_TRIES = 4
# Indirected so tests can neutralise the backoff without waiting on it.
_sleep = asyncio.sleep


def _is_hard_quota(response: httpx.Response) -> bool:
    """A 429 that says the per-model limit is 0 — retrying will never clear it."""
    try:
        message = response.json()["error"]["message"]
    except Exception:
        return False
    return "limit: 0" in message or "free_tier" in message


async def _call(model: str, body: dict, settings: Settings) -> dict:
    url = f"{BASE}/{model}:generateContent"
    last: httpx.Response | None = None
    for attempt in range(_MAX_TRIES):
        async with httpx.AsyncClient(timeout=settings.request_timeout_s) as http:
            response = await http.post(
                url, params={"key": _key(settings)}, json=body,
                headers={"Content-Type": "application/json"},
            )
        if response.status_code < 400:
            return response.json()

        last = response
        retryable = response.status_code in _RETRY_STATUSES
        if response.status_code == 429 and _is_hard_quota(response):
            retryable = False
        if not retryable or attempt == _MAX_TRIES - 1:
            break
        wait = min(2.0 * (2 ** attempt), 8.0)
        log.warning("Google %s on %s; retrying in %.0fs (%d/%d)",
                    response.status_code, model, wait, attempt + 1, _MAX_TRIES)
        await _sleep(wait)

    assert last is not None
    if last.status_code == 429 and _is_hard_quota(last):
        raise GoogleError(
            "This Google model has no free-tier quota. Image generation on the "
            "Gemini API needs billing enabled on the Cloud project (a card, "
            "pay-as-you-go, about 4 cents an image; no ID check). Reading the "
            "room stays free."
        )
    if last.status_code == 429:
        raise GoogleError(
            "The free tier's rate limit was hit and did not clear after several "
            "retries. Wait a minute, or enable billing for room-after-room use."
        )
    raise GoogleError(_explain(last))


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


def _truncated(payload: dict) -> bool:
    """True when Gemini stopped because it hit the output-token ceiling."""
    for candidate in payload.get("candidates") or []:
        return candidate.get("finishReason") == "MAX_TOKENS"
    return False


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
                # The survey answer (is_room, room, every item, three full
                # directions) plus Gemini 3's thinking tokens overran the old
                # 1600 cap and came back as truncated JSON. Give it room, and
                # hold the thinking down so the budget goes on the answer.
                "maxOutputTokens": 8192,
                "thinkingConfig": {"thinkingLevel": "low"},
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
        if _truncated(payload):
            raise GoogleError(
                "The reader ran out of output space before finishing. Retry, "
                "or raise google_vision_model's maxOutputTokens."
            ) from exc
        raise GoogleError("The reader returned something unreadable") from exc


async def discuss(system: str, turns: list[dict], settings: Settings,
                  design: bytes | None = None) -> str:
    """Keep talking about a room that has already been read.

    `design` is the generated render, when one exists. Without it this model
    is in the same position the OpenAI path used to be in: told only what the
    *original* photograph contained, so it argues with the client about a
    picture it has never seen. Attached to the newest turn, the same way the
    reader attaches the room photo, so it reads as "this is what I am being
    asked about" rather than background.
    """
    contents = []
    for turn in turns[-12:]:
        role = "model" if turn.get("role") == "assistant" else "user"
        text = str(turn.get("content", ""))[:4000]
        if text:
            contents.append({"role": role, "parts": [{"text": text}]})
    if not contents:
        raise GoogleError("Nothing was asked")

    if design and contents[-1]["role"] == "user":
        contents[-1]["parts"].append(_part_image(design, mime="image/png"))

    payload = await _call(
        settings.google_vision_model,
        {
            "contents": contents,
            "system_instruction": {"parts": [{"text": system}]},
            "generationConfig": {
                "temperature": 0.7,
                # Thinking tokens count against this, so 700 could leave the
                # visible answer cut off mid-sentence.
                "maxOutputTokens": 2048,
                "thinkingConfig": {"thinkingLevel": "low"},
            },
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
# instruction alone, and a vague mood word ("warmer") is not concrete enough
# for it to act on. The first slot used to be "" — no instruction at all, so
# one option in every batch was never told to differ from anything. Every
# slot now names something physical — a colour, a material, a silhouette —
# which is what actually pushes two runs of the same model apart.
VARIATIONS = (
    " Make one confident, specific choice — one accent colour, one material, "
    "one statement piece — and carry it through the whole room rather than "
    "scattering small variations.",
    " Take the quiet version: a tight neutral palette, natural materials, "
    "nothing glossy or saturated, more empty floor than furniture.",
    " Change the furniture's actual silhouettes, not just their colours — a "
    "different shape of bed frame, desk or seating — while keeping the same "
    "style and the same footprint in the room.",
    " Lead with a different dominant material than a plain reading would: "
    "stone, metal or lacquer where wood would be the obvious choice, or the "
    "reverse.",
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
