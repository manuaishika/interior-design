"""Reading a room, and talking about it, from your own server.

The published-artifact build could ask Claude straight from the browser. A
deployment cannot: that route only exists inside claude.ai. So the same two
jobs live here as ordinary endpoints backed by your own key, which is what
makes the app deployable anywhere.

Vision only — no GPU. This runs fine on Vercel, Render, Fly, a cheap VPS.
Only picture *generation* needs a GPU, and that goes to Replicate.
"""

from __future__ import annotations

import base64
import json
import logging

from openai import AsyncOpenAI

from .config import Settings

log = logging.getLogger(__name__)


class ReadingError(RuntimeError):
    pass


SURVEY = """\
You are surveying a {room} from one photograph, for an interior designer.

Return JSON only, exactly this shape:
{{"room": "one sentence on the room and its condition",
  "items": [{{"name": "bed", "count": 2, "treatment": "keep"}}],
  "directions": [{{"title": "", "palette": "", "pieces": ["", ""], "why": ""}}]}}

items: everything notable you can see, with how many there are. Count
carefully — two single beds are two, not one double.
treatment is "keep" for doors, windows and the floor people walk on, because a
renovation must not move those. Everything else is "redraw".

directions: exactly three genuinely different directions for THIS room, not
generic advice. palette is three or four colours and materials. pieces is three
specific things to buy. why is one sentence on who it suits.
"""

DESIGNER = """\
You are an interior designer talking to a client about this room:

{room}

Be specific and brief — a short paragraph. Name real furniture and materials.
Never move a door or a window; say so plainly if asked to.
"""


def _client(settings: Settings) -> AsyncOpenAI:
    if not settings.openai_api_key:
        raise ReadingError(
            "No OPENAI_API_KEY set — the room reader needs one. "
            "Add it to your environment and restart."
        )
    return AsyncOpenAI(
        api_key=settings.openai_api_key, timeout=settings.request_timeout_s
    )


async def read_room(photo: bytes, room_type: str, settings: Settings) -> dict:
    """Look at the photograph and describe what is in the room."""
    data_uri = "data:image/jpeg;base64," + base64.b64encode(photo).decode("ascii")

    try:
        response = await _client(settings).chat.completions.create(
            model=settings.vlm_model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": SURVEY.format(room=room_type or "room")},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }],
            response_format={"type": "json_object"},
            max_tokens=1200,
            temperature=0.4,
        )
    except ReadingError:
        raise
    except Exception as exc:
        raise ReadingError(f"Could not read the room: {exc}") from exc

    try:
        return json.loads(response.choices[0].message.content or "{}")
    except json.JSONDecodeError as exc:
        raise ReadingError("The reader returned something unreadable") from exc


async def discuss(
    room_summary: str, turns: list[dict], settings: Settings
) -> str:
    """Continue the conversation about a room that has already been read.

    The room summary is re-sent every turn: the model holds no memory between
    calls, and without it the answers drift into generic advice.
    """
    if not turns:
        raise ReadingError("Nothing was asked")

    messages = [{"role": "system", "content": DESIGNER.format(room=room_summary)}]
    for turn in turns[-12:]:                       # keep the request bounded
        role = "assistant" if turn.get("role") == "assistant" else "user"
        content = str(turn.get("content", ""))[:4000]
        if content:
            messages.append({"role": role, "content": content})

    try:
        response = await _client(settings).chat.completions.create(
            model=settings.vlm_model,
            messages=messages,
            max_tokens=600,
            temperature=0.7,
        )
    except ReadingError:
        raise
    except Exception as exc:
        raise ReadingError(f"Could not answer: {exc}") from exc

    return (response.choices[0].message.content or "").strip()
