"""FastAPI surface for the interior-design prototype.

Endpoints:
    GET  /                 the site: hero, studio (upload -> style -> generate),
                           method, gallery
    GET  /api/styles       available style presets
    POST /api/analyze      upload -> structured room JSON (analysis only)
    POST /api/generate     upload + style -> generated image AND room JSON
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import auth
from .config import LOCK_PROFILES, Settings, get_settings, resolve_backend
from .generation import STYLES, GenerationError
from .models import AnalyzeResponse, GenerateResponse
from .pipeline import analyze_room, prepare_image, run_pipeline
from .reading import NotARoomError, ReadingError, discuss, read_room
from .segmentation import SegmentationError

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

app = FastAPI(title="AI Interior Design", version="0.2.0")

# The published site and the engine live on different hosts: the page is a
# stable URL the studio owns, the engine is wherever a GPU happens to be that
# day. The browser blocks that call unless the engine says it is allowed.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://.*",
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _parse_ids(raw: str | None) -> set[str]:
    """Parse a comma-separated mask_id list from a form field."""
    if not raw:
        return set()
    return {part.strip() for part in raw.split(",") if part.strip()}


async def _read_upload(photo: UploadFile, settings: Settings) -> bytes:
    if photo.content_type and not photo.content_type.startswith("image/"):
        raise HTTPException(400, f"Expected an image, got {photo.content_type}")
    data = await photo.read()
    if not data:
        raise HTTPException(400, "Uploaded file is empty")
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(
            413,
            f"Image is {len(data) / 1e6:.1f} MB; limit is "
            f"{settings.max_upload_bytes / 1e6:.0f} MB",
        )
    return data


@app.get("/", include_in_schema=False)
async def index():
    """The site. The studio section posts back to the endpoints below."""
    page = STATIC_DIR / "showcase.html"
    if not page.is_file():
        raise HTTPException(404, "UI not installed")
    return FileResponse(page)


@app.get("/api/health")
async def health():
    """What this deployment can actually do, right now.

    The page reads `can_read` and `can_draw` to decide what to offer, and
    `engine` to say so out loud. Guessing from key names would go wrong the
    moment a second engine existed, so the resolution happens in one place
    and is reported rather than inferred.
    """
    settings = get_settings()
    engine = resolve_backend(settings)

    if engine == "free":
        can_read = can_draw = bool(settings.google_api_key)
        models = {"reading": settings.google_vision_model,
                  "drawing": settings.google_image_model}
    elif engine == "local":
        can_read = can_draw = True
        models = {"reading": settings.local_seg_model,
                  "drawing": settings.local_inpaint_model}
    else:
        can_read = bool(settings.openai_api_key)
        can_draw = bool(settings.replicate_api_token)
        models = {"segmentation": settings.sam2_model,
                  "labeling": settings.vlm_model,
                  "generation": settings.inpaint_model}

    return {
        "status": "ok",
        "engine": engine,
        # The free path has no mask: the lock is asked for, not enforced.
        "locks_are_enforced": engine != "free",
        "replicate_configured": bool(settings.replicate_api_token),
        "openai_configured": bool(settings.openai_api_key),
        "google_configured": bool(settings.google_api_key),
        "can_read": can_read,
        "can_draw": can_draw,
        "models": models,
    }


@app.get("/api/session")
async def session_state(request: Request):
    """Whether there is a door, and whether you are through it."""
    settings = get_settings()
    return {"required": auth.required(settings),
            "signed_in": auth.signed_in(request, settings)}


@app.post("/api/login")
async def login(request: Request, response: Response, code: str = Form(...)):
    settings = get_settings()
    if not auth.required(settings):
        return {"signed_in": True, "required": False}
    if not auth.matches(code, settings):
        raise HTTPException(401, "That code is not right.")
    token, max_age = auth.issue(settings)
    auth.set_cookie(response, token, max_age, secure=auth.over_https(request))
    return {"signed_in": True, "required": True}


@app.post("/api/logout")
async def logout(response: Response):
    auth.clear_cookie(response)
    return {"signed_in": False}


@app.get("/api/styles")
async def list_styles():
    return {
        "styles": [{"id": k, "description": v} for k, v in STYLES.items()],
        "lock_profiles": LOCK_PROFILES,
    }


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze_endpoint(
    photo: UploadFile = File(...),
    profile: str | None = Form(None),
    keep_mask_ids: str | None = Form(None),
    replace_mask_ids: str | None = Form(None),
):
    """Run the analysis step alone — useful for inspecting detection."""
    settings = get_settings()
    data = await _read_upload(photo, settings)
    image = prepare_image(data, settings)
    try:
        analysis, _ = await analyze_room(
            image,
            settings,
            profile=profile,
            keep_mask_ids=_parse_ids(keep_mask_ids),
            replace_mask_ids=_parse_ids(replace_mask_ids),
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except SegmentationError as exc:
        raise HTTPException(502, f"Segmentation failed: {exc}") from exc
    except RuntimeError as exc:
        raise HTTPException(502, str(exc)) from exc
    return AnalyzeResponse(analysis=analysis)


@app.post("/api/read")
async def read_endpoint(
    request: Request,
    photo: UploadFile = File(...),
    room_type: str = Form("room"),
):
    """Look at a room and say what is in it, plus three directions.

    Vision only — no GPU — so this works on any ordinary host.
    """
    settings = get_settings()
    auth.guard(request, settings)
    data = await _read_upload(photo, settings)
    try:
        return await read_room(data, room_type, settings)
    except NotARoomError as exc:
        # 422, not 502: nothing is broken, the picture is just not a room.
        raise HTTPException(422, str(exc)) from exc
    except ReadingError as exc:
        raise HTTPException(502, str(exc)) from exc


@app.post("/api/chat")
async def chat_endpoint(
    request: Request,
    room_summary: str = Form(...),
    turns: str = Form(...),
):
    """Continue the conversation about a room already read."""
    settings = get_settings()
    auth.guard(request, settings)
    try:
        parsed = json.loads(turns)
        if not isinstance(parsed, list):
            raise ValueError("turns must be a list")
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(400, f"Bad turns: {exc}") from exc

    try:
        return {"reply": await discuss(room_summary, parsed, settings)}
    except ReadingError as exc:
        raise HTTPException(502, str(exc)) from exc


@app.post("/api/generate", response_model=GenerateResponse)
async def generate_endpoint(
    request: Request,
    photo: UploadFile = File(...),
    style: str = Form(...),
    extra_prompt: str = Form(""),
    seed: int | None = Form(None),
    variants: int | None = Form(None),
    profile: str | None = Form(None),
    keep_mask_ids: str | None = Form(None),
    replace_mask_ids: str | None = Form(None),
):
    """Upload -> analyze -> masked generation.

    Returns N design options plus the structured JSON they were built from.
    """
    settings = get_settings()
    auth.guard(request, settings)
    data = await _read_upload(photo, settings)
    try:
        analysis, generations = await run_pipeline(
            data,
            style,
            settings,
            extra_prompt=extra_prompt,
            seed=seed,
            variants=variants,
            profile=profile,
            keep_mask_ids=_parse_ids(keep_mask_ids),
            replace_mask_ids=_parse_ids(replace_mask_ids),
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except SegmentationError as exc:
        raise HTTPException(502, f"Segmentation failed: {exc}") from exc
    except GenerationError as exc:
        raise HTTPException(502, f"Generation failed: {exc}") from exc
    except RuntimeError as exc:
        raise HTTPException(502, str(exc)) from exc
    return GenerateResponse(analysis=analysis, generations=generations)
