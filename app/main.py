"""FastAPI surface for the interior-design prototype.

Endpoints:
    GET  /                 the site: hero, studio (upload -> style -> generate),
                           method, gallery
    GET  /api/styles       available style presets
    POST /api/analyze      upload -> structured room JSON (analysis only)
    POST /api/generate     upload + style -> generated image AND room JSON
"""

from __future__ import annotations

import hmac
import json
import logging
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import auth, google_login, store
from .openai_images import image_models as openai_images_chain
from urllib.parse import quote

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


@app.on_event("startup")
async def _open_the_books() -> None:
    """Tables are created on boot rather than by a migration step.

    There is one schema and no history to migrate, and a deploy that needs a
    second manual command is a deploy somebody eventually forgets to finish.
    When the schema starts changing under real users, this is the line that
    becomes Alembic.
    """
    settings = get_settings()
    store.configure(settings.database_url)
    await store.create_tables()


@app.on_event("shutdown")
async def _close_the_books() -> None:
    await store.dispose()


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
    elif engine == "openai":
        can_read = can_draw = bool(settings.openai_api_key)
        models = {"reading": settings.vlm_model,
                  "structure": settings.vlm_model,
                  "drawing": openai_images_chain(settings)[0]}
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
        # Every other engine paints the openings out before generating.
        "locks_are_enforced": engine != "free",
        "replicate_configured": bool(settings.replicate_api_token),
        "openai_configured": bool(settings.openai_api_key),
        "google_configured": bool(settings.google_api_key),
        "can_read": can_read,
        "can_draw": can_draw,
        "models": models,
    }


def _land(request: Request, response: Response, settings: Settings,
          user_id: int) -> None:
    token, max_age = auth.issue(settings, user_id)
    auth.set_cookie(response, token, max_age, secure=auth.over_https(request))


def _card(user) -> dict:
    return {"email": user.email, "name": user.name,
            "google": bool(user.google_id)}


@app.get("/api/session")
async def session_state(request: Request):
    """Who is signed in, and what this deployment lets a stranger do."""
    settings = get_settings()
    who = auth.current_user_id(request, settings)
    account = None
    if who is not None:
        async with store.session() as db:
            user = await store.by_id(db, who)
            account = _card(user) if user else None
    return {
        "required": auth.required(settings),
        "signed_in": auth.signed_in(request, settings),
        "account": account,
        "google_available": google_login.configured(settings),
    }


@app.post("/api/signup")
async def signup(request: Request, response: Response,
                 email: str = Form(...), password: str = Form(...),
                 name: str = Form("")):
    settings = get_settings()
    if len(password) < 8:
        raise HTTPException(400, "Use at least 8 characters.")
    if "@" not in email or "." not in email.split("@")[-1]:
        raise HTTPException(400, "That does not look like an email address.")
    async with store.session() as db:
        try:
            user = await store.sign_up(db, email, password, name)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        card = _card(user)
        user_id = user.id
    _land(request, response, settings, user_id)
    return {"signed_in": True, "account": card}


@app.post("/api/login")
async def login(request: Request, response: Response,
                email: str = Form(""), password: str = Form(""),
                code: str = Form("")):
    """Two doors, one endpoint.

    An email and password signs you into an account. The access code, where a
    deployment sets one, only opens the studio — it makes you admitted but
    nobody in particular, which is why it cannot save anything.
    """
    settings = get_settings()

    if email:
        async with store.session() as db:
            user = await store.sign_in(db, email, password)
            if user is None:
                raise HTTPException(401, "That email and password do not match.")
            card, user_id = _card(user), user.id
        _land(request, response, settings, user_id)
        return {"signed_in": True, "account": card}

    if not auth.required(settings):
        return {"signed_in": True, "account": None}
    if not auth.matches(code, settings):
        raise HTTPException(401, "That code is not right.")
    token, max_age = auth.issue(settings)
    auth.set_cookie(response, token, max_age, secure=auth.over_https(request))
    return {"signed_in": True, "account": None}


@app.post("/api/logout")
async def logout(response: Response):
    auth.clear_cookie(response)
    return {"signed_in": False}


# --- Continue with Google ----------------------------------------------

def _redirect_uri(request: Request) -> str:
    """Must match, character for character, what is registered with Google."""
    scheme = "https" if auth.over_https(request) else request.url.scheme
    return f"{scheme}://{request.url.netloc}/api/auth/google/callback"


@app.get("/api/auth/google")
async def google_start(request: Request):
    settings = get_settings()
    if not google_login.configured(settings):
        raise HTTPException(503, "Google sign-in is not set up on this server.")

    state = google_login.new_state()
    response = RedirectResponse(
        google_login.start_url(settings, _redirect_uri(request), state))
    # The state travels to Google in the URL and is kept here to be compared on
    # the way back. That is what stops somebody else's sign-in being replayed
    # at you to land you in their account.
    response.set_cookie(google_login.STATE_COOKIE, state, max_age=600,
                        httponly=True, samesite="lax",
                        secure=auth.over_https(request), path="/")
    return response


@app.get("/api/auth/google/callback")
async def google_callback(request: Request, code: str = "", state: str = "",
                          error: str = ""):
    settings = get_settings()
    expected = request.cookies.get(google_login.STATE_COOKIE)

    def back(message: str = "") -> RedirectResponse:
        # Always land back on the site. A person who cancelled a sign-in should
        # see the studio again, not a JSON error page.
        target = "/#/studio"
        if message:
            target = "/?auth_error=" + quote(message) + "#/studio"
        out = RedirectResponse(target, status_code=303)
        out.delete_cookie(google_login.STATE_COOKIE, path="/")
        return out

    if error:
        return back("Google sign-in was cancelled.")
    if not code or not state or not expected:
        return back("That sign-in link had expired. Try again.")
    if not hmac.compare_digest(state, expected):
        return back("That sign-in link did not match. Try again.")

    try:
        profile = await google_login.exchange(settings, code,
                                              _redirect_uri(request))
    except google_login.GoogleLoginError as exc:
        log.warning("Google sign-in failed: %s", exc)
        return back(str(exc))

    async with store.session() as db:
        user = await store.from_google(db, profile["google_id"],
                                       profile["email"], profile["name"])
        user_id = user.id

    out = back()
    _land(request, out, settings, user_id)
    return out


# --- designs somebody kept ---------------------------------------------

@app.get("/api/designs")
async def list_designs(request: Request):
    settings = get_settings()
    who = auth.require_user(request, settings)
    async with store.session() as db:
        rows = await store.designs_for(db, who)
        return {"designs": [
            {"id": d.id, "title": d.title, "room": d.room, "style": d.style,
             "note": d.note, "created_at": d.created_at.isoformat(),
             "url": f"/api/designs/{d.id}/image"}
            for d in rows
        ]}


@app.post("/api/designs")
async def keep_design(request: Request, image: UploadFile = File(...),
                      title: str = Form(""), room: str = Form(""),
                      style: str = Form(""), note: str = Form("")):
    settings = get_settings()
    who = auth.require_user(request, settings)
    data = await _read_upload(image, settings)
    async with store.session() as db:
        design = await store.save_design(db, who, data, title=title, room=room,
                                         style=style, note=note)
        design_id = design.id
    return {"id": design_id, "url": f"/api/designs/{design_id}/image"}


@app.get("/api/designs/{design_id}/image")
async def design_image(request: Request, design_id: int):
    settings = get_settings()
    who = auth.require_user(request, settings)
    async with store.session() as db:
        design = await store.design_for(db, who, design_id)
        if design is None:
            # 404 rather than 403: somebody else's id should not be confirmed
            # to exist merely by not being yours.
            raise HTTPException(404, "No such design.")
        return Response(design.image, media_type="image/png")


@app.delete("/api/designs/{design_id}")
async def drop_design(request: Request, design_id: int):
    settings = get_settings()
    who = auth.require_user(request, settings)
    async with store.session() as db:
        if not await store.delete_design(db, who, design_id):
            raise HTTPException(404, "No such design.")
    return {"deleted": design_id}


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
    design: UploadFile | None = File(None),
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

    drawn = await _read_upload(design, settings) if design is not None else None

    try:
        return {"reply": await discuss(room_summary, parsed, settings,
                                       design=drawn)}
    except ReadingError as exc:
        raise HTTPException(502, str(exc)) from exc


@app.post("/api/generate", response_model=GenerateResponse)
async def generate_endpoint(
    request: Request,
    photo: UploadFile = File(...),
    style: str = Form(...),
    room_type: str = Form(""),
    contents: str = Form(""),
    keep: str = Form(""),
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
            room=room_type,
            contents=contents,
            keep=keep,
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
