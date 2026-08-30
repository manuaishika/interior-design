"""FastAPI surface for the interior-design prototype.

Endpoints:
    GET  /                 the site: hero, studio (upload -> style -> generate),
                           method, gallery
    GET  /api/styles       available style presets
    POST /api/analyze      upload -> structured room JSON (analysis only)
    POST /api/generate     upload + style -> generated image AND room JSON
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import LOCK_PROFILES, Settings, get_settings
from .generation import STYLES, GenerationError
from .models import AnalyzeResponse, GenerateResponse
from .pipeline import analyze_room, prepare_image, run_pipeline
from .segmentation import SegmentationError

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

app = FastAPI(title="AI Interior Design", version="0.2.0")
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
    settings = get_settings()
    return {
        "status": "ok",
        "replicate_configured": bool(settings.replicate_api_token),
        "openai_configured": bool(settings.openai_api_key),
        "models": {
            "segmentation": settings.sam2_model,
            "labeling": settings.vlm_model,
            "generation": settings.inpaint_model,
        },
    }


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


@app.post("/api/generate", response_model=GenerateResponse)
async def generate_endpoint(
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
