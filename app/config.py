"""Configuration for the room-analysis + generation pipeline.

Everything that is likely to drift (hosted model slugs, thresholds, lock policy)
lives here so it can be tuned without touching pipeline code.
"""

from __future__ import annotations

import os
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Label taxonomy
# ---------------------------------------------------------------------------

# The closed set of categories the vision-language model is allowed to return.
# "furniture" additionally carries a free-form `name` ("sofa", "coffee table").
#
# "clutter" is deliberately separate from "furniture": real room photos are full
# of laundry, carrier bags, water bottles, loose cables and stacked paperwork. A
# renovation render is supposed to make those disappear, so they need a category
# of their own rather than being swept into "other" (which is preserved).
CATEGORIES = (
    "furniture",
    "clutter",
    "door",
    "window",
    "wall",
    "floor",
    "walkway",
    "other",
)

# Lock profiles decide which categories survive generation untouched.
#
# Both profiles pin the four categories from the spec:
#   locked   -> door, window, walkway
#   unlocked -> furniture, floor ("open floor space")
#
# They differ only in how much benefit of the doubt they give a region:
#
#   "renovate"  full redesign. Strip everything that is not structure. An
#               unidentified region gets regenerated, because leaving frozen
#               islands scattered through a full renovation looks worse than
#               re-imagining them.
#   "restyle"   conservative. An unidentified region is preserved. Use when the
#               room should stay recognisably itself.
#
# `wall` is unlocked in both: repainting is a core interior-design edit, and
# locking walls would freeze the largest surface in most rooms.
LOCK_PROFILES: dict[str, dict[str, bool]] = {
    "renovate": {
        "furniture": False,
        "clutter": False,
        "door": True,
        "window": True,
        "wall": False,
        "floor": False,
        "walkway": True,
        "other": False,
    },
    "restyle": {
        "furniture": False,
        "clutter": False,
        "door": True,
        "window": True,
        "wall": False,
        "floor": False,
        "walkway": True,
        "other": True,
    },
}

DEFAULT_PROFILE = "renovate"

# Backwards-compatible alias for the active default profile.
LOCK_POLICY = LOCK_PROFILES[DEFAULT_PROFILE]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- credentials -------------------------------------------------------
    replicate_api_token: str = ""
    openai_api_key: str = ""
    # One key, no card, both halves of the job. aistudio.google.com/apikey
    google_api_key: str = ""

    # --- backend -----------------------------------------------------------
    # "hosted" calls Replicate + OpenAI and needs two paid accounts.
    # "free"   calls Google AI Studio for both halves on a no-card key.
    # "local"  runs open-weight models in-process and needs a GPU.
    #
    # Left unset this resolves itself from whichever keys are present, so a
    # deploy needs one environment variable rather than two that must agree.
    backend: str = ""

    # --- the free path -----------------------------------------------------
    google_vision_model: str = "gemini-2.5-flash"
    google_image_model: str = "gemini-2.5-flash-image"

    # --- the door ----------------------------------------------------------
    # Set a code and the studio asks for it before spending anything. Leave it
    # blank and there is no door, which is what local development wants.
    studio_access_code: str = ""
    # Set this and sessions survive a restart. Leave it and they do not, which
    # is the safe direction to fail.
    session_secret: str = ""

    # --- keyless (local) models ---------------------------------------------
    # One ADE20K segmentation model replaces both SAM2 and the VLM: its class
    # list already names walls, floors, doors, windows and furnishings.
    local_seg_model: str = "nvidia/segformer-b4-finetuned-ade-512-512"
    local_inpaint_model: str = "runwayml/stable-diffusion-inpainting"
    # SD 1.x inpainting is trained at 512; generating larger produces mush.
    local_generation_size: int = 512

    # --- hosted models -----------------------------------------------------
    # SAM2 in automatic-mask-generation mode. Unpinned so Replicate resolves the
    # latest version; pin to "meta/sam-2:<version-sha>" for reproducibility.
    sam2_model: str = "meta/sam-2"

    # Vision-language model used to label each mask. Note: the original
    # "gpt-4-vision-preview" ("GPT-4V") checkpoint has been retired by OpenAI;
    # gpt-4o is its current vision-capable successor and speaks the same
    # image_url message format.
    vlm_model: str = "gpt-4o"

    # Mask-conditioned (inpainting) image generation.
    inpaint_model: str = "stability-ai/stable-diffusion-inpainting"

    # --- SAM2 tuning -------------------------------------------------------
    sam2_points_per_side: int = 32
    sam2_pred_iou_thresh: float = 0.88
    sam2_stability_score_thresh: float = 0.95

    # --- mask filtering ----------------------------------------------------
    # SAM2's automatic mode happily returns 100+ masks on a busy room photo, and
    # every one of them would cost a VLM call. Filter hard before labeling.
    min_mask_area_frac: float = 0.004  # drop specks below 0.4% of the image
    max_mask_area_frac: float = 0.95   # drop the "whole image" mask
    max_masks: int = 24                # label at most N masks, largest first
    dedupe_iou_thresh: float = 0.80    # collapse near-duplicate masks

    # --- labeling ----------------------------------------------------------
    label_concurrency: int = 6
    crop_padding_frac: float = 0.12    # context padding around each crop

    # --- inpainting mask ---------------------------------------------------
    # Grow locked regions by this many pixels before generation. Diffusion
    # models bleed across mask boundaries, so a small cushion keeps door frames
    # and window reveals genuinely intact.
    locked_dilation_px: int = 12
    # Most Stable-Diffusion inpainting endpoints treat WHITE as "repaint this".
    # Set to True for endpoints using the opposite convention.
    invert_inpaint_mask: bool = False

    # --- generation --------------------------------------------------------
    # How many design options to render per request. Each is a separate call to
    # the generator, run concurrently, differing by seed.
    default_variants: int = 2
    max_variants: int = 4
    generation_steps: int = 30
    generation_guidance: float = 7.5

    # --- misc --------------------------------------------------------------
    request_timeout_s: float = 180.0
    max_upload_bytes: int = 20 * 1024 * 1024
    # Longest edge the uploaded photo is resized to before analysis.
    max_image_edge: int = 1536


def resolve_backend(settings: Settings) -> str:
    """Which engine actually runs, given the keys that exist.

    An explicit BACKEND always wins. Otherwise: the paid pair if it is there,
    then the free key, then the paid path anyway so the error message names
    the key that is missing rather than silently doing nothing.

    Auto-resolving matters because the commonest way to deploy this wrong is
    to paste a key and forget the flag that turns its engine on.
    """
    chosen = (settings.backend or "").strip().lower()
    if chosen:
        return chosen
    if settings.openai_api_key and settings.replicate_api_token:
        return "hosted"
    if settings.google_api_key:
        return "free"
    return "hosted"


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    # The SDKs read these from the environment rather than from us.
    if settings.replicate_api_token:
        os.environ.setdefault("REPLICATE_API_TOKEN", settings.replicate_api_token)
    return settings


def resolve_profile(profile: str | None) -> dict[str, bool]:
    """Look up a lock profile by name, falling back to the default."""
    return LOCK_PROFILES.get((profile or DEFAULT_PROFILE).strip().lower(),
                             LOCK_PROFILES[DEFAULT_PROFILE])


def is_locked(category: str, profile: str | None = None) -> bool:
    """Lock decision for a category, defaulting to locked when unknown.

    An unknown category is always locked regardless of profile: the profiles
    only speak for categories they actually list.
    """
    return resolve_profile(profile).get(category, True)
