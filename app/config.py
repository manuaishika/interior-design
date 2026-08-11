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
CATEGORIES = (
    "furniture",
    "door",
    "window",
    "wall",
    "floor",
    "walkway",
    "other",
)

# Which categories are structurally locked, i.e. must survive generation
# untouched.
#
# The spec pins four of these explicitly:
#   locked   -> door, window, walkway
#   unlocked -> furniture, floor ("open floor space")
#
# `wall` was not specified either way. It defaults to UNLOCKED here because
# repainting / re-wallpapering walls is a core interior-design edit, and locking
# them would prevent the generator from restyling the largest surface in most
# rooms. Flip it to True if you would rather freeze room geometry completely.
#
# `other` (the model's escape hatch when it cannot confidently classify a
# region) defaults to LOCKED — an unrecognised region is safer preserved than
# regenerated.
LOCK_POLICY: dict[str, bool] = {
    "furniture": False,
    "door": True,
    "window": True,
    "wall": False,
    "floor": False,
    "walkway": True,
    "other": True,
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- credentials -------------------------------------------------------
    replicate_api_token: str = ""
    openai_api_key: str = ""

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
    generation_steps: int = 30
    generation_guidance: float = 7.5

    # --- misc --------------------------------------------------------------
    request_timeout_s: float = 180.0
    max_upload_bytes: int = 20 * 1024 * 1024
    # Longest edge the uploaded photo is resized to before analysis.
    max_image_edge: int = 1536


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    # The SDKs read these from the environment rather than from us.
    if settings.replicate_api_token:
        os.environ.setdefault("REPLICATE_API_TOKEN", settings.replicate_api_token)
    return settings


def is_locked(category: str) -> bool:
    """Lock decision for a category, defaulting to locked when unknown."""
    return LOCK_POLICY.get(category, True)
