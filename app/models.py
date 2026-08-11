"""Pydantic schemas for the room analysis payload.

`RoomObject` is the contract the spec asks for: {label, mask_id, bounding_box,
locked}. Everything beyond those four fields is debugging metadata and is safe
to ignore downstream.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Axis-aligned box in pixel coordinates of the analyzed (resized) image."""

    x: int
    y: int
    width: int
    height: int

    @property
    def area(self) -> int:
        return self.width * self.height

    def as_xyxy(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.x + self.width, self.y + self.height)


class RoomObject(BaseModel):
    # --- the four fields the pipeline contract requires ---------------------
    label: str = Field(
        ...,
        description=(
            "Human-readable label. For furniture this is the specific item "
            "name ('sofa', 'coffee table'); otherwise the category itself "
            "('door', 'window', 'wall', 'floor', 'walkway')."
        ),
    )
    mask_id: str
    bounding_box: BoundingBox
    locked: bool

    # --- debugging metadata -------------------------------------------------
    category: str = Field(..., description="One of config.CATEGORIES.")
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    area_px: int = 0
    area_frac: float = 0.0
    lock_source: Literal["policy", "user_override"] = "policy"
    notes: str = ""


class RoomAnalysis(BaseModel):
    image_width: int
    image_height: int
    objects: list[RoomObject]

    # Debugging counters — how many masks SAM2 produced vs. how many survived
    # filtering and got labeled.
    masks_returned: int = 0
    masks_labeled: int = 0

    @property
    def locked_objects(self) -> list[RoomObject]:
        return [o for o in self.objects if o.locked]


class GenerationResult(BaseModel):
    image_base64: str = Field(..., description="PNG data, base64-encoded.")
    image_url: str | None = Field(
        None, description="Provider-hosted URL, when the provider returns one."
    )
    inpaint_mask_base64: str = Field(
        ...,
        description=(
            "The mask actually sent to the generator, for debugging. White = "
            "regenerated, black = preserved (before any provider inversion)."
        ),
    )
    prompt: str


class AnalyzeResponse(BaseModel):
    analysis: RoomAnalysis


class GenerateResponse(BaseModel):
    """Both halves of the deliverable: the image and the structured JSON."""

    analysis: RoomAnalysis
    generation: GenerationResult
