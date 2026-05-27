"""JSON contract per ProCalc proposal §2.

Every extracted field has value, unit, source_page, overlay coordinates,
and one of three detection states (§7 — collapsing these is a named failure mode).
"""
from __future__ import annotations

from enum import Enum
from typing import Any, List, Optional

from pydantic import BaseModel, Field


class DetectionState(str, Enum):
    DETECTED_ACTUAL = "detected_actual"
    DETECTED_PLACEHOLDER = "detected_placeholder"
    NOT_DETECTED = "not_detected"


class BoundingBox(BaseModel):
    x: float = Field(..., description="Left edge in image pixels")
    y: float = Field(..., description="Top edge in image pixels")
    w: float = Field(..., description="Width in image pixels")
    h: float = Field(..., description="Height in image pixels")
    label: Optional[str] = Field(None, description="Optional per-box label, e.g. 'Door 3'")


class ExtractedField(BaseModel):
    variable: str = Field(..., description="ProCalc calculator field key, e.g. 'ground_floor_area_m2'")
    display_name: str = Field(..., description="Human-readable name for the UI")
    value: Optional[Any] = None
    unit: Optional[str] = None
    source_page: Optional[int] = Field(None, description="1-indexed page in the original PDF")
    detection_state: DetectionState
    overlay_coordinates: List[BoundingBox] = Field(default_factory=list)
    confidence_note: Optional[str] = Field(
        None, description="Why placeholder vs actual; surfaced to the reviewer"
    )


class ApprovalState(str, Enum):
    PENDING_REVIEW = "pending_review"
    SIGHTED = "sighted"
    CORRECTED = "corrected"
    REJECTED = "rejected"


class ReviewableField(BaseModel):
    """Wraps an ExtractedField with the per-field approval state machine (§3, §7)."""
    extracted: ExtractedField
    approval: ApprovalState = ApprovalState.PENDING_REVIEW
    corrected_value: Optional[Any] = None
    reviewer_note: Optional[str] = None


class ExtractionResult(BaseModel):
    fields: List[ExtractedField]
    prompt_version: str
    model: str
    source_page: int
    page_image_width: int
    page_image_height: int
    raw_response: Optional[str] = None
    error: Optional[str] = None
