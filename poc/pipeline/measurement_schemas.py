"""Pydantic models for the ProCalc-spec measurement-extraction output.

Matches the JSON contract from `ProCalc_AI_Measurement_Response.docx` exactly:

  {
    "document_id": "DOC-YYYYMMDD-XXXXXX",
    "extraction_id": "EXT-xxxxxxxx",
    "status": "success" | "partial" | "failed",
    "source_file": "...",
    "extractions": [ ExtractionField, ... ],
    "errors": [],
    "processing_time_ms": int,
    "model_version": "techuz-procalc-v0.1",
    "scale_detected": { value, source, confidence } | null,
    "cross_validation": { ... } | null     (added in Phase 5)
  }

THREE-STATE COORDINATE CONTRACT (proposal §7 #5):

  - "actual"      => the producer has pixel-precise coordinates from a
                     deterministic detector (YOLO bbox, OCR span bbox,
                     Hough wall vertex, dimension-derived polygon vertex)
  - "placeholder" => the field is known to exist but precise coords are
                     not yet derivable (e.g. floor area before Phase 2's
                     scale calibration lands)
  - "not_detected" => nothing found in the source

The contract matters because the boss's example output uses
`coordinate_status: "actual"`. Emitting "actual" for a placeholder would
deceive downstream consumers. The three-state design lets us be honest
field-by-field about confidence in coordinates.
"""
from __future__ import annotations

import datetime as dt
import uuid
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


# ---------- Enums + small dataclasses ----------

CoordinateStatus = Literal["actual", "placeholder", "not_detected"]
ExtractionStatus = Literal["success", "partial", "failed"]


class ImageDimensions(BaseModel):
    width: int
    height: int


class ScaleDetected(BaseModel):
    """Top-level scale field — separate from the `scale_detected` field
    inside extractions[]. This block tells downstream consumers the
    overall scale calibration the system arrived at."""
    model_config = ConfigDict(frozen=False)

    value: str           # e.g. "1:100"
    source: str          # "scale_bar" | "dimension_chain" | "user_supplied"
    confidence: float = Field(..., ge=0.0, le=1.0)


# ---------- Coordinate variants ----------
#
# Stored as plain dicts in ExtractionField.coordinates so the JSON output
# matches the boss's example structure exactly (where "type" determines
# the shape of the rest of the block). The builder helpers below produce
# correctly-shaped dicts.

def make_polygon_coords(
    points: List[List[int]],
    image_dims: tuple[int, int],
) -> Dict[str, Any]:
    """Polygon for floor area / building envelope."""
    return {
        "type": "polygon",
        "points": [[int(x), int(y)] for x, y in points],
        "coordinate_system": "pixel",
        "image_dimensions": {"width": image_dims[0], "height": image_dims[1]},
    }


def make_bbox_coords(
    bbox: tuple[int, int, int, int],
    image_dims: tuple[int, int],
) -> Dict[str, Any]:
    """Single bounding box."""
    return {
        "type": "bounding_box",
        "bbox": [int(v) for v in bbox],
        "coordinate_system": "pixel",
        "image_dimensions": {"width": image_dims[0], "height": image_dims[1]},
    }


def make_point_coords(
    point: tuple[int, int],
    image_dims: tuple[int, int],
) -> Dict[str, Any]:
    """Single point (e.g. window centre)."""
    return {
        "type": "point",
        "position": [int(point[0]), int(point[1])],
        "coordinate_system": "pixel",
        "image_dimensions": {"width": image_dims[0], "height": image_dims[1]},
    }


def make_multi_region_coords(
    regions: List[Dict[str, Any]],
    image_dims: tuple[int, int],
) -> Dict[str, Any]:
    """Multi-region for counts (e.g. 3 bathrooms as 3 bboxes).

    Each region dict is shaped {label, type, bbox|position}.
    """
    return {
        "type": "multi_region",
        "regions": regions,
        "coordinate_system": "pixel",
        "image_dimensions": {"width": image_dims[0], "height": image_dims[1]},
    }


# ---------- Top-level models ----------

class ExtractionField(BaseModel):
    """One row in the `extractions[]` array of the client output."""
    model_config = ConfigDict(frozen=False)

    field_key: str = Field(..., description="snake_case identifier, e.g. 'total_floor_area'")
    field_label: str = Field(..., description="Human-readable label, e.g. 'Total Floor Area'")
    value: Any = Field(..., description="Scalar (int/float/string), or list/dict for compound")
    unit: str = Field(..., description="'m2', 'm', 'count', 'ratio', or '' for unitless")
    source_page: int = Field(..., ge=1)
    coordinates: Optional[Dict[str, Any]] = None
    coordinate_status: CoordinateStatus
    confidence: float = Field(..., ge=0.0, le=1.0)
    notes: str = ""


class MeasurementOutput(BaseModel):
    """Top-level extraction output. Serialises to the client JSON schema."""
    model_config = ConfigDict(frozen=False)

    document_id: str
    extraction_id: str
    status: ExtractionStatus
    source_file: str
    extractions: List[ExtractionField] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    processing_time_ms: int = 0
    model_version: str = "techuz-procalc-v0.1"
    scale_detected: Optional[ScaleDetected] = None
    # Phase 5 — schedule cross-validation; populated when page 6 schedule
    # was read and diffed against the measurement output.
    cross_validation: Optional[Dict[str, Any]] = None


# ---------- ID factories ----------

def new_document_id() -> str:
    """DOC-YYYYMMDD-XXXXXX — date prefix + 6-char hex suffix."""
    today = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d")
    suffix = uuid.uuid4().hex[:6].upper()
    return f"DOC-{today}-{suffix}"


def new_extraction_id() -> str:
    """EXT-xxxxxxxx — 8-char hex."""
    return f"EXT-{uuid.uuid4().hex[:8]}"


# ---------- Placeholder factory for "not yet implemented" fields ----------

def placeholder_field(
    *,
    field_key: str,
    field_label: str,
    unit: str,
    source_page: int,
    reason: str,
) -> ExtractionField:
    """Emit a not-yet-derived field with `coordinate_status="placeholder"`.
    Used in Phase 1 for fields that need Phase 2's scale calibration."""
    return ExtractionField(
        field_key=field_key,
        field_label=field_label,
        value=None,
        unit=unit,
        source_page=source_page,
        coordinates=None,
        coordinate_status="placeholder",
        confidence=0.0,
        notes=f"Placeholder — {reason}",
    )


def not_detected_field(
    *,
    field_key: str,
    field_label: str,
    unit: str,
    source_page: int,
    reason: str = "Not detected in source",
) -> ExtractionField:
    """Emit a 'we looked but found nothing' field."""
    return ExtractionField(
        field_key=field_key,
        field_label=field_label,
        value=0 if unit == "count" else None,
        unit=unit,
        source_page=source_page,
        coordinates=None,
        coordinate_status="not_detected",
        confidence=0.0,
        notes=reason,
    )
