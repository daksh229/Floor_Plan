"""Phase 5 — schedule-page cross-validation.

The variant_e fixture's page 6 contains three tabulated schedules:
  - Window schedule (W1..Wn: width, height, type, glazing)
  - Door schedule (D1, SD1, ...: width, height, type, glass yes/no)
  - Area schedule (per-room width × depth = area, wet flag, totals)

This module:
  1. Sends the schedule page to Claude vision with a strict-JSON prompt
  2. Parses the response into a `ScheduleReadResult`
  3. Cross-validates against the `MeasurementOutput` from page 5 — for
     every comparable metric (total_floor_area, wet_area_total,
     bathroom_count, glass_doors_windows), reports either AGREE or
     DISAGREE with the absolute and percentage difference

The output `cross_validation` block sits on the CombinedResult and gets
serialised into the client schema. The intended use is observability:
the estimator sees that the two sources of truth (architectural plan +
schedule table) agree, and trust in the extraction goes up.

When they disagree, that's the most useful signal of all — the
estimator knows immediately which fields need a sight-check before
hitting Calculate/Save.
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from PIL import Image


DEFAULT_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")
DEFAULT_TIMEOUT_S = 90
DEFAULT_MAX_TOKENS = 2500
INPUT_MAX_LONG_EDGE = 2400


# ============================================================
# Data classes
# ============================================================

@dataclass
class ScheduleArea:
    room: str
    width_mm: Optional[int]
    depth_mm: Optional[int]
    area_m2: Optional[float]
    is_wet: bool


@dataclass
class ScheduleOpening:
    """Row from window or door schedule."""
    kind: str            # "window" | "door"
    label: str           # e.g. "W1", "SD1"
    location: str        # e.g. "Living"
    width_mm: Optional[int]
    height_mm: Optional[int]
    type_: str           # "Awning", "Sliding", "Hinged", ...
    is_glass: bool       # for windows always True; for doors per schedule


@dataclass
class ScheduleReadResult:
    """Everything we extracted from the schedule page."""
    total_floor_area_m2: Optional[float]
    wet_area_total_m2: Optional[float]
    areas: list[ScheduleArea] = field(default_factory=list)
    openings: list[ScheduleOpening] = field(default_factory=list)
    raw_response: str = ""
    error: Optional[str] = None

    @property
    def window_count(self) -> int:
        return sum(1 for o in self.openings if o.kind == "window")

    @property
    def door_count(self) -> int:
        return sum(1 for o in self.openings if o.kind == "door" and not o.is_glass)

    @property
    def glass_count(self) -> int:
        # Per the client schema's `glass_doors_windows`: all windows
        # (glass by definition) + glass doors.
        return sum(1 for o in self.openings if o.is_glass)

    @property
    def bathroom_count(self) -> int:
        # Anything labelled Bath/Ensuite/WC in the area schedule.
        # NOT Laundry — Laundry is wet but isn't a "bathroom" by the
        # strict convention we use in measurement_extractor.
        pat = re.compile(r"\b(bathroom|ensuite|wc|toilet|powder|bath)\b", re.IGNORECASE)
        return sum(1 for a in self.areas
                   if a.room and pat.search(a.room) and "laundry" not in a.room.lower())


@dataclass
class CrossValidationFinding:
    """One field's comparison. agree=True iff within tolerance_pct."""
    field_key: str
    measurement_value: Any
    schedule_value: Any
    diff_abs: Optional[float]
    diff_pct: Optional[float]
    tolerance_pct: float
    agree: bool
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "field_key": self.field_key,
            "measurement_value": self.measurement_value,
            "schedule_value": self.schedule_value,
            "diff_abs": self.diff_abs,
            "diff_pct": self.diff_pct,
            "tolerance_pct": self.tolerance_pct,
            "agree": self.agree,
            "note": self.note,
        }


# ============================================================
# Claude vision schedule reader
# ============================================================

_SYSTEM_PROMPT = """You are a construction-drawing schedule reader.

You will see ONE page from a residential plan set. The page contains
TABULATED schedules — typically a Window Schedule, a Door Schedule, and
an Area Schedule. Read every row of every table and return strict JSON.

Tables you may see (any subset can appear; ignore other content):

  WINDOW SCHEDULE — columns: ID, ROOM/LOCATION, WIDTH, HEIGHT, TYPE, GLAZING
  DOOR SCHEDULE   — columns: ID, LOCATION, WIDTH, HEIGHT, TYPE, GLASS (Yes/No)
  AREA SCHEDULE   — columns: ROOM, WIDTH MM, DEPTH MM, AREA M2, WET AREA (Yes/No)

Also look for total summary lines like:
  TOTAL ENCLOSED FLOOR AREA: 82.96 m2
  WET AREA TOTAL: 8.64 m2

Return ONLY this JSON object, no prose around it:

{
  "total_floor_area_m2": <number or null>,
  "wet_area_total_m2":   <number or null>,
  "areas": [
    {"room": "<name>", "width_mm": <int or null>, "depth_mm": <int or null>,
     "area_m2": <number or null>, "is_wet": <true|false>}, ...
  ],
  "openings": [
    {"kind": "window", "label": "W1", "location": "<room>",
     "width_mm": <int or null>, "height_mm": <int or null>,
     "type": "<text>", "is_glass": true}, ...
    {"kind": "door",   "label": "D1", "location": "<room>",
     "width_mm": <int or null>, "height_mm": <int or null>,
     "type": "<text>", "is_glass": <true|false>}, ...
  ]
}

Never invent rows. If a table is missing, leave its array empty.
If a value is illegible, return null for that field.
"""


def _downsample(img: Image.Image, max_long_edge: int = INPUT_MAX_LONG_EDGE
                ) -> tuple[Image.Image, float]:
    le = max(img.width, img.height)
    if le <= max_long_edge:
        return img, 1.0
    r = max_long_edge / le
    return img.resize((int(img.width * r), int(img.height * r)), Image.LANCZOS), 1.0 / r


def _pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=88)
    return base64.standard_b64encode(buf.getvalue()).decode("ascii")


def _parse_json(text: str) -> dict:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return json.loads(fenced.group(1))
    first, last = text.find("{"), text.rfind("}")
    if first == -1 or last == -1:
        raise ValueError("No JSON in schedule response")
    return json.loads(text[first : last + 1])


def read_schedule_page(
    page_image: Image.Image,
    *,
    model: str = DEFAULT_MODEL,
    timeout_s: int = DEFAULT_TIMEOUT_S,
) -> ScheduleReadResult:
    """Send the schedule page to Claude vision; parse rows into typed
    records. Returns a result with `error` set if the call fails."""
    try:
        from anthropic import Anthropic, APIError, APITimeoutError
    except ImportError:
        return ScheduleReadResult(
            total_floor_area_m2=None, wet_area_total_m2=None,
            error="anthropic SDK not installed",
        )

    downsampled, _ = _downsample(page_image)
    img_b64 = _pil_to_b64(downsampled)

    client = Anthropic(timeout=timeout_s)
    raw_text = ""
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=DEFAULT_MAX_TOKENS,
            system=_SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {
                        "type": "base64", "media_type": "image/jpeg", "data": img_b64,
                    }},
                    {"type": "text", "text": (
                        f"Image is {downsampled.width} x {downsampled.height} px. "
                        "Return the JSON per system instructions."
                    )},
                ],
            }],
        )
        raw_text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
    except (APITimeoutError, APIError) as exc:
        return ScheduleReadResult(
            total_floor_area_m2=None, wet_area_total_m2=None,
            raw_response=raw_text, error=f"{type(exc).__name__}: {exc}",
        )
    except Exception as exc:  # noqa: BLE001
        return ScheduleReadResult(
            total_floor_area_m2=None, wet_area_total_m2=None,
            raw_response=raw_text, error=f"{type(exc).__name__}: {exc}",
        )

    try:
        parsed = _parse_json(raw_text)
    except (ValueError, json.JSONDecodeError) as exc:
        return ScheduleReadResult(
            total_floor_area_m2=None, wet_area_total_m2=None,
            raw_response=raw_text, error=f"JSON parse failed: {exc}",
        )

    def _num(v, cast=float):
        try:
            return cast(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    areas = []
    for a in (parsed.get("areas") or []):
        if not isinstance(a, dict):
            continue
        areas.append(ScheduleArea(
            room=str(a.get("room", "")).strip(),
            width_mm=_num(a.get("width_mm"), int),
            depth_mm=_num(a.get("depth_mm"), int),
            area_m2=_num(a.get("area_m2")),
            is_wet=bool(a.get("is_wet", False)),
        ))

    openings = []
    for o in (parsed.get("openings") or []):
        if not isinstance(o, dict):
            continue
        kind = str(o.get("kind", "")).lower().strip()
        if kind not in ("window", "door"):
            continue
        openings.append(ScheduleOpening(
            kind=kind,
            label=str(o.get("label", "")).strip(),
            location=str(o.get("location", "")).strip(),
            width_mm=_num(o.get("width_mm"), int),
            height_mm=_num(o.get("height_mm"), int),
            type_=str(o.get("type", "")).strip(),
            is_glass=bool(o.get("is_glass", kind == "window")),
        ))

    return ScheduleReadResult(
        total_floor_area_m2=_num(parsed.get("total_floor_area_m2")),
        wet_area_total_m2=_num(parsed.get("wet_area_total_m2")),
        areas=areas,
        openings=openings,
        raw_response=raw_text,
        error=None,
    )


# ============================================================
# Cross-validation
# ============================================================

# Per-field tolerance (percent of the larger value). Wall thickness
# alone can move floor area ±3-5%, so a 6% tolerance on areas is
# fair. Counts compare exactly (tolerance 0).
DEFAULT_TOLERANCES = {
    "total_floor_area":    6.0,
    "wet_area_total":      8.0,
    "building_perimeter":  3.0,
    "bathroom_count":      0.0,
    "glass_doors_windows": 0.0,
    "window_count":        0.0,
    "external_door_count": 0.0,
    "room_count":          0.0,
}


def _measurement_value(meas, field_key: str):
    """Find a measurement's field value by key."""
    for f in meas.extractions:
        if f.field_key == field_key:
            return f.value
    return None


def cross_validate(
    measurement,
    schedule: ScheduleReadResult,
    *,
    tolerances: Optional[dict[str, float]] = None,
) -> list[CrossValidationFinding]:
    """Per-field diff between measurement extraction and schedule read.

    Skips fields where either side has no value (can't compare). Only
    emits a finding when both sides have a number to compare.
    """
    tol = {**DEFAULT_TOLERANCES, **(tolerances or {})}
    findings: list[CrossValidationFinding] = []

    def _check(key: str, m_val, s_val, t_pct: float, note: str = ""):
        if m_val is None or s_val is None:
            return
        try:
            m = float(m_val); s = float(s_val)
        except (TypeError, ValueError):
            # non-numeric (e.g. string counts) — compare as equality
            agree = (m_val == s_val)
            findings.append(CrossValidationFinding(
                field_key=key, measurement_value=m_val, schedule_value=s_val,
                diff_abs=None, diff_pct=None, tolerance_pct=t_pct,
                agree=agree, note=note,
            ))
            return
        diff_abs = abs(m - s)
        base = max(abs(m), abs(s), 1e-9)
        diff_pct = (diff_abs / base) * 100.0
        agree = diff_pct <= t_pct
        findings.append(CrossValidationFinding(
            field_key=key, measurement_value=round(m, 2),
            schedule_value=round(s, 2),
            diff_abs=round(diff_abs, 2),
            diff_pct=round(diff_pct, 2),
            tolerance_pct=t_pct, agree=agree, note=note,
        ))

    _check("total_floor_area",
           _measurement_value(measurement, "total_floor_area"),
           schedule.total_floor_area_m2, tol["total_floor_area"],
           note="Compares Phase-2 derived area vs the total printed on the area schedule.")

    _check("wet_area_total",
           _measurement_value(measurement, "wet_area_total"),
           schedule.wet_area_total_m2, tol["wet_area_total"])

    _check("bathroom_count",
           _measurement_value(measurement, "bathroom_count"),
           schedule.bathroom_count, tol["bathroom_count"],
           note="Counts rooms whose name matches Bathroom/Ensuite/WC/Toilet/Powder. "
                "Laundry counted as wet area but excluded from bathroom count.")

    _check("glass_doors_windows",
           _measurement_value(measurement, "glass_doors_windows"),
           schedule.glass_count, tol["glass_doors_windows"])

    _check("window_count",
           _measurement_value(measurement, "window_count"),
           schedule.window_count, tol["window_count"])

    _check("external_door_count",
           _measurement_value(measurement, "external_door_count"),
           schedule.door_count, tol["external_door_count"])

    return findings


def build_cross_validation_block(
    findings: list[CrossValidationFinding],
    schedule: ScheduleReadResult,
    schedule_page: int,
) -> dict:
    """Shape into the dict that goes onto CombinedResult.cross_validation
    and ultimately into the client schema."""
    n_total = len(findings)
    n_agree = sum(1 for f in findings if f.agree)
    n_disagree = n_total - n_agree
    return {
        "schedule_page": schedule_page,
        "schedule_error": schedule.error,
        "schedule_total_floor_area_m2": schedule.total_floor_area_m2,
        "schedule_wet_area_total_m2": schedule.wet_area_total_m2,
        "schedule_window_count": schedule.window_count,
        "schedule_door_count": schedule.door_count,
        "schedule_glass_count": schedule.glass_count,
        "schedule_bathroom_count": schedule.bathroom_count,
        "comparisons": [f.to_dict() for f in findings],
        "summary": {
            "fields_compared": n_total,
            "fields_agree": n_agree,
            "fields_disagree": n_disagree,
            "all_agree": n_disagree == 0,
        },
    }


# ============================================================
# Demo: force a disagreement on cue
# ============================================================

def perturb_measurement_for_demo(
    measurement,
    *,
    floor_area_delta_pct: float = -15.0,
    bathroom_count_delta: int = 1,
) -> None:
    """Demo-only: mutate the measurement's `total_floor_area` and
    `bathroom_count` so the schedule cross-validation produces a
    DISAGREE finding on cue. Used to demonstrate the safety net live.
    """
    def _annotate(f, note: str) -> None:
        # Defensive: ExtractionField has `notes` (str), but tests sometimes
        # pass simpler mocks. Skip annotation if attribute missing.
        existing = getattr(f, "notes", None)
        if existing is None:
            existing = ""
        try:
            f.notes = existing + note
        except AttributeError:
            pass

    for f in measurement.extractions:
        if f.field_key == "total_floor_area" and isinstance(f.value, (int, float)):
            f.value = round(f.value * (1.0 + floor_area_delta_pct / 100.0), 2)
            _annotate(f,
                f"  [DEMO PERTURBATION: extraction multiplied by "
                f"{1 + floor_area_delta_pct/100:.2f} to demonstrate the "
                f"schedule cross-validation safety net.]"
            )
        elif f.field_key == "bathroom_count" and isinstance(f.value, int):
            f.value = max(0, f.value + bathroom_count_delta)
            _annotate(f,
                f"  [DEMO PERTURBATION: bathroom_count perturbed by "
                f"{bathroom_count_delta:+d} for safety-net demo.]"
            )
