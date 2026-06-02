"""Phase 7 — error catalog + failure-mode detection.

The ProCalc proposal §7 #7 calls this out as a primary failure mode for
inexperienced teams: *"no retry, no timeout, no partial-result handling.
Every failure mode in section 10 of the brief needs a defined message
and fallback."* This module is that defined-message-and-fallback layer.

Three responsibilities:

  1. Define a canonical catalog of named error codes that the pipeline
     can emit. Each has a severity, user-facing title + message, and a
     recommended next action (which can be a UI button-action).
  2. Detection functions — given a CombinedResult, scan for each
     failure mode and emit ExtractionError records. Pure functions; the
     orchestrator collects them into `CombinedResult.failure_modes`.
  3. Helper to determine if any BLOCKING error exists — used by the
     UI to gate Calculate/Save (independently of approval state).

The detectors are independent — adding a new failure mode is two
edits: a new ErrorCode entry + a new `_detect_xxx()` function.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


# ============================================================
# Catalog
# ============================================================

class ErrorCode(str, Enum):
    """Named failure modes. Stringly-typed for JSON-friendliness."""
    NO_PLAN_DETECTED       = "no_plan_detected"
    MULTIPLE_FLOOR_PLANS   = "multiple_floor_plans"
    SCALE_DISAGREEMENT     = "scale_disagreement"
    PARTIAL_EXTRACTION     = "partial_extraction"
    CLAUDE_PARTIAL_FAILURE = "claude_partial_failure"
    CLAUDE_TOTAL_FAILURE   = "claude_total_failure"
    NO_DIMENSIONS_FOUND    = "no_dimensions_found"
    OCR_UNAVAILABLE        = "ocr_unavailable"
    PDF_RENDER_FAILED      = "pdf_render_failed"
    SCHEDULE_UNREADABLE    = "schedule_unreadable"


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    BLOCKING = "blocking"   # gates Calculate/Save


@dataclass
class ExtractionError:
    """One detected failure mode. Designed for direct rendering in the UI."""
    code: ErrorCode
    severity: Severity
    title: str
    message: str
    recommended_action: str
    field_keys_affected: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "code": self.code.value,
            "severity": self.severity.value,
            "title": self.title,
            "message": self.message,
            "recommended_action": self.recommended_action,
            "field_keys_affected": list(self.field_keys_affected),
            "extra": dict(self.extra),
        }


# ============================================================
# Detection functions
# ============================================================
#
# Each detector takes a CombinedResult and returns either an
# ExtractionError or None. They're aggregated by `detect_all`.
# Avoiding a hard import of CombinedResult to keep this module
# free of cycles — the type is implicit via duck-typing.

def _detect_no_plan(combined) -> Optional[ExtractionError]:
    """No electrical_plan AND no floor_plan page found."""
    if combined.electrical_pipeline is None and combined.measurement is None:
        return ExtractionError(
            code=ErrorCode.NO_PLAN_DETECTED,
            severity=Severity.BLOCKING,
            title="No plan content detected",
            message=(
                "The page classifier could not find a plan page (electrical "
                "or architectural floor plan) in this PDF. None of the "
                "extractors ran. Possible reasons: the upload isn't a "
                "construction drawing, the plan is on a page the classifier "
                "couldn't recognise, or Claude vision returned an error."
            ),
            recommended_action=(
                "Verify the upload contains a residential plan. If yes, "
                "try the manual page-picker in Step 4 to specify which "
                "page contains the plan."
            ),
        )
    return None


def _detect_multiple_floor_plans(combined) -> Optional[ExtractionError]:
    """Classifier flagged >1 page as floor_plan / electrical_plan."""
    if combined.classification is None:
        return None
    plan_classes = {"electrical_plan", "floor_plan", "proposed_ground_floor_plan",
                    "proposed_electrical_plan", "proposed_floor_plan"}
    plans = [p for p in combined.classification.pages if p.sheet_class in plan_classes]
    if len(plans) <= 2:   # 1 electrical + 1 architectural is the normal case
        return None
    plan_summary = ", ".join(f"p{p.page_index}={p.sheet_class}" for p in plans)
    return ExtractionError(
        code=ErrorCode.MULTIPLE_FLOOR_PLANS,
        severity=Severity.WARNING,
        title=f"{len(plans)} plan pages detected — possible multi-storey set",
        message=(
            f"The classifier identified {len(plans)} plan pages "
            f"({plan_summary}). The combined pipeline processed only the "
            f"first electrical + first architectural plan it found. If this "
            f"is a multi-storey project (ground + first floor), other floors "
            f"are NOT in the current extraction."
        ),
        recommended_action=(
            "If multi-storey: re-run separately for each floor by specifying "
            "the page index in Step 4. Phase 2 of the proposal will handle "
            "multi-storey automatically."
        ),
        extra={"plan_pages": [p.model_dump() for p in plans]},
    )


def _detect_scale_disagreement(combined) -> Optional[ExtractionError]:
    """Scale-bar reader (Phase 1, Claude vision) and dimension-derived
    calibration (Phase 2) both returned a scale, and they differ by more
    than 20%.
    """
    if combined.measurement is None:
        return None
    meas = combined.measurement
    scale_bar = meas.scale_detected   # Claude scale-bar reader
    # Find the scale_detected ExtractionField (may be sourced from dim chain)
    field_scale = next(
        (f for f in meas.extractions if f.field_key == "scale_detected"), None,
    )
    if scale_bar is None or field_scale is None:
        return None
    if field_scale.value is None or scale_bar.value == field_scale.value:
        return None

    def _parse_ratio(s: str) -> Optional[float]:
        if not isinstance(s, str) or ":" not in s:
            return None
        try:
            _, n = s.split(":", 1)
            return float(n.strip())
        except (ValueError, AttributeError):
            return None

    a = _parse_ratio(scale_bar.value)
    b = _parse_ratio(field_scale.value)
    if a is None or b is None or max(a, b) == 0:
        return None
    rel_diff = abs(a - b) / max(a, b)
    if rel_diff < 0.20:
        return None
    return ExtractionError(
        code=ErrorCode.SCALE_DISAGREEMENT,
        severity=Severity.WARNING,
        title="Scale-bar and dimension calibration disagree",
        message=(
            f"Two scale sources returned different values: "
            f"scale bar → '{scale_bar.value}' ({scale_bar.source}, "
            f"confidence {scale_bar.confidence:.0%}); "
            f"dimension chain → '{field_scale.value}' "
            f"(confidence {field_scale.confidence:.0%}). "
            f"Relative difference: {rel_diff:.0%}. "
            "On a real drawing this is usually the scale-bar glyph being "
            "wrong (e.g. label says 1:100 but plan is rendered at 1:50 for "
            "a detail). The dimension-chain calibration is usually correct."
        ),
        recommended_action=(
            "Trust the dimension-chain calibration unless you have a reason "
            "not to. Sight-check one or two perimeter walls against the "
            "printed dim chain to confirm."
        ),
        field_keys_affected=["scale_detected", "total_floor_area",
                             "building_perimeter"],
    )


def _detect_partial_extraction(combined) -> Optional[ExtractionError]:
    """More than 30% of measurement fields ended up not_detected."""
    if combined.measurement is None:
        return None
    fields = combined.measurement.extractions
    if not fields:
        return None
    n_total = len(fields)
    n_not_detected = sum(1 for f in fields if f.coordinate_status == "not_detected")
    if n_not_detected / n_total < 0.30:
        return None
    missing = [f.field_key for f in fields if f.coordinate_status == "not_detected"]
    return ExtractionError(
        code=ErrorCode.PARTIAL_EXTRACTION,
        severity=Severity.WARNING,
        title=f"Partial extraction — {n_not_detected} of {n_total} fields not detected",
        message=(
            f"The extractors ran but {n_not_detected} field(s) "
            f"could not be detected. Missing: "
            f"{', '.join(missing[:6])}{' …' if len(missing) > 6 else ''}. "
            "This usually means the plan didn't include the relevant "
            "information (no scale bar, no window schedule, etc.) or the "
            "labels were too small for OCR + Claude vision to read."
        ),
        recommended_action=(
            "Check each not_detected row in the Measurement & Approval tab. "
            "Use the Edit button on each one to enter the value manually "
            "(falls into user_corrected state, unlocks Calculate/Save when "
            "all rows are resolved)."
        ),
        field_keys_affected=missing,
    )


def _detect_claude_failures(combined) -> Optional[ExtractionError]:
    """Look at combined.errors for Claude-related failure strings.
    If multiple Claude-touching steps failed, surface a single
    consolidated error."""
    if not combined.errors:
        return None
    claude_hits = [
        e for e in combined.errors
        if "claude" in e.lower() or "anthropic" in e.lower() or "apierror" in e.lower()
    ]
    if not claude_hits:
        return None
    severity = Severity.ERROR if len(claude_hits) >= 2 else Severity.WARNING
    code = (
        ErrorCode.CLAUDE_TOTAL_FAILURE if len(claude_hits) >= 3
        else ErrorCode.CLAUDE_PARTIAL_FAILURE
    )
    title = (
        "Claude vision unavailable — manual fallback recommended"
        if code == ErrorCode.CLAUDE_TOTAL_FAILURE
        else "Some Claude calls failed — partial extraction"
    )
    return ExtractionError(
        code=code, severity=severity,
        title=title,
        message=(
            f"{len(claude_hits)} Claude-touching step(s) reported an error:\n"
            + "\n".join(f"  • {e}" for e in claude_hits[:5])
        ),
        recommended_action=(
            "Check the ANTHROPIC_API_KEY is set in poc/.env. If the API is "
            "down, the deterministic extractors (template matching / OCR / "
            "Hough wall detection) still produced what they could — review "
            "fields in the Measurement & Approval tab and fill any gaps "
            "manually."
        ),
        extra={"claude_errors": claude_hits},
    )


def _detect_no_dimensions(combined) -> Optional[ExtractionError]:
    """Phase 2 detected fewer than 2 dimension readings — cannot calibrate."""
    if combined.measurement is None:
        return None
    # Heuristic: look at notes on total_floor_area — if it's still
    # placeholder, we know calibration didn't land.
    for f in combined.measurement.extractions:
        if (f.field_key == "total_floor_area"
                and f.coordinate_status == "placeholder"):
            return ExtractionError(
                code=ErrorCode.NO_DIMENSIONS_FOUND,
                severity=Severity.WARNING,
                title="Cannot derive scale — no usable perimeter dimensions",
                message=(
                    "Phase 2's dimension calibrator did not find enough "
                    "numeric dimension labels on the plan to derive a "
                    "scale. Without scale, floor area, perimeter, and "
                    "envelope cannot be computed."
                ),
                recommended_action=(
                    "Either: (a) supply the scale manually via the Edit "
                    "button on the scale_detected field, then re-run "
                    "(future enhancement); or (b) ensure the plan has "
                    "legible perimeter dimensions on a non-rotated wall."
                ),
                field_keys_affected=["total_floor_area", "building_perimeter",
                                     "building_envelope", "wet_area_total"],
            )
    return None


def _detect_ocr_unavailable(combined) -> Optional[ExtractionError]:
    if combined.measurement is None:
        return None
    for e in combined.measurement.errors:
        if "tesseract" in e.lower() or "ocr unavailable" in e.lower():
            return ExtractionError(
                code=ErrorCode.OCR_UNAVAILABLE,
                severity=Severity.ERROR,
                title="OCR (Tesseract) unavailable",
                message=(
                    "Tesseract binary was not found on this system. OCR-"
                    "derived fields (window/door counts, room labels via "
                    "OCR, dimension OCR for scale calibration) will be "
                    "empty or low-quality. Claude-vision fallbacks may "
                    "still recover most of them."
                ),
                recommended_action=(
                    "Install Tesseract from https://github.com/UB-Mannheim/"
                    "tesseract/wiki (Windows) or `brew install tesseract` "
                    "(macOS) / `apt install tesseract-ocr` (Linux). Then "
                    "re-run the wizard."
                ),
            )
    return None


def _detect_schedule_unreadable(combined) -> Optional[ExtractionError]:
    if combined.cross_validation is None:
        return None
    err = combined.cross_validation.get("schedule_error")
    if not err:
        return None
    return ExtractionError(
        code=ErrorCode.SCHEDULE_UNREADABLE,
        severity=Severity.INFO,
        title="Schedule page couldn't be read — cross-validation skipped",
        message=(
            f"The schedule reader returned an error: {err}. "
            "Cross-validation between the page-5 measurement and the "
            "schedule table did not run. The measurement extraction is "
            "unaffected — only the cross-validation layer."
        ),
        recommended_action=(
            "If the PDF doesn't have a schedule page, this is expected. "
            "If it does, check that the schedule is on the last page and "
            "rendered legibly."
        ),
    )


# Registry of detectors — order is the order they appear in the UI
_DETECTORS: list[Callable[[Any], Optional[ExtractionError]]] = [
    _detect_no_plan,
    _detect_claude_failures,
    _detect_multiple_floor_plans,
    _detect_no_dimensions,
    _detect_scale_disagreement,
    _detect_partial_extraction,
    _detect_ocr_unavailable,
    _detect_schedule_unreadable,
]


def detect_all(combined) -> list[ExtractionError]:
    """Run every registered detector. Returns the union of findings."""
    out: list[ExtractionError] = []
    for detector in _DETECTORS:
        try:
            err = detector(combined)
            if err is not None:
                out.append(err)
        except Exception as exc:  # noqa: BLE001
            # A detector itself raising is suspicious but shouldn't break
            # the run — emit a meta-error so we know the detector misfired.
            out.append(ExtractionError(
                code=ErrorCode.CLAUDE_PARTIAL_FAILURE,
                severity=Severity.WARNING,
                title=f"Error detector {detector.__name__} crashed",
                message=f"{type(exc).__name__}: {exc}",
                recommended_action="Inspect the logs and file a bug.",
            ))
    return out


def has_blocking_error(errors: list[ExtractionError]) -> bool:
    """The proposal's Calculate/Save lock checks this in addition to
    the approval-state gate. A BLOCKING failure mode means the user
    shouldn't be allowed to push to the calculator regardless of
    field-level confirmations."""
    return any(e.severity == Severity.BLOCKING for e in errors)


def errors_to_dict_list(errors: list[ExtractionError]) -> list[dict]:
    """Serialise for inclusion in the client-schema output."""
    return [e.to_dict() for e in errors]
