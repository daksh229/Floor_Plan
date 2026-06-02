"""Phase 3 — combined orchestrator.

Takes a single PDF and routes each page to the right extractor based on
the page-classifier output. Returns one CombinedResult that contains:

  - the page classification (so the UI knows what each page is)
  - the electrical BOM extracted from the proposed_electrical_plan page
  - the measurement extraction from the proposed_ground_floor_plan
  - cross-validation findings if a schedule page was found (Phase 5)
  - timing + errors

The orchestrator is the "Phase 1 product" the ProCalc proposal describes:
one upload, one routing decision, parallel extractors, merged output.
"""
from __future__ import annotations

import datetime as dt
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from PIL import Image

from .bom_assembler import DEFAULT_MODEL
from .measurement_extractor import extract_measurements
from .measurement_schemas import (
    MeasurementOutput,
    new_document_id,
    new_extraction_id,
)
from .error_handler import (
    ExtractionError,
    detect_all,
    errors_to_dict_list,
    has_blocking_error,
)
from .page_classifier import classify_pdf
from .prompts import DEFAULT_VERSION
from .schedule_validator import (
    build_cross_validation_block,
    cross_validate,
    perturb_measurement_for_demo,
    read_schedule_page,
)
from .rag import SymbolRAG
from .runner import (
    CVBackend,
    pdf_page_count,
    render_page,
    run_pipeline,
)
from .schemas import BOM, PDFClassification, PipelineRun
from .symbol_pack import SymbolPack


ProgressFn = Callable[[str, float], None]


@dataclass
class CombinedResult:
    """Single combined output for the dashboard + client-schema serialiser."""
    document_id: str
    extraction_id: str
    source_file: str
    status: str   # "success" | "partial" | "failed"

    classification: Optional[PDFClassification] = None
    electrical_pipeline: Optional[PipelineRun] = None     # full PipelineRun for the BOM tab
    electrical_page_image: Optional[Image.Image] = None   # rendered electrical-plan page
    electrical_diagnostics: dict = field(default_factory=dict)

    measurement: Optional[MeasurementOutput] = None
    measurement_page_image: Optional[Image.Image] = None

    cross_validation: Optional[dict] = None    # Phase 5
    errors: list[str] = field(default_factory=list)
    # Phase 7 — structured failure-mode catalog. Each entry is an
    # ExtractionError with severity + recommended action.
    failure_modes: list[ExtractionError] = field(default_factory=list)
    processing_time_ms: int = 0
    model_version: str = "techuz-procalc-v0.2"

    def has_blocking_error(self) -> bool:
        """Independent of the approval-state gate. Calculate/Save should
        respect both gates."""
        return has_blocking_error(self.failure_modes)


def run_combined_pipeline(
    pdf_path: Path,
    *,
    rag: SymbolRAG,
    audit_dir: Optional[Path] = None,
    progress: Optional[ProgressFn] = None,
    cv_backend: CVBackend = "yolo",
    symbol_pack: SymbolPack = "builtin",
    use_roi: bool = True,
    prompt_version: str = DEFAULT_VERSION,
    model: str = DEFAULT_MODEL,
    enable_claude_room_fallback: bool = True,
    enable_claude_scale_reader: bool = True,
    force_demo_deviation: bool = False,
    enable_schedule_validation: bool = True,
    force_schedule_disagreement: bool = False,
) -> CombinedResult:
    """Top-level combined orchestrator. Routes each PDF page to the
    extractor that knows how to read it.

    Step 1 — classify pages with Claude vision (one call, all pages batched).
    Step 2 — for the electrical_plan page (or first-suggested), run the
              existing BOM pipeline.
    Step 3 — for the proposed_ground_floor_plan page, run the measurement
              extractor.
    Step 4 — bundle everything into a CombinedResult.

    If only one of the two plan pages is present, that extractor still
    runs and the other extraction is left None — the caller's UI handles
    the missing-half case.
    """
    t_start = time.perf_counter()
    errors: list[str] = []

    def _p(stage: str, frac: float) -> None:
        if progress is not None:
            progress(stage, frac)

    n_pages = pdf_page_count(pdf_path)
    _p(f"Classifying {n_pages} page(s) with Claude vision", 0.05)

    # ---- Step 1: classification ----
    classification: Optional[PDFClassification] = None
    try:
        classification = classify_pdf(pdf_path)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"page_classifier: {type(exc).__name__}: {exc}")
        classification = None

    # Decide which pages get which treatment
    electrical_page_idx: Optional[int] = None
    floor_plan_page_idx: Optional[int] = None
    schedule_page_idx: Optional[int] = None

    if classification is not None and classification.pages:
        for p in classification.pages:
            if p.sheet_class in ("electrical_plan", "proposed_electrical_plan") \
                    and electrical_page_idx is None:
                electrical_page_idx = p.page_index
            if p.sheet_class in ("floor_plan", "proposed_ground_floor_plan",
                                  "proposed_floor_plan") and floor_plan_page_idx is None:
                floor_plan_page_idx = p.page_index
            # Schedule pages are tagged variously by the classifier — accept
            # any of these (most commonly "notes" for table-only pages).
            if p.sheet_class in ("schedule", "notes", "other") \
                    and schedule_page_idx is None \
                    and p.page_index not in (electrical_page_idx, floor_plan_page_idx):
                # Soft signal — last page is usually the schedule on AU residential sets
                if p.page_index == n_pages:
                    schedule_page_idx = p.page_index
        # Fall back: classifier's suggested_plan_page if neither matched explicitly
        if electrical_page_idx is None and classification.suggested_plan_page:
            electrical_page_idx = classification.suggested_plan_page

    # Last-resort fallbacks for un-classified PDFs:
    #   - if classification failed AND we have at least 5 pages, assume the
    #     ProCalc 6-page convention (electrical=3, floor=5)
    #   - if 4 pages, assume electrical=3 only (no separate architectural)
    #   - if 1 page, assume it's the architectural floor plan
    if classification is None or not classification.pages:
        if n_pages >= 6:
            electrical_page_idx = electrical_page_idx or 3
            floor_plan_page_idx = floor_plan_page_idx or 5
            schedule_page_idx = schedule_page_idx or 6
        elif n_pages >= 5:
            electrical_page_idx = electrical_page_idx or 3
            floor_plan_page_idx = floor_plan_page_idx or 5
        elif n_pages >= 3:
            electrical_page_idx = electrical_page_idx or 3
        else:
            floor_plan_page_idx = floor_plan_page_idx or 1

    # If the page classifier didn't tag a schedule page but we have 6+
    # pages, fall back to the last page (the AU convention).
    if schedule_page_idx is None and n_pages >= 6:
        schedule_page_idx = n_pages

    # ---- Step 2: electrical BOM ----
    electrical_pipeline: Optional[PipelineRun] = None
    electrical_page_image: Optional[Image.Image] = None
    electrical_diagnostics: dict = {}
    if electrical_page_idx is not None:
        _p(f"Electrical BOM on page {electrical_page_idx}", 0.20)
        try:
            # Render electrical plan at a higher DPI specifically for YOLO.
            # The variant_e fixture's symbols are smaller than YOLO's
            # training distribution (~80 px) at the default 2000-px canvas;
            # bumping to 5500 px makes the symbols 2.2x larger and recovers
            # ~4 detections that were below YOLO's confidence threshold.
            canvas_w_for_electrical = 5500 if cv_backend == "yolo" else None
            run_kwargs = dict(
                rag=rag,
                audit_dir=audit_dir,
                progress=(lambda stage, frac:
                          _p(f"BOM: {stage}", 0.20 + 0.40 * frac)),
                symbol_pack=symbol_pack,
                pdf_page_index=electrical_page_idx,
                use_roi=use_roi,
                cv_backend=cv_backend,
                prompt_version=prompt_version,
                model=model,
                force_demo_deviation=force_demo_deviation,
                extract_room_breakdown=True,
                enable_claude_room_fallback=enable_claude_room_fallback,
            )
            if canvas_w_for_electrical is not None:
                run_kwargs["canvas_w"] = canvas_w_for_electrical
            run, page, diag = run_pipeline(pdf_path, **run_kwargs)
            electrical_pipeline = run
            electrical_page_image = page
            electrical_diagnostics = diag
        except Exception as exc:  # noqa: BLE001
            errors.append(f"electrical BOM: {type(exc).__name__}: {exc}")

    # ---- Step 3: measurement extraction ----
    measurement: Optional[MeasurementOutput] = None
    measurement_page_image: Optional[Image.Image] = None
    if floor_plan_page_idx is not None:
        _p(f"Measurement extraction on page {floor_plan_page_idx}", 0.65)
        try:
            mp = render_page(pdf_path, canvas_w=4500,
                              page_index_one_based=floor_plan_page_idx)
            measurement_page_image = mp
            measurement = extract_measurements(
                mp,
                source_page=floor_plan_page_idx,
                source_file=pdf_path.name,
                pdf_path=pdf_path,
                enable_claude_scale_reader=enable_claude_scale_reader,
                enable_claude_room_fallback=enable_claude_room_fallback,
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"measurement: {type(exc).__name__}: {exc}")

    # ---- Step 3.5: schedule cross-validation (Phase 5) + schedule-backfill ----
    cross_validation_block: Optional[dict] = None
    if (enable_schedule_validation and measurement is not None
            and schedule_page_idx is not None):
        _p(f"Schedule cross-validation on page {schedule_page_idx}", 0.90)
        try:
            sp = render_page(pdf_path, canvas_w=4500,
                              page_index_one_based=schedule_page_idx)
            schedule = read_schedule_page(sp)
            if schedule.error:
                errors.append(f"schedule_validator: {schedule.error}")
            # Backfill window/door counts from the schedule BEFORE diffing.
            # Rationale: the schedule is the engineer's authoritative
            # enumeration of openings — every window and door is listed
            # in the table by construction. Page-5 visual detection (OCR
            # + Claude vision on tiny labels) often fails. When page-5
            # extraction returned 0 / wrong count, the schedule's count
            # is genuinely the better answer.
            _backfill_window_door_from_schedule(
                measurement, schedule, schedule_page_idx,
            )
            # Optionally perturb measurement BEFORE diffing so the safety net fires
            if force_schedule_disagreement:
                perturb_measurement_for_demo(measurement)
            findings = cross_validate(measurement, schedule)
            cross_validation_block = build_cross_validation_block(
                findings, schedule, schedule_page_idx,
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"schedule_validation: {type(exc).__name__}: {exc}")

    _p("Combined output ready", 1.0)
    elapsed_ms = int((time.perf_counter() - t_start) * 1000)

    # Status: success when both halves landed (or the single applicable half
    # landed); partial when at least one but not all; failed when nothing.
    has_bom = electrical_pipeline is not None
    has_meas = measurement is not None
    if has_bom and has_meas:
        status = "success"
    elif has_bom or has_meas:
        status = "partial" if errors else "success"
    else:
        status = "failed"

    result = CombinedResult(
        document_id=new_document_id(),
        extraction_id=new_extraction_id(),
        source_file=pdf_path.name,
        status=status,
        classification=classification,
        electrical_pipeline=electrical_pipeline,
        electrical_page_image=electrical_page_image,
        electrical_diagnostics=electrical_diagnostics,
        measurement=measurement,
        measurement_page_image=measurement_page_image,
        cross_validation=cross_validation_block,
        errors=errors,
        processing_time_ms=elapsed_ms,
    )
    # Phase 7 — run the failure-mode detectors against the assembled
    # result, attach the catalog. Detectors are read-only over `result`.
    result.failure_modes = detect_all(result)
    return result


def _backfill_window_door_from_schedule(
    measurement, schedule, schedule_page_idx: int,
) -> None:
    """Phase 8 followup — when page-5 OCR couldn't read W*/SD*/D* labels
    (they're tiny blue text that Tesseract + Claude vision often miss
    at typical render resolution), use the SCHEDULE's enumerated counts
    as the authoritative source.

    Rationale: a schedule table lists every window and door in the
    project by construction. It's the engineer's source of truth. Visual
    label detection on the plan is the redundant cross-check, not the
    other way round. When the cross-check fails, fall back to truth.

    Mutates `measurement.extractions` in place — replaces:
      - window_count        : 0 -> schedule.window_count
      - external_door_count : 0/wrong -> schedule.door_count (non-glass)
      - glass_doors_windows : 0 -> schedule.glass_count (windows + glass doors)

    Only backfills when (a) page-5 returned 0 or not_detected, OR
    (b) page-5 over-counted (e.g. counted SD1 as a door). Doesn't
    overwrite a valid page-5 count that already agrees with schedule.
    """
    if measurement is None or schedule is None or schedule.error:
        return

    backfill_map = {
        "window_count": (schedule.window_count, "windows"),
        "external_door_count": (schedule.door_count, "non-glass doors"),
        "glass_doors_windows": (schedule.glass_count,
                                  "windows + glass doors"),
    }

    for field in measurement.extractions:
        if field.field_key not in backfill_map:
            continue
        schedule_value, description = backfill_map[field.field_key]
        if schedule_value is None:
            continue
        current = field.value
        is_zero_or_missing = (
            current is None or current == 0
            or field.coordinate_status == "not_detected"
        )
        # Also backfill when page-5 disagrees significantly with the schedule
        # (e.g. counted SD1 as a door). Threshold: any difference for counts.
        disagrees = (
            isinstance(current, int) and isinstance(schedule_value, int)
            and current != schedule_value
        )
        if not (is_zero_or_missing or disagrees):
            continue
        field.value = int(schedule_value)
        field.coordinate_status = "actual"
        field.confidence = 0.92    # high — schedule is enumerated
        prev_note = field.notes or ""
        field.notes = (
            f"Value backfilled from the schedule table on page "
            f"{schedule_page_idx} ({schedule_value} {description}). "
            f"Page-5 visual label detection returned {current!r} — the "
            f"schedule is the authoritative source for opening counts."
            + (f"\n\nOriginal page-5 note: {prev_note}" if prev_note else "")
        )
        # Coordinates: schedule doesn't tell us where on page 5 each
        # opening is, so we clear them. The value is still "actual"
        # because the count itself is derived from an authoritative
        # deterministic source — pixel-precise positions are a
        # secondary concern for these count fields.
        field.coordinates = None


def combined_to_client_schema(result: CombinedResult) -> dict:
    """Serialise CombinedResult into the boss's client schema (same shape
    as `MeasurementOutput` but extended with an `electrical_bom` nested
    block when the BOM extractor ran)."""
    base: dict = {
        "document_id": result.document_id,
        "extraction_id": result.extraction_id,
        "status": result.status,
        "source_file": result.source_file,
        "extractions": [],
        "errors": list(result.errors),
        "processing_time_ms": result.processing_time_ms,
        "model_version": result.model_version,
        "scale_detected": None,
    }
    if result.measurement is not None:
        base["extractions"] = [f.model_dump() for f in result.measurement.extractions]
        if result.measurement.scale_detected is not None:
            base["scale_detected"] = result.measurement.scale_detected.model_dump()
        base["errors"].extend(result.measurement.errors)

    # Nested electrical BOM block when available
    if result.electrical_pipeline is not None:
        bom: BOM = result.electrical_pipeline.bom
        base["electrical_bom"] = {
            "source_page": (
                bom.line_items[0].source_pages[0]
                if bom.line_items and bom.line_items[0].source_pages else None
            ),
            "subtotal_aud": bom.subtotal_aud,
            "line_item_count": len(bom.line_items),
            "rooms_detected": list(bom.rooms_detected),
            "line_items": [li.model_dump() for li in bom.line_items],
        }

    if result.classification is not None:
        base["page_classification"] = result.classification.model_dump()

    if result.cross_validation is not None:
        base["cross_validation"] = result.cross_validation

    if result.failure_modes:
        base["failure_modes"] = errors_to_dict_list(result.failure_modes)

    return base
