"""Single end-to-end pipeline runner + cached-run persistence.

Used by both `scripts/capture_cached_run.py` (one-shot capture for the demo
fallback) and `app.py` (live runs initiated from the Streamlit UI).

The function takes an optional `progress` callback so the UI can advance a
status indicator after each stage without coupling the runner to Streamlit.
"""
from __future__ import annotations

import json
import shutil
import time
import uuid
from pathlib import Path
from typing import Callable, Literal, Optional

import pypdfium2 as pdfium
from PIL import Image

from .bom_assembler import (
    DEFAULT_MODEL,
    assemble_bom,
    compute_aggregated_payload,
)
from .cv_detect import detect_symbols as detect_symbols_tm
from .ocr import TesseractNotInstalledError, extract_text_spans
from .prompts import DEFAULT_VERSION, FORCE_DEVIATION_VERSION
from .rag import SymbolRAG
from .roi_extractor import extract_plan_roi
from .room_extractor import (
    assign_detections_to_rooms,
    compute_distribution_warning,
    extract_rooms_with_fallback,
)
from .schemas import Detection, PipelineRun, ROIResult, RoomExtraction, SymbolWithMatches
from .symbol_pack import SymbolPack, resolve_symbol_pngs


DEFAULT_CANVAS_W = 2000
# When a user pack is active we render at a higher DPI so symbols on the plan
# page are large enough to match templates extracted from the legend page
# (which the legend extractor renders at ~2.5x). Without this, multi-scale
# template matching can't bridge the resolution gap.
USER_PACK_CANVAS_W = 4500
# Plan symbols are often considerably smaller than the legend cells they were
# cropped from (legend ~80 px, plan ~20-50 px). The default 0.7-1.3 scale
# range can't bridge that. For user pack, extend down to 0.30 so a 80 px
# template can match a 24 px plan symbol — required to catch tiny GPOs /
# switches / data points / downlights. Cost is ~80 % more CV runtime.
USER_PACK_SCALES = (0.30, 0.40, 0.55, 0.70, 0.85, 1.00, 1.15, 1.30)
# Smaller templates score lower in TM_CCOEFF_NORMED (less signal per pixel)
# so a lower default threshold is needed to keep recall up. False positives
# get filtered downstream by cross-class NMS and the BOM confidence-band
# warning surfaced in the UI.
USER_PACK_DEFAULT_THRESHOLD = 0.50

# Per-class threshold overrides specifically for user-pack mode. The
# built-in PER_CLASS_THRESHOLD was tuned for padded 80 px built-in
# templates; with tight-cropped user templates at scales down to 0.30,
# a few classes explode in false positives without a higher floor.
# These were calibrated against the variant_b plan page to keep counts
# in the plausible range while letting all 15 classes register.
USER_PACK_PER_CLASS_THRESHOLD: dict[str, float] = {
    "distribution_board": 0.70,
    "exhaust_fan":        0.62,   # iterated: 0.68 missed all, 0.55 over-fired (33), settle 0.62
    "downlight":          0.78,
    "single_gpo":         0.55,
    "double_gpo":         0.62,
    "wp_gpo":             0.58,
    "ceiling_fan":        0.53,   # iterated: 0.50 over-fires (11), 0.58 misses; settle 0.53
    "ceiling_light":      0.55,
    "smoke_detector":     0.60,
    "dimmer":             0.40,   # tiny + "DIM" text; floor of what we can risk
    "wall_light":         0.65,
    "two_way_switch":     0.55,
    "single_pole_switch": 0.55,
    "data_point":         0.58,
    "tv_point":           0.58,
}

CVBackend = Literal["template", "yolo"]
ProgressFn = Callable[[str, float], None]


def render_page(pdf_path: Path, canvas_w: int = DEFAULT_CANVAS_W,
                page_index_one_based: int = 1) -> Image.Image:
    """Render a chosen page of a PDF at a target pixel width."""
    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        page = pdf[page_index_one_based - 1]
        page_w_pt = page.get_size()[0]
        scale = canvas_w / page_w_pt
        return page.render(scale=scale).to_pil()
    finally:
        pdf.close()


def pdf_page_count(pdf_path: Path) -> int:
    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        return len(pdf)
    finally:
        pdf.close()


def _compute_roi_padding_pct(roi: ROIResult, page_w: int, page_h: int) -> float:
    """How much to expand Claude's bbox before cropping, as a percentage of
    page-edge length per side.

    Two factors push it up:
      - Low confidence — Claude is less sure the bbox is right, so be more
        generous with the safety margin.
      - Tight bbox — when Claude's bbox covers a small fraction of the page,
        it's more likely to have over-cropped, so we add more padding.

    A perfectly-classified bbox covering >=35 % of the page gets just 4 %
    padding; a low-confidence tiny bbox can get up to ~18 %.

    Empirical calibration: variant_b (square plan, conf 0.88, area 11 %)
    lands ~9 %; variant_c (long horizontal plan, same numbers) also ~9 %.
    The earlier fixed 12 % over-padded variant_c into the title-boundary
    region, causing the FPs the user reported.
    """
    base = 0.04
    # Confidence penalty: confidence below 0.85 adds up to 8 pp
    conf_pad = max(0.0, 0.85 - roi.confidence) * 0.10
    # Tightness penalty: bbox area < 35 % of page adds up to 7 pp
    tight_pad = 0.0
    if roi.bbox is not None:
        bx0, by0, bx1, by1 = roi.bbox
        bbox_area = max(0, bx1 - bx0) * max(0, by1 - by0)
        page_area = page_w * page_h
        if page_area > 0:
            area_frac = bbox_area / page_area
            tight_pad = max(0.0, 0.35 - area_frac) * 0.20
    pad_pct = base + conf_pad + tight_pad
    # Sanity bounds
    return max(0.04, min(0.18, pad_pct))


def _translate_by_offset(
    detections: list[Detection], offset: tuple[int, int],
) -> list[Detection]:
    """Shift detection bboxes by (dx, dy) — used when CV ran on a cropped
    region and we need full-page coordinates for the overlay."""
    dx, dy = offset
    if dx == 0 and dy == 0:
        return detections
    out: list[Detection] = []
    for d in detections:
        x0, y0, x1, y1 = d.bbox
        out.append(d.model_copy(update={
            "bbox": (x0 + dx, y0 + dy, x1 + dx, y1 + dy),
        }))
    return out


def _load_mask_regions_for(pdf_path: Path) -> list[tuple[int, int, int, int]]:
    """Look up an adjacent <stem>_ground_truth.json and pull mask_regions out.

    The synth fixture writes the legend rectangle there. Real-PDF uploads
    have no GT file, so no masking is applied — which is the correct
    real-world behaviour.
    """
    gt_path = pdf_path.with_name(pdf_path.stem + "_ground_truth.json")
    if not gt_path.exists():
        return []
    try:
        gt = json.loads(gt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    out: list[tuple[int, int, int, int]] = []
    for entry in gt.get("mask_regions", []) or []:
        bbox = entry.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            out.append(tuple(int(v) for v in bbox))  # type: ignore[arg-type]
    return out


def run_pipeline(
    pdf_path: Path,
    *,
    symbol_dir: Path | None = None,  # legacy: ignored when symbol_pack != "builtin"
    rag: SymbolRAG,
    canvas_w: int = DEFAULT_CANVAS_W,
    audit_dir: Optional[Path] = None,
    progress: Optional[ProgressFn] = None,
    prompt_version: str = DEFAULT_VERSION,
    model: str = DEFAULT_MODEL,
    force_demo_deviation: bool = False,
    mask_regions: Optional[list[tuple[int, int, int, int]]] = None,
    symbol_pack: SymbolPack = "builtin",
    pdf_page_index: int = 1,
    use_roi: bool = False,
    cv_backend: CVBackend = "template",
    extract_room_breakdown: bool = True,
    enable_claude_room_fallback: bool = True,
) -> tuple[PipelineRun, Image.Image, dict]:
    """Run the full PDF → BOM pipeline and return (run, rendered page image, diagnostics).

    `rag` is passed in so the embedding model is loaded at most once across
    multiple runs (Streamlit re-runs the script on every interaction).

    If `force_demo_deviation` is True, the Claude system prompt is swapped
    for the `bom_v1_force_deviation` variant — the safety net will trigger
    and `diagnostics` will populate, so the override mechanism is observable.

    `mask_regions` (list of (x0,y0,x1,y1) rectangles) suppresses detections
    inside known non-content regions (legend, title block). If omitted, we
    auto-load mask regions from the PDF's adjacent ground-truth JSON.
    """
    def _p(stage: str, frac: float) -> None:
        if progress is not None:
            progress(stage, frac)

    t_total = time.perf_counter()
    run_id = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"

    # Bump render resolution when matching against the user pack: legend
    # templates are extracted at ~2.5x scale and the plan page needs to be
    # rendered at a compatible DPI for multi-scale matching to bridge.
    effective_canvas_w = canvas_w
    if symbol_pack in ("user", "both") and canvas_w == DEFAULT_CANVAS_W:
        effective_canvas_w = USER_PACK_CANVAS_W

    _p("Rendering page", 0.05)
    page = render_page(pdf_path, canvas_w=effective_canvas_w,
                       page_index_one_based=pdf_page_index)

    if mask_regions is None:
        mask_regions = _load_mask_regions_for(pdf_path)

    # Optional Phase-7 step: ask Claude to crop the page to just the
    # floor-plan region, excluding title block / legend / notes. CV then
    # runs on the crop, and detections are translated back to full-page
    # coords before being returned.
    roi: Optional[ROIResult] = None
    cv_target = page
    crop_offset = (0, 0)
    roi_pad_info: Optional[dict] = None
    if use_roi:
        _p("ROI: Claude selects plan region", 0.07)
        roi = extract_plan_roi(page, page_index=pdf_page_index)
        if roi.has_roi:
            assert roi.bbox is not None
            # Dynamic padding: smaller / less-confident bboxes get more pad,
            # confident wide bboxes get less. Avoids over-padding into the
            # title-boundary region on long horizontal plans while still
            # catching edge symbols on tight central ones. See
            # _compute_roi_padding_pct() for the formula.
            pad_pct = _compute_roi_padding_pct(roi, page.width, page.height)
            pad_x = max(60, int(page.width * pad_pct))
            pad_y = max(60, int(page.height * pad_pct))
            x0, y0, x1, y1 = roi.bbox
            x0p = max(0, x0 - pad_x)
            y0p = max(0, y0 - pad_y)
            x1p = min(page.width, x1 + pad_x)
            y1p = min(page.height, y1 + pad_y)
            cv_target = page.crop((x0p, y0p, x1p, y1p))
            crop_offset = (x0p, y0p)
            # surface chosen padding for the UI diagnostics
            roi_pad_info = {
                "pad_pct": round(pad_pct, 3),
                "pad_x_px": pad_x,
                "pad_y_px": pad_y,
                "padded_bbox": (x0p, y0p, x1p, y1p),
            }

    _p(f"CV symbol detection ({cv_backend})", 0.10)
    if cv_backend == "yolo":
        # YOLO learned scale + rotation invariance + the 15-class library at
        # training time, so it ignores the symbol-pack PNGs and the
        # user-pack threshold overrides entirely. The model is the library.
        from .yolo_detect import detect_symbols as detect_symbols_yolo
        raw_detections = detect_symbols_yolo(
            cv_target,
            source_page=pdf_page_index,
            mask_regions=mask_regions if cv_target is page else None,
        )
    else:
        pack_pngs = resolve_symbol_pngs(symbol_pack)
        detect_kwargs: dict = {}
        if symbol_pack in ("user", "both"):
            detect_kwargs["scales"] = USER_PACK_SCALES
            detect_kwargs["default_threshold"] = USER_PACK_DEFAULT_THRESHOLD
            detect_kwargs["per_class_threshold"] = USER_PACK_PER_CLASS_THRESHOLD
        raw_detections = detect_symbols_tm(
            cv_target,
            symbol_pngs=pack_pngs,
            source_page=pdf_page_index,
            mask_regions=mask_regions if cv_target is page else None,
            **detect_kwargs,
        )
    detections = _translate_by_offset(raw_detections, crop_offset)

    _p("OCR pass", 0.60)
    try:
        ocr_spans = extract_text_spans(page, source_page=pdf_page_index)
    except TesseractNotInstalledError:
        ocr_spans = []  # graceful degradation; BOM stage still works

    # Room extraction + per-detection assignment. Runs in full-page
    # coordinate space (detections already translated above) so the ROI
    # offset doesn't need to be applied to room label bboxes.
    #
    # Three layers of robustness:
    #   1. OCR-based extraction (cheap, fast, primary path)
    #   2. Claude-vision fallback when OCR yields <5 rooms (toggleable)
    #   3. Distance-gated nearest-centroid assignment so detections in
    #      undetected rooms fall into Unassigned rather than silently
    #      mis-attributing to the nearest labelled room
    room_extraction: Optional[RoomExtraction] = None
    room_assignments: dict[int, str | None] = {}
    distribution_warning: Optional[str] = None
    if extract_room_breakdown and ocr_spans:
        _p("Extracting room labels", 0.65)
        roi_bbox_full_page = roi.bbox if (roi is not None and roi.has_roi) else None
        room_extraction = extract_rooms_with_fallback(
            ocr_spans,
            page_image=page,
            roi_bbox=roi_bbox_full_page,
            source_page=pdf_page_index,
            enable_claude_fallback=enable_claude_room_fallback,
        )
        room_assignments = assign_detections_to_rooms(
            detections,
            room_extraction.rooms,
            page_size=(page.width, page.height),
        )
        distribution_warning = compute_distribution_warning(
            detections, room_assignments, len(room_extraction.rooms)
        )

    _p("RAG: matching detections to library", 0.70)
    rag.build()  # idempotent
    symbol_matches: list[SymbolWithMatches] = rag.match_detections(detections, k=3)

    _p("Aggregating payload", 0.80)
    payload = compute_aggregated_payload(
        detections, rag,
        ocr_spans=ocr_spans,
        pdf_name=pdf_path.name,
        page_count=1,
        room_assignments=room_assignments,
        rooms=room_extraction.rooms if room_extraction else None,
    )

    _p("Claude BOM assembly", 0.85)
    effective_prompt = FORCE_DEVIATION_VERSION if force_demo_deviation else prompt_version
    bom, diagnostics = assemble_bom(
        payload,
        prompt_version=effective_prompt,
        model=model,
        audit_dir=audit_dir,
    )

    _p("Done", 1.0)

    run = PipelineRun(
        run_id=run_id,
        source_pdf=pdf_path.name,
        detections=detections,
        ocr_spans=ocr_spans,
        symbol_matches=symbol_matches,
        bom=bom,
        elapsed_seconds=round(time.perf_counter() - t_total, 2),
    )
    # Surface the ROI result in diagnostics so the UI can show it
    if roi is not None:
        diagnostics["roi"] = roi.model_dump()
        if roi_pad_info is not None:
            diagnostics["roi_padding"] = roi_pad_info
    if room_extraction is not None:
        diagnostics["room_extraction"] = room_extraction.model_dump()
    if distribution_warning is not None:
        diagnostics["room_distribution_warning"] = distribution_warning
    return run, page, diagnostics


# ---------- cached-run persistence ----------

def save_pipeline_run(
    run: PipelineRun,
    page_image: Image.Image,
    dest_dir: Path,
    *,
    source_pdf_path: Optional[Path] = None,
) -> None:
    """Persist a PipelineRun + rendered page (+ optional source PDF copy)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    (dest_dir / "run.json").write_text(
        run.model_dump_json(indent=2), encoding="utf-8"
    )
    page_image.save(dest_dir / "page.png", "PNG")
    if source_pdf_path is not None and source_pdf_path.exists():
        shutil.copy2(source_pdf_path, dest_dir / "source.pdf")


def load_pipeline_run(src_dir: Path) -> tuple[PipelineRun, Image.Image]:
    """Inverse of save_pipeline_run. Raises FileNotFoundError if missing."""
    run = PipelineRun.model_validate_json(
        (src_dir / "run.json").read_text(encoding="utf-8")
    )
    page_image = Image.open(src_dir / "page.png").convert("RGB")
    return run, page_image
