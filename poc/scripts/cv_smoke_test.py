"""Phase 2 smoke test for the CV symbol detector.

What it does:
  1. Renders samples/synthetic_layout.pdf at the same pixel dimensions as the
     synth canvas (so ground-truth coords are directly comparable).
  2. Runs pipeline.cv_detect.detect_symbols against the rendered page.
  3. Prints per-class counts: ground-truth vs detected, diff.
  4. Saves samples/synthetic_layout_detections.png with the overlay drawn.

Run from poc/:
    python scripts/cv_smoke_test.py
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import pypdfium2 as pdfium

# Make `pipeline.*` importable when invoked as `python scripts/cv_smoke_test.py`
HERE = Path(__file__).resolve().parent
POC_ROOT = HERE.parent
if str(POC_ROOT) not in sys.path:
    sys.path.insert(0, str(POC_ROOT))

from pipeline.cv_detect import detect_symbols  # noqa: E402
from pipeline.overlay import draw_detections    # noqa: E402

PROJECT_ROOT = POC_ROOT.parent
PDF_PATH = PROJECT_ROOT / "samples" / "synthetic_layout.pdf"
GT_PATH = PROJECT_ROOT / "samples" / "synthetic_layout_ground_truth.json"
SYMBOL_DIR = POC_ROOT / "symbol_library"
OUT_OVERLAY = PROJECT_ROOT / "samples" / "synthetic_layout_detections.png"


def _render_page_at_canvas_size(pdf_path: Path, canvas_w: int):
    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        page = pdf[0]
        # page.get_size() returns (width_pt, height_pt); scale so render width matches canvas_w
        page_w_pt = page.get_size()[0]
        scale = canvas_w / page_w_pt
        img = page.render(scale=scale).to_pil()
        return img
    finally:
        pdf.close()


def main() -> int:
    if not PDF_PATH.exists():
        print(f"FATAL: {PDF_PATH} not found. Run synth/generate_sample.py first.")
        return 1
    if not GT_PATH.exists():
        print(f"FATAL: {GT_PATH} not found.")
        return 1

    gt = json.loads(GT_PATH.read_text(encoding="utf-8"))
    canvas_w = gt["canvas"]["width"]
    gt_counts: dict[str, int] = gt["counts_by_class"]
    mask_regions = []
    for entry in gt.get("mask_regions", []) or []:
        bbox = entry.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            mask_regions.append(tuple(int(v) for v in bbox))

    print(f"Rendering {PDF_PATH.name} at {canvas_w}px wide (matches GT canvas)...")
    page = _render_page_at_canvas_size(PDF_PATH, canvas_w)
    print(f"  rendered: {page.size}")

    print(f"Running CV symbol detection (mask_regions={len(mask_regions)})...")
    t0 = time.perf_counter()
    detections = detect_symbols(
        page, SYMBOL_DIR, source_page=1, mask_regions=mask_regions,
    )
    elapsed = time.perf_counter() - t0
    print(f"  took {elapsed:.2f}s, {len(detections)} detections after NMS")

    det_counts = Counter(d.symbol_class for d in detections)

    # ---- per-class comparison ----
    all_classes = sorted(set(gt_counts) | set(det_counts))
    print()
    print(f"{'class':24s} {'GT':>4s} {'det':>4s} {'diff':>6s} {'recall':>8s}")
    print("-" * 52)
    total_gt = 0
    total_det = 0
    total_correct_recall = 0
    for cls in all_classes:
        g = gt_counts.get(cls, 0)
        d = det_counts.get(cls, 0)
        diff = d - g
        # treat min(d, g) as the recallable subset (no IoU here; counts only)
        correct = min(d, g)
        recall = (correct / g) if g > 0 else 0.0
        total_gt += g
        total_det += d
        total_correct_recall += correct
        print(f"{cls:24s} {g:>4d} {d:>4d} {diff:>+6d} {recall:>7.0%}")
    print("-" * 52)
    overall_recall = (total_correct_recall / total_gt) if total_gt else 0.0
    overcount = total_det - total_gt
    print(f"{'TOTAL':24s} {total_gt:>4d} {total_det:>4d} {overcount:>+6d} {overall_recall:>7.0%}")
    print()

    # ---- overlay ----
    overlay = draw_detections(page, detections, show_score=True)
    overlay.save(OUT_OVERLAY, "PNG")
    print(f"Overlay saved -> {OUT_OVERLAY}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
