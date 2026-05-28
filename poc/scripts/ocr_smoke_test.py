"""Phase 3 smoke test for the Tesseract OCR pass.

Renders the synthetic PDF at GT canvas size, runs OCR, and:
  - prints each extracted text span with bbox + confidence
  - saves samples/synthetic_layout_ocr.png with OCR bboxes drawn

Run from poc/:
    python scripts/ocr_smoke_test.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
POC_ROOT = HERE.parent
if str(POC_ROOT) not in sys.path:
    sys.path.insert(0, str(POC_ROOT))

from pipeline.ocr import (        # noqa: E402
    TesseractNotInstalledError,
    extract_text_spans,
)


PROJECT_ROOT = POC_ROOT.parent
PDF_PATH = PROJECT_ROOT / "samples" / "synthetic_layout.pdf"
GT_PATH = PROJECT_ROOT / "samples" / "synthetic_layout_ground_truth.json"
OUT_OVERLAY = PROJECT_ROOT / "samples" / "synthetic_layout_ocr.png"


def _render_page_at_canvas_size(pdf_path: Path, canvas_w: int) -> Image.Image:
    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        page = pdf[0]
        page_w_pt = page.get_size()[0]
        scale = canvas_w / page_w_pt
        return page.render(scale=scale).to_pil()
    finally:
        pdf.close()


def _draw_overlay(page: Image.Image, spans) -> Image.Image:
    canvas = page.convert("RGBA").copy()
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    try:
        font = ImageFont.truetype("arial.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
    color = (20, 132, 220)  # blue for OCR
    for s in spans:
        x0, y0, x1, y1 = s.bbox
        draw.rectangle((x0, y0, x1, y1), outline=color + (255,), width=2)
        label = f"{s.text} ({s.confidence:.2f})"
        try:
            bb = draw.textbbox((0, 0), label, font=font)
            tw, th = bb[2] - bb[0], bb[3] - bb[1]
        except AttributeError:
            tw, th = font.getsize(label)
        pad = 2
        ly1 = max(0, y0 - th - 2 * pad)
        draw.rectangle((x0, ly1, x0 + tw + 2 * pad, ly1 + th + 2 * pad),
                       fill=color + (200,))
        draw.text((x0 + pad, ly1 + pad), label,
                  fill=(255, 255, 255, 255), font=font)
    return Image.alpha_composite(canvas, overlay).convert("RGB")


def main() -> int:
    if not PDF_PATH.exists():
        print(f"FATAL: {PDF_PATH} not found. Run synth/generate_sample.py first.")
        return 1
    gt = json.loads(GT_PATH.read_text(encoding="utf-8"))
    canvas_w = gt["canvas"]["width"]

    print(f"Rendering {PDF_PATH.name} at {canvas_w}px wide ...")
    page = _render_page_at_canvas_size(PDF_PATH, canvas_w)
    print(f"  rendered: {page.size}")

    print("Running OCR (Tesseract --psm 11 sparse-text) ...")
    t0 = time.perf_counter()
    try:
        spans = extract_text_spans(page, source_page=1)
    except TesseractNotInstalledError as exc:
        print(f"FATAL: {exc}")
        return 2
    elapsed = time.perf_counter() - t0
    print(f"  took {elapsed:.2f}s, {len(spans)} spans after filtering")
    print()

    # ordered top-to-bottom for readability
    spans_sorted = sorted(spans, key=lambda s: (s.bbox[1], s.bbox[0]))
    print(f"{'conf':>5s}  {'bbox':>23s}   text")
    print("-" * 70)
    for s in spans_sorted:
        bbox_str = f"({s.bbox[0]},{s.bbox[1]})-({s.bbox[2]},{s.bbox[3]})"
        print(f"{s.confidence:>5.2f}  {bbox_str:>23s}   {s.text!r}")
    print()

    overlay = _draw_overlay(page, spans)
    overlay.save(OUT_OVERLAY, "PNG")
    print(f"Overlay saved -> {OUT_OVERLAY}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
