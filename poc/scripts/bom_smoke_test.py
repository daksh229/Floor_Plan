"""Phase 4 end-to-end smoke test: PDF -> CV -> OCR -> RAG -> Claude BOM.

Runs the full pipeline and prints the assembled BOM as a table, plus any
diagnostics from Claude-vs-truth validation. The final BOM JSON is written
to samples/synthetic_layout_bom.json and the audit record to
poc/audit_store/.

Run from poc/ (requires ANTHROPIC_API_KEY in poc/.env):
    python scripts/bom_smoke_test.py
    python scripts/bom_smoke_test.py --force-deviation   # demo override mechanism
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pypdfium2 as pdfium
from dotenv import load_dotenv
from PIL import Image

HERE = Path(__file__).resolve().parent
POC_ROOT = HERE.parent
if str(POC_ROOT) not in sys.path:
    sys.path.insert(0, str(POC_ROOT))

load_dotenv(POC_ROOT / ".env")

from pipeline.bom_assembler import (    # noqa: E402
    assemble_bom,
    compute_aggregated_payload,
)
from pipeline.cv_detect import detect_symbols    # noqa: E402
from pipeline.ocr import (                       # noqa: E402
    TesseractNotInstalledError,
    extract_text_spans,
)
from pipeline.prompts import (                   # noqa: E402
    DEFAULT_VERSION,
    FORCE_DEVIATION_VERSION,
)
from pipeline.rag import SymbolRAG               # noqa: E402


PROJECT_ROOT = POC_ROOT.parent
PDF_PATH = PROJECT_ROOT / "samples" / "synthetic_layout.pdf"
GT_PATH = PROJECT_ROOT / "samples" / "synthetic_layout_ground_truth.json"
SYMBOL_DIR = POC_ROOT / "symbol_library"
AUDIT_DIR = POC_ROOT / "audit_store"
BOM_OUT = PROJECT_ROOT / "samples" / "synthetic_layout_bom.json"


def _render_page_at_canvas_size(pdf_path: Path, canvas_w: int) -> Image.Image:
    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        page = pdf[0]
        page_w_pt = page.get_size()[0]
        scale = canvas_w / page_w_pt
        return page.render(scale=scale).to_pil()
    finally:
        pdf.close()


def _check_env() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY", "").strip():
        print("FATAL: ANTHROPIC_API_KEY is not set in poc/.env")
        sys.exit(2)


def _section(name: str) -> None:
    print()
    print(f"=== {name} ===")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force-deviation",
        action="store_true",
        help="Use the bom_v1_force_deviation prompt so Claude deliberately "
             "misquotes — the override mechanism will trigger and the "
             "diagnostics section will populate. For demoing the safety net.",
    )
    args = parser.parse_args()

    _check_env()

    if not PDF_PATH.exists():
        print(f"FATAL: {PDF_PATH} not found. Run synth/generate_sample.py first.")
        return 1

    gt = json.loads(GT_PATH.read_text(encoding="utf-8"))
    canvas_w = gt["canvas"]["width"]
    mask_regions = []
    for entry in gt.get("mask_regions", []) or []:
        bbox = entry.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            mask_regions.append(tuple(int(v) for v in bbox))

    prompt_version = FORCE_DEVIATION_VERSION if args.force_deviation else DEFAULT_VERSION
    if args.force_deviation:
        print("=== DEMO MODE: force-deviation prompt — override mechanism will trigger ===")

    t_total = time.perf_counter()

    # ---- 1. ingest ----
    _section("Step 1: ingest")
    page = _render_page_at_canvas_size(PDF_PATH, canvas_w)
    print(f"rendered {PDF_PATH.name} -> {page.size}")

    # ---- 2. CV detect ----
    _section("Step 2: CV symbol detection")
    t = time.perf_counter()
    detections = detect_symbols(
        page, SYMBOL_DIR, source_page=1, mask_regions=mask_regions
    )
    print(f"detect_symbols: {len(detections)} detections in {time.perf_counter()-t:.2f}s "
          f"(mask_regions={len(mask_regions)})")

    # ---- 3. OCR ----
    _section("Step 3: OCR")
    t = time.perf_counter()
    try:
        ocr_spans = extract_text_spans(page, source_page=1)
    except TesseractNotInstalledError as exc:
        print(f"OCR skipped: {exc}")
        ocr_spans = []
    print(f"extract_text_spans: {len(ocr_spans)} spans in {time.perf_counter()-t:.2f}s")

    # ---- 4. RAG ----
    _section("Step 4: RAG build + aggregation")
    t = time.perf_counter()
    rag = SymbolRAG()
    rag.build()
    payload = compute_aggregated_payload(
        detections, rag,
        ocr_spans=ocr_spans,
        pdf_name=PDF_PATH.name,
        page_count=1,
    )
    print(f"RAG build + aggregate: {time.perf_counter()-t:.2f}s")
    print(f"aggregated into {len(payload['aggregated'])} unique library entries")

    # ---- 5. BOM assembly (Claude) ----
    _section("Step 5: BOM assembly (Claude)")
    t = time.perf_counter()
    bom, diagnostics = assemble_bom(
        payload, prompt_version=prompt_version, audit_dir=AUDIT_DIR,
    )
    print(f"assemble_bom: {time.perf_counter()-t:.2f}s")
    print(f"Claude error: {diagnostics['claude_error'] or 'none'}")
    print(f"Claude responded: {diagnostics['claude_responded']}")

    if (
        diagnostics["qty_overrides"]
        or diagnostics["cost_overrides"]
        or diagnostics["spec_overrides"]
        or diagnostics["spurious_line_items_dropped"]
        or diagnostics["missing_line_items_added"]
    ):
        print("Validation findings (deterministic source-of-truth enforced):")
        if diagnostics["qty_overrides"]:
            print(f"  qty overrides:    {diagnostics['qty_overrides']}")
        if diagnostics["cost_overrides"]:
            print(f"  cost overrides:   {diagnostics['cost_overrides']}")
        if diagnostics["spec_overrides"]:
            print(f"  spec overrides:   {diagnostics['spec_overrides']}")
        if diagnostics["spurious_line_items_dropped"]:
            print(f"  spurious dropped: {diagnostics['spurious_line_items_dropped']}")
        if diagnostics["missing_line_items_added"]:
            print(f"  missing added:    {diagnostics['missing_line_items_added']}")
    else:
        print("Validation findings: none (Claude output matched deterministic truth).")

    # ---- 6. Print BOM ----
    _section("Final BOM")
    print(f"Project:        {bom.project_name or '-'}")
    print(f"Drawing no.:    {bom.drawing_number or '-'}")
    print(f"Revision:       {bom.revision or '-'}")
    print(f"Date:           {bom.drawing_date or '-'}")
    print()
    header = f"{'library_key':24s} {'qty':>4s} {'unit':>4s} {'unit AUD':>10s} {'total AUD':>11s}  {'conf':>4s} {'avg_s':>6s}"
    print(header)
    print("-" * len(header))
    for li in bom.line_items:
        print(
            f"{li.library_key:24s} "
            f"{li.quantity:>4d} "
            f"{li.unit:>4s} "
            f"{li.unit_cost_aud:>10.2f} "
            f"{li.line_total_aud:>11.2f}  "
            f"{li.confidence_band:>4s} "
            f"{li.avg_detection_score:>6.3f}"
        )
    print("-" * len(header))
    print(f"{'SUBTOTAL':24s} {'':>4s} {'':>4s} {'':>10s} {bom.subtotal_aud:>11.2f}  AUD")
    print()
    if bom.notes_from_assembler:
        print("Notes from assembler:")
        print(f"  {bom.notes_from_assembler}")
    print()
    print(f"Total elapsed: {time.perf_counter()-t_total:.2f}s")

    # ---- 7. Persist ----
    BOM_OUT.write_text(bom.model_dump_json(indent=2), encoding="utf-8")
    print(f"\nBOM JSON saved -> {BOM_OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
