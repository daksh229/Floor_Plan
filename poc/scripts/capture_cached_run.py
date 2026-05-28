"""Capture one clean end-to-end pipeline run for the Streamlit demo fallback.

Runs the full pipeline against samples/synthetic_layout.pdf and persists the
result + rendered page image to poc/cached_runs/synthetic/. The Streamlit
app loads this by default so the demo starts in a known-good state and has
something to fall back to if the live Claude call fails on demo day.

Run from poc/ (requires ANTHROPIC_API_KEY in poc/.env):
    python scripts/capture_cached_run.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
POC_ROOT = HERE.parent
if str(POC_ROOT) not in sys.path:
    sys.path.insert(0, str(POC_ROOT))

load_dotenv(POC_ROOT / ".env")

from pipeline.rag import SymbolRAG    # noqa: E402
from pipeline.runner import (          # noqa: E402
    run_pipeline,
    save_pipeline_run,
)


PROJECT_ROOT = POC_ROOT.parent
SAMPLE_PDF = PROJECT_ROOT / "samples" / "synthetic_layout.pdf"
SYMBOL_DIR = POC_ROOT / "symbol_library"
CACHE_DIR = POC_ROOT / "cached_runs" / "synthetic"
AUDIT_DIR = POC_ROOT / "audit_store"


def _progress(stage: str, frac: float) -> None:
    print(f"  [{frac:3.0%}] {stage}")


def main() -> int:
    if not os.environ.get("ANTHROPIC_API_KEY", "").strip():
        print("FATAL: ANTHROPIC_API_KEY is not set in poc/.env")
        return 2
    if not SAMPLE_PDF.exists():
        print(f"FATAL: {SAMPLE_PDF} not found. Run synth/generate_sample.py first.")
        return 1

    print(f"Building RAG index (uses cached embeddings if present) ...")
    t = time.perf_counter()
    rag = SymbolRAG()
    rag.build()
    print(f"  RAG ready in {time.perf_counter()-t:.2f}s")

    print(f"Running pipeline against {SAMPLE_PDF.name} ...")
    run, page, diagnostics = run_pipeline(
        SAMPLE_PDF,
        symbol_dir=SYMBOL_DIR,
        rag=rag,
        audit_dir=AUDIT_DIR,
        progress=_progress,
    )
    print(f"Pipeline done in {run.elapsed_seconds}s")
    print(f"  detections: {len(run.detections)}")
    print(f"  ocr spans : {len(run.ocr_spans)}")
    print(f"  bom items : {len(run.bom.line_items)}")
    print(f"  subtotal  : AUD {run.bom.subtotal_aud:.2f}")
    if diagnostics.get("claude_error"):
        print(f"  WARNING: claude error -> {diagnostics['claude_error']}")

    save_pipeline_run(run, page, CACHE_DIR, source_pdf_path=SAMPLE_PDF)
    print(f"\nCached run saved -> {CACHE_DIR}")
    print(f"  run.json   ({(CACHE_DIR / 'run.json').stat().st_size // 1024} KB)")
    print(f"  page.png   ({(CACHE_DIR / 'page.png').stat().st_size // 1024} KB)")
    print(f"  source.pdf ({(CACHE_DIR / 'source.pdf').stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
