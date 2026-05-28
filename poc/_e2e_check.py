"""Verify scale-0.30 + lower threshold + tight-crop + noise-strip improvements."""
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dotenv import load_dotenv
load_dotenv(HERE / ".env")

from pipeline.rag import SymbolRAG
from pipeline.runner import run_pipeline

PDF = HERE.parent / "samples" / "synthetic_procalc_style_electrical_set_variant_b.pdf"

rag = SymbolRAG(); rag.build()
print(f"RAG entries: {len(rag.entries)}\n")

def _p(stage, frac):
    print(f"  [{frac:>5.0%}] {stage}")

run, page, diag = run_pipeline(
    PDF, rag=rag,
    audit_dir=HERE / "audit_store",
    progress=_p, symbol_pack="user", pdf_page_index=3,
)
classes_in_pack = {p.stem for p in (HERE / "symbol_library" / "user").glob("*.png")}
detected_classes = set(d.symbol_class for d in run.detections)
print(f"\nUser-pack classes available: {len(classes_in_pack)}")
print(f"Detections: {len(run.detections)}")
print(f"Distinct classes detected: {len(detected_classes)} of {len(classes_in_pack)}")
print(f"Missing classes: {sorted(classes_in_pack - detected_classes)}")
print(f"BOM lines: {len(run.bom.line_items)}")
print(f"Subtotal: AUD {run.bom.subtotal_aud:.2f}")
print(f"\nCounts by class:")
for k, v in sorted(Counter(d.symbol_class for d in run.detections).items()):
    print(f"  {k:24s} {v}")
print(f"\nElapsed: {run.elapsed_seconds}s")
