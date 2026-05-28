"""Phase 3 smoke test for SymbolRAG.

Runs three flavours of query against the symbol-library RAG layer:

  1. Exact detection class names (what Phase 4 will actually use)
  2. Alias / paraphrase queries (probes that aliases enrich the embedding doc)
  3. Free-text queries (probes that arbitrary OCR/user text can hit the
     right entry via semantics, not just literal class name)

Output: top-3 ranked matches with cosine similarity scores for each query.

Run from poc/:
    python scripts/rag_smoke_test.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
POC_ROOT = HERE.parent
if str(POC_ROOT) not in sys.path:
    sys.path.insert(0, str(POC_ROOT))

from pipeline.rag import SymbolRAG   # noqa: E402


EXACT_CLASSES = [
    "single_pole_switch",
    "double_gpo",
    "wp_gpo",
    "distribution_board",
    "smoke_detector",
]

ALIAS_PARAPHRASES = [
    "1-gang toggle wall switch",
    "twin power point",
    "weatherproof outdoor socket",
    "main switchboard panel",
    "interconnected smoke alarm",
]

FREE_TEXT = [
    "Cat6 network outlet by the desk",
    "rotary trailing-edge dimmer for LED downlights",
    "external IP54 socket on patio wall",
    "12-pole consumer unit",
    "ceiling sweep fan with wall control",
]


def _print_block(title: str, queries: list[str], rag: SymbolRAG, k: int = 3) -> None:
    print(f"\n=== {title} ===")
    for q in queries:
        matches = rag.match(q, k=k)
        print(f'\nQUERY: "{q}"')
        for rank, m in enumerate(matches, start=1):
            print(
                f"  #{rank}  sim={m.similarity:+.3f}  "
                f"{m.library_key:24s}  {m.canonical_name}"
            )


def main() -> int:
    print("Building RAG index (first run downloads the embedding model ~80MB)...")
    t0 = time.perf_counter()
    rag = SymbolRAG(verbose=True)
    rag.build()
    print(f"  built in {time.perf_counter() - t0:.2f}s "
          f"({len(rag.entries)} library entries)")

    _print_block("Exact class names (Phase-4 use case)", EXACT_CLASSES, rag)
    _print_block("Alias / paraphrase queries", ALIAS_PARAPHRASES, rag)
    _print_block("Free-text queries", FREE_TEXT, rag)

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
