"""CLI: ingest a legend page from a construction-drawing PDF into the
user-extensible symbol library.

Workflow:
  1. Extracts every legend row from the chosen page (symbol image + canonical
     name + alias + unit + indicative cost).
  2. Prints a preview table + warnings.
  3. With --dry-run, stops here.
  4. Otherwise writes the symbol crops to `symbol_library/user/<key>.png`
     and merges the metadata into `symbol_library/user_additions.json`
     (existing user keys with the same name are overwritten — "user wins").

Run from poc/:
    python scripts/ingest_legend.py --pdf <path> --page <N> [--dry-run]
    python scripts/ingest_legend.py --pdf ../samples/synthetic_procalc_style_electrical_set.pdf --page 4
    python scripts/ingest_legend.py --reset           # delete all user additions
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
POC_ROOT = HERE.parent
if str(POC_ROOT) not in sys.path:
    sys.path.insert(0, str(POC_ROOT))

from pipeline.legend_extractor import extract_legend         # noqa: E402
from pipeline.rag import (                                    # noqa: E402
    USER_ADDITIONS_PATH,
    USER_SYMBOL_DIR,
)


def _print_preview(rows, warnings) -> None:
    print(f"{'#':>2}  {'library_key':24s}  {'canonical_name':28s}  {'alias':22s}  {'unit':>4s}  {'AUD':>8s}")
    print("-" * 96)
    for i, r in enumerate(rows):
        print(f"{i:>2}  {r.library_key:24s}  {r.canonical_name[:28]:28s}  {r.alias[:22]:22s}  {r.unit:>4s}  {r.indicative_cost_aud:>8.2f}")
    if warnings:
        print()
        print("Warnings:")
        for w in warnings:
            print(f"  ! {w}")


def _row_to_library_entry(row) -> dict:
    aliases = [a.strip() for a in row.alias.split(",") if a.strip()]
    if not aliases:
        aliases = [row.canonical_name.lower()]
    return {
        "canonical_name": row.canonical_name,
        "aliases": aliases,
        "unit": row.unit or "ea",
        "spec": (f"Ingested from legend; alias phrase: '{row.alias}'."
                 if row.alias else "Ingested from legend."),
        "indicative_cost_aud": float(row.indicative_cost_aud),
        "user_added": True,
        "source_pdf": row.source_pdf,
        "source_page": row.source_page,
    }


def _write_png(row, dest_dir: Path) -> Path:
    dest = dest_dir / f"{row.library_key}.png"
    raw = base64.b64decode(row.symbol_image_b64)
    dest.write_bytes(raw)
    return dest


def _merge_into_additions(rows, source_pdf: Path, source_page: int) -> tuple[int, int]:
    """Merge rows into user_additions.json. Returns (added, overwritten)."""
    if USER_ADDITIONS_PATH.exists():
        try:
            existing = json.loads(USER_ADDITIONS_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
    else:
        existing = {}

    if "symbols" not in existing:
        existing["symbols"] = {}
    if "_meta" not in existing:
        existing["_meta"] = {}

    added = 0
    overwritten = 0
    for row in rows:
        entry = _row_to_library_entry(row)
        if row.library_key in existing["symbols"]:
            overwritten += 1
        else:
            added += 1
        existing["symbols"][row.library_key] = entry

    existing["_meta"].update({
        "version": existing["_meta"].get("version", "v1"),
        "last_ingested_from": str(source_pdf),
        "last_ingested_page": source_page,
        "last_ingested_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "total_symbols": len(existing["symbols"]),
    })

    USER_ADDITIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    USER_ADDITIONS_PATH.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    return added, overwritten


def _reset() -> int:
    deleted_pngs = 0
    if USER_SYMBOL_DIR.exists():
        for p in USER_SYMBOL_DIR.glob("*.png"):
            p.unlink()
            deleted_pngs += 1
        try:
            USER_SYMBOL_DIR.rmdir()
        except OSError:
            pass
    deleted_json = USER_ADDITIONS_PATH.exists()
    if deleted_json:
        USER_ADDITIONS_PATH.unlink()
    print(f"Removed {deleted_pngs} user PNG(s) and "
          f"{'1' if deleted_json else '0'} user_additions.json.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, help="Path to the source PDF containing the legend.")
    parser.add_argument("--page", type=int, help="1-indexed legend page in the PDF.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print preview but don't write anything.")
    parser.add_argument("--reset", action="store_true",
                        help="Delete all user additions and exit.")
    args = parser.parse_args()

    if args.reset:
        return _reset()

    if args.pdf is None or args.page is None:
        parser.error("--pdf and --page are required (or use --reset).")

    if not args.pdf.exists():
        print(f"FATAL: PDF not found: {args.pdf}")
        return 1

    print(f"Extracting legend from {args.pdf.name} page {args.page} ...")
    extraction = extract_legend(args.pdf, args.page)
    print(f"  page rendered at {extraction.page_image_width}x{extraction.page_image_height}")
    print(f"  rows extracted: {len(extraction.rows)}")
    print()
    _print_preview(extraction.rows, extraction.warnings)

    if not extraction.rows:
        print("\nNothing to ingest.")
        return 2

    if args.dry_run:
        print("\n--dry-run: no files written.")
        return 0

    USER_SYMBOL_DIR.mkdir(parents=True, exist_ok=True)
    for row in extraction.rows:
        _write_png(row, USER_SYMBOL_DIR)
    added, overwritten = _merge_into_additions(extraction.rows, args.pdf, args.page)
    print()
    print(f"Wrote {len(extraction.rows)} PNG(s) to {USER_SYMBOL_DIR}")
    print(f"Updated {USER_ADDITIONS_PATH.name}: +{added} new, ~{overwritten} overwritten")
    print("\nThe Streamlit app's sidebar should now show 'symbol pack: user' as an option.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
