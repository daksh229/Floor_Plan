"""BOM assembly: deterministic aggregation + Claude as editorialiser.

The architecture is deliberate. Quantities, specs, and costs are computed
deterministically from (Detections, library.json). Claude's job is narrower
than the sales email implies: it extracts project metadata from OCR text and
writes a short estimator-facing note. Anything Claude returns that conflicts
with the deterministic source-of-truth is silently overwritten — Claude
never gets the chance to invent a $1000 wiring line item that wasn't in the
detections.

This is the only design that survives the "Claude hallucinated counts /
invented specs" failure mode from the challenges discussion.

Pipeline:
  1. compute_aggregated_payload(detections, ocr_spans, rag_matches)
       -> deterministic per-class aggregation with library values
  2. assemble_bom(payload)
       -> calls Claude; validates + corrects qty/cost/spec against payload
  3. _write_audit(...) saves the request + raw response + final BOM
"""
from __future__ import annotations

import base64
import datetime as dt
import io
import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Iterable, Optional

from anthropic import Anthropic, APIError, APITimeoutError
from PIL import Image

from .prompts import DEFAULT_VERSION, PromptVersion, get_prompt
from .rag import SymbolRAG
from .schemas import (
    BOM,
    BOMLineItem,
    Detection,
    OCRSpan,
    SymbolWithMatches,
)


DEFAULT_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")
DEFAULT_MAX_TOKENS = 3000
DEFAULT_TIMEOUT_S = 90
DEFAULT_RETRIES = 2


# ---------- deterministic aggregation ----------

def _confidence_band(avg_score: float) -> str:
    if avg_score >= 0.80:
        return "high"
    if avg_score >= 0.65:
        return "medium"
    return "low"


def compute_aggregated_payload(
    detections: Iterable[Detection],
    rag: SymbolRAG,
    ocr_spans: Iterable[OCRSpan] | None = None,
    pdf_name: str = "",
    page_count: int = 1,
) -> dict[str, Any]:
    """Group detections by symbol_class, resolve each to its top-1 library
    entry via RAG, and emit the payload that the prompt renderer consumes.
    The same payload is the source of truth for post-validation."""
    detections = list(detections)
    spans = list(ocr_spans or [])

    # Counts per class, with source-page tracking
    by_class: dict[str, list[Detection]] = {}
    for d in detections:
        by_class.setdefault(d.symbol_class, []).append(d)

    aggregated: list[dict[str, Any]] = []
    for symbol_class, dets in sorted(by_class.items()):
        matches = rag.match(symbol_class, k=1)
        if not matches:
            continue
        m = matches[0]
        scores = [d.score for d in dets]
        avg_s = round(sum(scores) / len(scores), 4) if scores else 0.0
        min_s = round(min(scores), 4) if scores else 0.0
        aggregated.append(
            {
                "library_key": m.library_key,
                "canonical_name": m.canonical_name,
                "quantity": len(dets),
                "unit": m.unit,
                "unit_cost_aud": m.indicative_cost_aud,
                "spec": m.spec,
                "source_pages": sorted({d.source_page for d in dets}),
                "source_detection_count": len(dets),
                "match_similarity": round(m.similarity, 4),
                "avg_detection_score": avg_s,
                "min_detection_score": min_s,
                "confidence_band": _confidence_band(avg_s),
            }
        )

    ocr_text_joined = " | ".join(
        s.text for s in sorted(spans, key=lambda s: (s.bbox[1], s.bbox[0]))
    )

    return {
        "aggregated": aggregated,
        "ocr_text_joined": ocr_text_joined,
        "pdf_name": pdf_name,
        "page_count": page_count,
    }


# ---------- Claude call ----------

def _parse_json_block(text: str) -> dict:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return json.loads(fenced.group(1))
    first = text.find("{")
    last = text.rfind("}")
    if first == -1 or last == -1:
        raise ValueError("No JSON object found in response")
    return json.loads(text[first : last + 1])


def _call_claude(prompt: PromptVersion, payload: dict, model: str) -> tuple[Optional[dict], Optional[str], str]:
    """Returns (parsed_json or None, error str or None, raw_response_text)."""
    client = Anthropic(timeout=DEFAULT_TIMEOUT_S)
    user_text = prompt.user_renderer(payload)
    last_exc: Optional[Exception] = None
    raw_text = ""

    for attempt in range(DEFAULT_RETRIES + 1):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=DEFAULT_MAX_TOKENS,
                system=prompt.system,
                messages=[{"role": "user", "content": user_text}],
            )
            raw_text = "".join(
                b.text for b in resp.content if getattr(b, "type", None) == "text"
            )
            break
        except (APITimeoutError, APIError) as exc:
            last_exc = exc
            if attempt < DEFAULT_RETRIES:
                time.sleep(1.5 * (attempt + 1))
                continue
            return None, f"{type(exc).__name__}: {exc}", raw_text
        except Exception as exc:  # noqa: BLE001
            return None, f"{type(exc).__name__}: {exc}", raw_text

    if not raw_text:
        return None, f"Empty response (last error: {last_exc})", raw_text

    try:
        parsed = _parse_json_block(raw_text)
    except (ValueError, json.JSONDecodeError) as exc:
        return None, f"JSON parse failed: {exc}", raw_text
    return parsed, None, raw_text


# ---------- assembly with validation ----------

def _build_deterministic_bom(
    payload: dict[str, Any],
    prompt_version: str,
    model: str,
    metadata: dict[str, Any] | None = None,
    notes: str | None = None,
) -> BOM:
    """The source-of-truth BOM, computed entirely from the payload."""
    line_items: list[BOMLineItem] = []
    subtotal = 0.0
    for row in payload["aggregated"]:
        qty = int(row["quantity"])
        unit_cost = float(row["unit_cost_aud"])
        line_total = round(qty * unit_cost, 2)
        subtotal += line_total
        avg_s = float(row.get("avg_detection_score", 0.0))
        min_s = float(row.get("min_detection_score", 0.0))
        line_items.append(
            BOMLineItem(
                library_key=row["library_key"],
                canonical_name=row["canonical_name"],
                quantity=qty,
                unit=row["unit"],
                spec=row["spec"],
                unit_cost_aud=unit_cost,
                line_total_aud=line_total,
                source_pages=list(row["source_pages"]),
                source_detection_count=int(row["source_detection_count"]),
                avg_detection_score=avg_s,
                min_detection_score=min_s,
                confidence_band=str(row.get("confidence_band", _confidence_band(avg_s))),
                notes=None,
            )
        )
    md = metadata or {}
    return BOM(
        project_name=md.get("project_name"),
        drawing_number=md.get("drawing_number"),
        revision=md.get("revision"),
        drawing_date=md.get("drawing_date"),
        line_items=line_items,
        subtotal_aud=round(subtotal, 2),
        notes_from_assembler=notes,
        prompt_version=prompt_version,
        model=model,
        page_count=int(payload.get("page_count", 1)),
        generated_at=dt.datetime.now(dt.timezone.utc).isoformat(),
    )


def assemble_bom(
    payload: dict[str, Any],
    *,
    prompt_version: str = DEFAULT_VERSION,
    model: str = DEFAULT_MODEL,
    audit_dir: Optional[Path] = None,
) -> tuple[BOM, dict[str, Any]]:
    """Run Claude assembly, then enforce deterministic invariants.

    Returns (bom, diagnostics) where diagnostics describes any Claude
    deviations that were silently corrected — useful for the demo.
    """
    prompt = get_prompt(prompt_version)
    parsed, error, raw_text = _call_claude(prompt, payload, model)

    diagnostics: dict[str, Any] = {
        "claude_error": error,
        "claude_responded": parsed is not None,
        "qty_overrides": [],
        "cost_overrides": [],
        "spec_overrides": [],
        "spurious_line_items_dropped": [],
        "missing_line_items_added": [],
        "subtotal_overridden": False,
    }

    # Default: deterministic BOM with no metadata, no narrative
    bom = _build_deterministic_bom(payload, prompt_version, model)

    if parsed is None:
        bom.notes_from_assembler = (
            f"(Auto note) Deterministic fallback engaged: {error}. "
            "All quantities and costs are sourced from CV detections + library entries."
        )
        _write_audit(payload, raw_text, bom, diagnostics, audit_dir, error=error)
        return bom, diagnostics

    # Metadata extraction is the part Claude actually owns
    bom.project_name = _safe_str(parsed.get("project_name"))
    bom.drawing_number = _safe_str(parsed.get("drawing_number"))
    bom.revision = _safe_str(parsed.get("revision"))
    bom.drawing_date = _safe_str(parsed.get("drawing_date"))
    bom.notes_from_assembler = _safe_str(parsed.get("notes_from_assembler"))

    # Validate Claude's line_items against deterministic source-of-truth
    claude_items_by_key: dict[str, dict] = {}
    for item in parsed.get("line_items") or []:
        key = item.get("library_key")
        if isinstance(key, str):
            claude_items_by_key[key] = item

    truth_keys = {li.library_key for li in bom.line_items}
    for ckey in list(claude_items_by_key):
        if ckey not in truth_keys:
            diagnostics["spurious_line_items_dropped"].append(ckey)

    for li in bom.line_items:
        c = claude_items_by_key.get(li.library_key)
        if c is None:
            diagnostics["missing_line_items_added"].append(li.library_key)
            continue
        try:
            c_qty = int(c.get("quantity"))
        except (TypeError, ValueError):
            c_qty = None
        try:
            c_cost = float(c.get("unit_cost_aud"))
        except (TypeError, ValueError):
            c_cost = None
        c_spec = c.get("spec")

        if c_qty is not None and c_qty != li.quantity:
            diagnostics["qty_overrides"].append(
                {"library_key": li.library_key, "claude": c_qty, "truth": li.quantity}
            )
        if c_cost is not None and abs(c_cost - li.unit_cost_aud) > 0.005:
            diagnostics["cost_overrides"].append(
                {"library_key": li.library_key, "claude": c_cost, "truth": li.unit_cost_aud}
            )
        if isinstance(c_spec, str) and c_spec != li.spec:
            diagnostics["spec_overrides"].append(li.library_key)
        # Note: we DO NOT mutate li from c. li is the truth.

    # Subtotal check
    try:
        c_subtotal = float(parsed.get("subtotal_aud"))
        if abs(c_subtotal - bom.subtotal_aud) > 0.05:
            diagnostics["subtotal_overridden"] = True
    except (TypeError, ValueError):
        pass

    _write_audit(payload, raw_text, bom, diagnostics, audit_dir, error=None)
    return bom, diagnostics


# ---------- audit ----------

def _write_audit(
    payload: dict[str, Any],
    raw_response: str,
    bom: BOM,
    diagnostics: dict[str, Any],
    audit_dir: Optional[Path],
    error: Optional[str],
) -> None:
    if audit_dir is None:
        return
    audit_dir.mkdir(parents=True, exist_ok=True)
    record_id = f"{int(time.time())}_{uuid.uuid4().hex[:8]}_bom"
    folder = audit_dir / record_id
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "payload.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (folder / "claude_raw_response.txt").write_text(raw_response or "", encoding="utf-8")
    (folder / "bom.json").write_text(bom.model_dump_json(indent=2), encoding="utf-8")
    (folder / "diagnostics.json").write_text(
        json.dumps({"error": error, **diagnostics}, indent=2), encoding="utf-8"
    )


def _safe_str(v: Any) -> str | None:
    if isinstance(v, str):
        s = v.strip()
        return s or None
    return None
