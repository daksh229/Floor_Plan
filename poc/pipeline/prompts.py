"""Versioned BOM-assembly prompt store.

Prompts are data, not code — a new version is a dict entry, not a deploy.
Each version pairs a system prompt with a user-text renderer that takes
the pre-aggregated input payload and returns the final user message.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict


@dataclass(frozen=True)
class PromptVersion:
    version: str
    system: str
    user_renderer: Callable[[dict[str, Any]], str]


# -------- v1 --------

_V1_SYSTEM = """You are an electrical-trade Bill of Materials assembler for a residential \
construction estimating system. You take pre-aggregated computer-vision detections, \
predefined library entries, and OCR text from a plan drawing, then return a strict \
JSON Bill of Materials.

HARD CONSTRAINTS — your output is post-validated and any violation is silently \
overwritten with the deterministic source-of-truth, so following these rules is the \
only way your work survives:

  1. Use EXACTLY the quantities provided in the input. Never infer or adjust counts.
  2. Use EXACTLY the unit_cost_aud, unit, and spec from the matched library entry. \
     Never paraphrase the spec. Never recompute the cost.
  3. Do NOT add line items for things that are not in the aggregated input list \
     (e.g. wiring, cable trays, conduit). If you think something common is missing, \
     mention it in `notes_from_assembler` instead.
  4. Do NOT drop line items, even if quantity is small.
  5. Compute line_total_aud as quantity * unit_cost_aud. Round to 2 decimals.
  6. Compute subtotal_aud as the sum of every line_total_aud. Round to 2 decimals.

WHAT YOU SHOULD DO:
  - Extract project metadata (project_name, drawing_number, revision, drawing_date) \
    from the OCR text. Use null if absent. Do not invent.
  - In `notes_from_assembler`, write 1-3 sentences of estimator-facing commentary: \
    counts that look unusually high or low for a residential ground floor, items you'd \
    flag as commonly-missing, etc. Keep it short and useful.

OUTPUT FORMAT — return ONLY this JSON object, no prose around it:

{
  "project_name": <string or null>,
  "drawing_number": <string or null>,
  "revision": <string or null>,
  "drawing_date": <string or null>,
  "line_items": [
    {
      "library_key": <string>,
      "canonical_name": <string>,
      "quantity": <integer>,
      "unit": <string>,
      "spec": <string>,
      "unit_cost_aud": <number>,
      "line_total_aud": <number>,
      "notes": <string or null>
    },
    ...
  ],
  "subtotal_aud": <number>,
  "notes_from_assembler": <string>
}
"""


def _v1_user_renderer(payload: dict[str, Any]) -> str:
    aggregated = payload["aggregated"]
    ocr_text = payload.get("ocr_text_joined") or "(no OCR text extracted)"
    pdf_name = payload.get("pdf_name", "(unknown)")
    page_count = payload.get("page_count", 1)

    lines = [
        f"Drawing: {pdf_name} ({page_count} page(s))",
        "",
        "AGGREGATED CV DETECTIONS + LIBRARY MATCHES (your source of truth — do not deviate):",
    ]
    for entry in aggregated:
        lines.append(
            f"  - library_key: {entry['library_key']}\n"
            f"    canonical_name: {entry['canonical_name']}\n"
            f"    quantity: {entry['quantity']}\n"
            f"    unit: {entry['unit']}\n"
            f"    unit_cost_aud: {entry['unit_cost_aud']:.2f}\n"
            f"    spec: {entry['spec']}\n"
            f"    source_pages: {entry['source_pages']}"
        )
    lines.append("")
    lines.append("OCR TEXT EXTRACTED FROM PLAN (use ONLY for project metadata):")
    lines.append(ocr_text[:4000])  # cap; metadata is usually in first part
    lines.append("")
    lines.append("Return the JSON object now, with no prose around it.")
    return "\n".join(lines)


# A deliberately-misbehaving system prompt used ONLY to demonstrate the
# post-validation override mechanism on the interview call. This is a CLEAN
# REPLACEMENT for the strict primary prompt (not an append) — appending to
# the strict prompt produces a model that follows the strict rules and
# ignores the demo instruction, which defeats the demo.
_V1_FORCE_DEVIATION_SYSTEM = """You are running a *test harness* for a BOM \
assembly safety net. Your job in this test is to produce a BOM that the \
downstream deterministic validator will CATCH and CORRECT, so the safety net \
can be observed.

You will receive aggregated CV detections + library entries + OCR text. \
Produce a JSON Bill of Materials in the exact schema below, BUT with the \
following deliberate deviations from the input — the validator expects \
these and will silently correct them:

  REQUIRED DEVIATIONS (do all four — the test fails if any are missing):
  1. Pick the FIRST line item. Set its `quantity` to (input_quantity + 2).
  2. Pick the SECOND line item. Set its `unit_cost_aud` to (input_cost + 10.00).
  3. INVENT one extra line item with library_key = "cable_run" and any plausible
     numbers. The validator will detect and drop it.
  4. OMIT the LAST line item entirely from your output. The validator will
     detect and reinstate it.

Recompute line_total_aud for the mutated lines and the subtotal_aud \
consistently with your deviated values. Set `notes_from_assembler` to: \
"TEST HARNESS: deliberate deviations injected for safety-net verification."

Extract project_name, drawing_number, revision, drawing_date from the OCR \
text honestly (no deviations on metadata).

OUTPUT FORMAT — return ONLY this JSON object, no prose around it:

{
  "project_name": <string or null>,
  "drawing_number": <string or null>,
  "revision": <string or null>,
  "drawing_date": <string or null>,
  "line_items": [
    {
      "library_key": <string>,
      "canonical_name": <string>,
      "quantity": <integer>,
      "unit": <string>,
      "spec": <string>,
      "unit_cost_aud": <number>,
      "line_total_aud": <number>,
      "notes": <string or null>
    }
  ],
  "subtotal_aud": <number>,
  "notes_from_assembler": <string>
}
"""


PROMPT_REGISTRY: Dict[str, PromptVersion] = {
    "bom_v1": PromptVersion(
        version="bom_v1",
        system=_V1_SYSTEM,
        user_renderer=_v1_user_renderer,
    ),
    "bom_v1_force_deviation": PromptVersion(
        version="bom_v1_force_deviation",
        system=_V1_FORCE_DEVIATION_SYSTEM,
        user_renderer=_v1_user_renderer,
    ),
}

DEFAULT_VERSION = "bom_v1"
FORCE_DEVIATION_VERSION = "bom_v1_force_deviation"


def get_prompt(version: str = DEFAULT_VERSION) -> PromptVersion:
    if version not in PROMPT_REGISTRY:
        raise KeyError(f"Unknown prompt version: {version}. Available: {list(PROMPT_REGISTRY)}")
    return PROMPT_REGISTRY[version]
