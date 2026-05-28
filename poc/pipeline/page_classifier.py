"""Claude vision classifier: for each page of an uploaded PDF, return what
KIND of sheet it is (cover / demolition_plan / electrical_plan / legend /
notes / etc). The Streamlit UI uses this to auto-suggest which page is the
plan (for Tab 1's run) and which is the legend (for Tab 6's ingest).

Defensible vs. the prior-work email: we're classifying SHEET METADATA, not
reading the drawing's symbols. CV + RAG still do the actual symbol counting.

The classifier renders each page at a small thumbnail size (max 800 px wide
to keep token cost down), sends them all in ONE Claude vision call, and
parses strict JSON back.
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import time
from pathlib import Path
from typing import Optional

import pypdfium2 as pdfium
from anthropic import Anthropic, APIError, APITimeoutError
from PIL import Image

from .schemas import PageClassification, PDFClassification


DEFAULT_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")
THUMB_MAX_LONG_EDGE = 800
DEFAULT_TIMEOUT_S = 60
DEFAULT_MAX_TOKENS = 1500

VALID_SHEET_CLASSES = {
    "cover", "demolition_plan", "electrical_plan", "floor_plan",
    "legend", "title_block", "notes", "other",
}


_SYSTEM_PROMPT = """You are a construction-drawing sheet classifier. You will see \
the rendered pages of a single PDF drawing set. For each page, classify it \
as one of these sheet types:

  - "cover":            Title/cover page, sheet list, project overview
  - "demolition_plan":  Existing-conditions / demolition floor plan
  - "electrical_plan":  Proposed electrical layout with placed symbols
  - "floor_plan":       General floor plan (architectural, not specifically electrical)
  - "legend":           Symbol library / legend table / key
  - "title_block":      Standalone title block / project info sheet
  - "notes":            General notes, specifications, schedules
  - "other":            None of the above

Also identify:
  - suggested_plan_page: 1-indexed page that is the most likely target for
    symbol detection (an electrical_plan, or a floor_plan if no electrical
    one exists). null if none.
  - suggested_legend_page: 1-indexed page that contains a symbol legend
    suitable for ingestion. null if none.

Return ONLY this JSON object, no prose:

{
  "pages": [
    {"page": 1, "sheet_class": "cover", "confidence": 0.95, "reasoning": "short reason"},
    {"page": 2, "sheet_class": "demolition_plan", "confidence": 0.9, "reasoning": "..."},
    ...
  ],
  "suggested_plan_page": 3,
  "suggested_legend_page": 4
}
"""


def _thumbnail(img: Image.Image, max_long_edge: int = THUMB_MAX_LONG_EDGE) -> Image.Image:
    long_edge = max(img.width, img.height)
    if long_edge <= max_long_edge:
        return img
    ratio = max_long_edge / long_edge
    return img.resize((int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)


def _pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=85)
    return base64.standard_b64encode(buf.getvalue()).decode("ascii")


def _render_all_pages(pdf_path: Path, max_long_edge: int) -> list[Image.Image]:
    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        images: list[Image.Image] = []
        for i in range(len(pdf)):
            page = pdf[i]
            page_w_pt = page.get_size()[0]
            scale = max_long_edge / page_w_pt  # rough; thumbnail step caps it
            scale = min(scale, 2.0)  # don't bloat tiny PDFs
            img = page.render(scale=scale).to_pil()
            images.append(_thumbnail(img, max_long_edge))
        return images
    finally:
        pdf.close()


def _parse_json(text: str) -> dict:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return json.loads(fenced.group(1))
    first = text.find("{")
    last = text.rfind("}")
    if first == -1 or last == -1:
        raise ValueError("No JSON object in classifier response")
    return json.loads(text[first : last + 1])


def classify_pdf(
    pdf_path: Path,
    *,
    model: str = DEFAULT_MODEL,
    max_long_edge: int = THUMB_MAX_LONG_EDGE,
) -> PDFClassification:
    """One Claude vision call to classify every page of a PDF.

    Returns a PDFClassification with per-page labels + suggested plan and
    legend pages. On any error, returns a struct with `.error` set and
    empty pages — caller falls back to "let user pick the page manually".
    """
    try:
        page_images = _render_all_pages(pdf_path, max_long_edge)
    except Exception as exc:  # noqa: BLE001
        return PDFClassification(
            source_pdf=pdf_path.name, pages=[],
            suggested_plan_page=None, suggested_legend_page=None,
            error=f"Could not render PDF: {exc}",
        )

    if not page_images:
        return PDFClassification(
            source_pdf=pdf_path.name, pages=[],
            suggested_plan_page=None, suggested_legend_page=None,
            error="PDF has no pages.",
        )

    # Build the content blocks: one image per page + a short prompt
    content_blocks: list[dict] = []
    for idx, img in enumerate(page_images, start=1):
        content_blocks.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": _pil_to_b64(img),
            },
        })
        content_blocks.append({
            "type": "text",
            "text": f"^ that is page {idx} of {len(page_images)}.",
        })
    content_blocks.append({
        "type": "text",
        "text": "Now produce the JSON classification described in your instructions.",
    })

    client = Anthropic(timeout=DEFAULT_TIMEOUT_S)
    raw_text = ""
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=DEFAULT_MAX_TOKENS,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": content_blocks}],
        )
        raw_text = "".join(
            b.text for b in resp.content if getattr(b, "type", None) == "text"
        )
    except (APITimeoutError, APIError) as exc:
        return PDFClassification(
            source_pdf=pdf_path.name, pages=[],
            suggested_plan_page=None, suggested_legend_page=None,
            error=f"{type(exc).__name__}: {exc}",
        )
    except Exception as exc:  # noqa: BLE001
        return PDFClassification(
            source_pdf=pdf_path.name, pages=[],
            suggested_plan_page=None, suggested_legend_page=None,
            error=f"{type(exc).__name__}: {exc}",
        )

    try:
        parsed = _parse_json(raw_text)
    except (ValueError, json.JSONDecodeError) as exc:
        return PDFClassification(
            source_pdf=pdf_path.name, pages=[],
            suggested_plan_page=None, suggested_legend_page=None,
            error=f"JSON parse failed: {exc}",
        )

    pages: list[PageClassification] = []
    for item in parsed.get("pages") or []:
        try:
            p_idx = int(item.get("page"))
        except (TypeError, ValueError):
            continue
        sheet = str(item.get("sheet_class", "other")).strip().lower()
        if sheet not in VALID_SHEET_CLASSES:
            sheet = "other"
        try:
            conf = max(0.0, min(1.0, float(item.get("confidence", 0.5))))
        except (TypeError, ValueError):
            conf = 0.5
        pages.append(PageClassification(
            page_index=p_idx,
            sheet_class=sheet,
            confidence=conf,
            reasoning=str(item.get("reasoning", "")).strip()[:240],
        ))

    def _to_opt_int(v) -> Optional[int]:
        try:
            return int(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    return PDFClassification(
        source_pdf=pdf_path.name,
        pages=pages,
        suggested_plan_page=_to_opt_int(parsed.get("suggested_plan_page")),
        suggested_legend_page=_to_opt_int(parsed.get("suggested_legend_page")),
        error=None,
    )
