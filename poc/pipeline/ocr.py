"""Tesseract OCR pass over a page image, returning filtered OCRSpan list.

Tesseract is trained on prose, not drawing labels, so we filter aggressively:
  - drop spans below a confidence floor
  - drop very-short tokens (single chars are often noise from symbol glyphs)
  - drop spans whose bbox height is implausibly small

Tesseract on Windows is installed as a system binary, not a pip package. This
module auto-detects common install locations and sets
`pytesseract.tesseract_cmd` so callers don't have to.

Used by:
  - Phase 4 (BOM assembly): panel/circuit labels and free-text annotations
    enrich the BOM context that Claude sees.
  - Phase 5 (Streamlit UI): a separate OCR panel makes the OCR step visible.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import pytesseract
from PIL import Image

from .schemas import OCRSpan


# Default filters tuned for drawing-style pages
MIN_CONFIDENCE: float = 0.40         # tesseract's own 0-100, normalised
MIN_TEXT_LENGTH: int = 2             # drop bare letters like "S", "F"
MIN_TEXT_HEIGHT_PX: int = 10         # drop dust-speck false positives

# Common Windows install paths for tesseract.exe (most-likely first)
_WIN_CANDIDATES = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
    os.path.expandvars(r"%LOCALAPPDATA%\Tesseract-OCR\tesseract.exe"),
)


def _resolve_tesseract_cmd() -> str | None:
    """Locate tesseract.exe on Windows; respect TESSERACT_CMD env var override."""
    override = os.environ.get("TESSERACT_CMD")
    if override and Path(override).exists():
        return override
    for candidate in _WIN_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    return None


def _ensure_configured() -> None:
    if os.name != "nt":
        return  # Unix/Mac usually have tesseract on PATH
    if getattr(_ensure_configured, "_done", False):
        return
    cmd = _resolve_tesseract_cmd()
    if cmd is not None:
        pytesseract.pytesseract.tesseract_cmd = cmd
    _ensure_configured._done = True  # type: ignore[attr-defined]


class TesseractNotInstalledError(RuntimeError):
    pass


def extract_text_spans(
    page_image: Image.Image,
    *,
    source_page: int = 1,
    min_confidence: float = MIN_CONFIDENCE,
    min_text_length: int = MIN_TEXT_LENGTH,
    min_text_height_px: int = MIN_TEXT_HEIGHT_PX,
) -> list[OCRSpan]:
    """Run Tesseract over the page and return filtered OCRSpan objects.

    Raises TesseractNotInstalledError on Windows if the binary cannot be located.
    """
    _ensure_configured()

    try:
        # PSM 11 = sparse text, ideal for drawings where text is scattered
        data = pytesseract.image_to_data(
            page_image,
            config="--psm 11",
            output_type=pytesseract.Output.DICT,
        )
    except pytesseract.TesseractNotFoundError as exc:
        raise TesseractNotInstalledError(
            "Tesseract binary not found. Install from "
            "https://github.com/UB-Mannheim/tesseract/wiki or set the "
            "TESSERACT_CMD environment variable to the full path of tesseract.exe."
        ) from exc

    spans: list[OCRSpan] = []
    n = len(data.get("text", []))
    for i in range(n):
        text = (data["text"][i] or "").strip()
        if not text or len(text) < min_text_length:
            continue
        try:
            conf_int = int(float(data["conf"][i]))
        except (TypeError, ValueError):
            continue
        if conf_int < 0:
            continue  # tesseract uses -1 for non-text rows in the DataFrame
        confidence = conf_int / 100.0
        if confidence < min_confidence:
            continue
        h = int(data["height"][i])
        if h < min_text_height_px:
            continue
        x = int(data["left"][i])
        y = int(data["top"][i])
        w = int(data["width"][i])
        spans.append(
            OCRSpan(
                text=text,
                bbox=(x, y, x + w, y + h),
                confidence=round(confidence, 3),
                source_page=source_page,
            )
        )
    return spans


def join_spans(spans: Iterable[OCRSpan], sep: str = " | ") -> str:
    """Compact one-line text dump of all spans, ordered top-to-bottom."""
    ordered = sorted(spans, key=lambda s: (s.bbox[1], s.bbox[0]))
    return sep.join(s.text for s in ordered)
