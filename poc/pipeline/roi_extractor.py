"""Claude vision: given a rendered page of a construction drawing, return
the bounding box of just the floor-plan / electrical-plan content,
excluding title block, legend, notes, property boundary diagrams, etc.

Why: running CV on the FULL page picks up structural false positives from
the title block (rectangles → distribution_board), the in-page legend
(every symbol gets detected once as 'a real placement'), and the notes
section (text boxes → exhaust_fan). Cropping to just the plan region
collapses all of those.

Pipeline integration (in runner.py):
  1. roi = extract_plan_roi(page_image)
  2. if roi.has_roi: cv_target = page.crop(roi.bbox)
  3. detections = detect_symbols(cv_target, ...)
  4. translate every detection.bbox by roi.offset_xy back to full-page coords
  5. overlay drawn on full page (not crop) so all the visual context stays

Defensible vs. the email: Claude is selecting the *content region*, not
counting symbols. CV + RAG still do the counting.
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
from typing import Optional

from anthropic import Anthropic, APIError, APITimeoutError
from PIL import Image

from .schemas import ROIResult


DEFAULT_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")
ROI_INPUT_MAX_LONG_EDGE = 1600  # downsample what Claude sees; we still crop the high-res original
DEFAULT_TIMEOUT_S = 60
DEFAULT_MAX_TOKENS = 800


_SYSTEM_PROMPT = """You are a construction-drawing region-of-interest selector.

You will see ONE page of a drawing. Return the bounding box of the MAIN \
FLOOR-PLAN or ELECTRICAL-PLAN content area — the part with placed symbols \
and rooms — EXCLUDING:

  - the title block (project info, date, drawing number, revision strip)
  - any standalone legend or symbol-library box inside the plan area
  - 'POC EXTRACTION NOTES' / 'IMPORTANT NOTES' / similar prose blocks
  - property-boundary or context diagrams
  - adjacent or neighbouring-property massing
  - blank margins

Coordinates are in pixels of the image you are looking at; origin is \
top-left. If you can't confidently locate a single plan region, return \
bbox=null.

Return ONLY this JSON object:

{
  "bbox": [x0, y0, x1, y1] | null,
  "confidence": <0..1>,
  "reasoning": "<one sentence>"
}
"""


def _downsample(img: Image.Image, max_long_edge: int) -> tuple[Image.Image, float]:
    """Return (downsampled_image, scale_to_original).

    Coordinates Claude returns are in downsampled-image px; multiply by
    scale_to_original to map back to the source page px.
    """
    long_edge = max(img.width, img.height)
    if long_edge <= max_long_edge:
        return img, 1.0
    ratio = max_long_edge / long_edge
    new_size = (int(img.width * ratio), int(img.height * ratio))
    return img.resize(new_size, Image.LANCZOS), (1.0 / ratio)


def _pil_to_b64_jpeg(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=85)
    return base64.standard_b64encode(buf.getvalue()).decode("ascii")


def _parse_json(text: str) -> dict:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return json.loads(fenced.group(1))
    first = text.find("{")
    last = text.rfind("}")
    if first == -1 or last == -1:
        raise ValueError("No JSON object in ROI response")
    return json.loads(text[first : last + 1])


def extract_plan_roi(
    page_image: Image.Image,
    *,
    page_index: int = 1,
    model: str = DEFAULT_MODEL,
    max_long_edge: int = ROI_INPUT_MAX_LONG_EDGE,
) -> ROIResult:
    full_w, full_h = page_image.width, page_image.height
    downsampled, scale_to_orig = _downsample(page_image, max_long_edge)
    img_b64 = _pil_to_b64_jpeg(downsampled)

    user_text = (
        f"Page rendered at {downsampled.width} x {downsampled.height} px. "
        "Return JSON per system instructions."
    )

    client = Anthropic(timeout=DEFAULT_TIMEOUT_S)
    raw_text = ""
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=DEFAULT_MAX_TOKENS,
            system=_SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64",
                                                  "media_type": "image/jpeg",
                                                  "data": img_b64}},
                    {"type": "text", "text": user_text},
                ],
            }],
        )
        raw_text = "".join(
            b.text for b in resp.content if getattr(b, "type", None) == "text"
        )
    except (APITimeoutError, APIError) as exc:
        return ROIResult(
            page_index=page_index, page_width=full_w, page_height=full_h,
            bbox=None, confidence=0.0, reasoning="",
            error=f"{type(exc).__name__}: {exc}",
        )
    except Exception as exc:  # noqa: BLE001
        return ROIResult(
            page_index=page_index, page_width=full_w, page_height=full_h,
            bbox=None, confidence=0.0, reasoning="",
            error=f"{type(exc).__name__}: {exc}",
        )

    try:
        parsed = _parse_json(raw_text)
    except (ValueError, json.JSONDecodeError) as exc:
        return ROIResult(
            page_index=page_index, page_width=full_w, page_height=full_h,
            bbox=None, confidence=0.0, reasoning="",
            error=f"JSON parse failed: {exc}",
        )

    raw_bbox = parsed.get("bbox")
    if raw_bbox is None or not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
        # Claude said "I can't find one" — fall through to no-ROI behavior
        return ROIResult(
            page_index=page_index, page_width=full_w, page_height=full_h,
            bbox=None,
            confidence=float(parsed.get("confidence", 0.0) or 0.0),
            reasoning=str(parsed.get("reasoning", ""))[:300],
            error=None,
        )

    try:
        x0, y0, x1, y1 = [float(v) for v in raw_bbox]
    except (TypeError, ValueError):
        return ROIResult(
            page_index=page_index, page_width=full_w, page_height=full_h,
            bbox=None, confidence=0.0, reasoning="bbox coords not numeric",
            error="malformed bbox",
        )

    # Map from downsampled-image px back to full-page px
    x0 = max(0, int(round(x0 * scale_to_orig)))
    y0 = max(0, int(round(y0 * scale_to_orig)))
    x1 = min(full_w, int(round(x1 * scale_to_orig)))
    y1 = min(full_h, int(round(y1 * scale_to_orig)))

    # Sanity checks: positive area, not the entire page, not absurdly small
    if x1 <= x0 + 20 or y1 <= y0 + 20:
        return ROIResult(
            page_index=page_index, page_width=full_w, page_height=full_h,
            bbox=None, confidence=float(parsed.get("confidence", 0.0) or 0.0),
            reasoning="degenerate bbox returned",
            error="degenerate bbox",
        )
    area_ratio = ((x1 - x0) * (y1 - y0)) / (full_w * full_h)
    if area_ratio > 0.96:
        # Claude returned the whole page — treat as "no useful ROI"
        return ROIResult(
            page_index=page_index, page_width=full_w, page_height=full_h,
            bbox=None,
            confidence=float(parsed.get("confidence", 0.0) or 0.0),
            reasoning="bbox covers whole page",
            error=None,
        )

    return ROIResult(
        page_index=page_index, page_width=full_w, page_height=full_h,
        bbox=(x0, y0, x1, y1),
        confidence=max(0.0, min(1.0, float(parsed.get("confidence", 0.5) or 0.5))),
        reasoning=str(parsed.get("reasoning", ""))[:300],
        error=None,
    )
