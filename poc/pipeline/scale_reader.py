"""Claude-vision scale-bar reader.

Reads the printed scale label (e.g. "1:100", "1:50") from a construction
drawing. We send a tight CROP of the upper portion of the page rather
than the whole page — keeps tokens cheap and isolates the scale-bar
glyph from the rest of the drawing.

Returns a `ScaleDetected` struct with:
  - value: e.g. "1:100"
  - source: "scale_bar"
  - confidence: how certain Claude is

Empty `value` ('' or None) when the scale isn't legible. Caller can then
fall back to the dimension-driven calibrator (Phase 2) which derives the
scale from perimeter dimensions instead of trusting a glyph.

Design choice: this is a separate small module rather than a method on a
larger class because the proposal's foundation #3 (coordinate translation)
includes scale handling, and we want one focused unit per concern.
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
from typing import Optional

from PIL import Image

from .measurement_schemas import ScaleDetected


DEFAULT_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")
DEFAULT_TIMEOUT_S = 30
DEFAULT_MAX_TOKENS = 200

# Where the scale bar typically lives on residential plans. The
# measurement-sample generator + variant_e fixture both put it in the
# top-left corner. Real construction plans usually do the same, but
# sometimes it's near the title block. Crop is generous to catch both.
# Made the top_left crop wider/taller after the first smoke test —
# the scale bar tick labels ("0 1m 2m 3m") were getting cut off at 45% width.
SCALE_BAR_CROP_FRAC = {
    "top_left":    (0.00, 0.00, 0.55, 0.25),
    "title_block": (0.00, 0.85, 0.50, 1.00),
}


_SYSTEM_PROMPT = """You are a construction-drawing scale-bar reader.

You will see a CROP of a floor-plan page that contains the SCALE BAR
(a small printed ruler with tick labels like 1m, 2m, 3m and a label
like "SCALE 1:100" or "1:50").

Read the scale and return ONLY this JSON object:

{
  "value": "1:100" | "1:50" | "1:200" | ... ,
  "confidence": <0..1, how legible the glyph is>,
  "reasoning": "<one sentence>"
}

If you cannot see a scale bar in the crop, return:
{ "value": "", "confidence": 0.0, "reasoning": "no scale bar visible" }

Never invent. Never guess. The estimator will fall back to a
dimension-driven calibrator if you return empty.
"""


def _crop_for_scale_bar(img: Image.Image, region: str = "top_left") -> Image.Image:
    fx0, fy0, fx1, fy1 = SCALE_BAR_CROP_FRAC.get(region, SCALE_BAR_CROP_FRAC["top_left"])
    w, h = img.width, img.height
    return img.crop((int(fx0 * w), int(fy0 * h), int(fx1 * w), int(fy1 * h)))


def _pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=88)
    return base64.standard_b64encode(buf.getvalue()).decode("ascii")


def _parse_json(text: str) -> dict:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return json.loads(fenced.group(1))
    first, last = text.find("{"), text.rfind("}")
    if first == -1 or last == -1:
        raise ValueError("No JSON in scale-reader response")
    return json.loads(text[first : last + 1])


def read_scale(
    page_image: Image.Image,
    *,
    region: str = "top_left",
    model: str = DEFAULT_MODEL,
    timeout_s: int = DEFAULT_TIMEOUT_S,
) -> Optional[ScaleDetected]:
    """Read the scale bar from `page_image`. Returns None on any error.

    `region` selects which corner of the page to crop:
      - "top_left"     (default — matches our synthetic fixture)
      - "title_block"  (alternative — some real plans put it here)

    Caller should handle None gracefully (fall back to dimension-driven
    calibration in Phase 2).
    """
    try:
        from anthropic import Anthropic, APIError, APITimeoutError
    except ImportError:
        return None

    crop = _crop_for_scale_bar(page_image, region=region)
    img_b64 = _pil_to_b64(crop)

    client = Anthropic(timeout=timeout_s)
    raw_text = ""
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=DEFAULT_MAX_TOKENS,
            system=_SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {
                        "type": "base64", "media_type": "image/jpeg", "data": img_b64,
                    }},
                    {"type": "text", "text": (
                        f"Image is {crop.width} x {crop.height} px (a crop from the "
                        f"{region.replace('_', ' ')} of the page). Return the JSON."
                    )},
                ],
            }],
        )
        raw_text = "".join(
            b.text for b in resp.content if getattr(b, "type", None) == "text"
        )
    except (APITimeoutError, APIError):
        return None
    except Exception:  # noqa: BLE001
        return None

    try:
        parsed = _parse_json(raw_text)
    except (ValueError, json.JSONDecodeError):
        return None

    value = str(parsed.get("value", "") or "").strip()
    if not value:
        return None
    try:
        confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0.5) or 0.5)))
    except (TypeError, ValueError):
        confidence = 0.5
    return ScaleDetected(value=value, source="scale_bar", confidence=confidence)
