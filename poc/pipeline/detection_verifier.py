"""Claude-vision sanity check on the CV detections.

Sends Claude (a) the overlay PNG showing what was detected and (b) a JSON
summary of class counts, then asks: *does anything look obviously wrong —
mis-placed bboxes, mis-classified symbols, regions where you'd expect a
detection but don't see one?*

Returns Claude's free-text findings. **Does not mutate the BOM** — this
is a reviewer aid, not a correction step. The BOM is still produced by
the deterministic aggregator.
"""
from __future__ import annotations

import base64
import io
import json
import os
from collections import Counter
from typing import Iterable

from anthropic import Anthropic, APIError, APITimeoutError
from PIL import Image

from .schemas import Detection


DEFAULT_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")
INPUT_MAX_LONG_EDGE = 1800
DEFAULT_TIMEOUT_S = 60
DEFAULT_MAX_TOKENS = 700


_SYSTEM_PROMPT = """You are an electrical-drawings reviewer doing a sanity \
check on a computer-vision symbol-detection result. You will see:

  1. An overlay image of the floor plan with class-coloured bounding boxes \
     drawn where the CV detector found each symbol.
  2. A JSON summary of class counts.

Look at the overlay and answer in 3-5 bullet points:

  - Any obvious **mis-placements** (bbox over a clearly different symbol or empty area)?
  - Any obvious **mis-classifications** (e.g. a switch shown coloured as a GPO)?
  - Any **systematic gaps** — areas of the plan where you'd expect detections but see none?
  - Any class with a **suspicious count** (way too many or implausibly few for a residential layout)?

Keep findings concrete and specific (mention coordinates or rooms if helpful). \
This is a reviewer aid, NOT a correction step — do not invent numbers; just flag. \
Be brief, clinical, no fluff.
"""


def _downsample(img: Image.Image, max_long_edge: int) -> Image.Image:
    long_edge = max(img.width, img.height)
    if long_edge <= max_long_edge:
        return img
    ratio = max_long_edge / long_edge
    return img.resize((int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)


def _img_to_b64_jpeg(img: Image.Image, quality: int = 85) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality, optimize=True)
    return base64.standard_b64encode(buf.getvalue()).decode("ascii")


def verify_detections(
    overlay_image: Image.Image,
    detections: Iterable[Detection],
    *,
    model: str = DEFAULT_MODEL,
) -> tuple[str, str | None]:
    """Returns (findings_text, error_str_or_None)."""
    dets = list(detections)
    counts = Counter(d.symbol_class for d in dets)
    summary = {
        "total_detections": len(dets),
        "counts_by_class": dict(sorted(counts.items())),
    }
    summary_text = (
        "JSON summary of the detector's findings:\n"
        + json.dumps(summary, indent=2)
        + "\n\nNow review the overlay image above and give your findings."
    )

    img_small = _downsample(overlay_image, INPUT_MAX_LONG_EDGE)
    img_b64 = _img_to_b64_jpeg(img_small)

    client = Anthropic(timeout=DEFAULT_TIMEOUT_S)
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
                    {"type": "text", "text": summary_text},
                ],
            }],
        )
        text = "".join(
            b.text for b in resp.content if getattr(b, "type", None) == "text"
        )
        return text.strip(), None
    except (APITimeoutError, APIError) as exc:
        return "", f"{type(exc).__name__}: {exc}"
    except Exception as exc:  # noqa: BLE001
        return "", f"{type(exc).__name__}: {exc}"
