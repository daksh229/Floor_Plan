"""Claude vision extractor with strict JSON validation, retry, and partial-result handling.

Mitigates the §7 failure mode "no retry, no timeout, no partial-result handling".
Every call writes a row to the audit store so the admin/feedback page (§3) has data.
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import List, Optional

from anthropic import Anthropic, APIError, APITimeoutError
from PIL import Image
from pydantic import ValidationError

from .prompts import DEFAULT_VERSION, PromptVersion, get_prompt
from .schemas import (
    BoundingBox,
    DetectionState,
    ExtractedField,
    ExtractionResult,
)


DEFAULT_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TIMEOUT_S = 90
DEFAULT_RETRIES = 2


def _encode_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode("ascii")


def _parse_json_block(text: str) -> dict:
    """Claude sometimes wraps JSON in ```json ... ``` even when told not to. Strip it."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return json.loads(fenced.group(1))
    first = text.find("{")
    last = text.rfind("}")
    if first == -1 or last == -1:
        raise ValueError("No JSON object found in response")
    return json.loads(text[first : last + 1])


def _coerce_fields(raw: dict, prompt: PromptVersion, source_page: int) -> List[ExtractedField]:
    """Map the model's raw dict into ExtractedField, filling in display_name + source_page."""
    out: List[ExtractedField] = []
    raw_fields = raw.get("fields", []) if isinstance(raw, dict) else []
    spec_by_key = {v.key: v for v in prompt.variables}

    seen_keys = set()
    for item in raw_fields:
        key = item.get("variable")
        spec = spec_by_key.get(key)
        if spec is None:
            continue  # silently drop unknown keys; the prompt enumerates the closed set
        seen_keys.add(key)

        state_raw = item.get("detection_state", "not_detected")
        try:
            state = DetectionState(state_raw)
        except ValueError:
            state = DetectionState.NOT_DETECTED

        boxes_raw = item.get("overlay_coordinates") or []
        boxes: List[BoundingBox] = []
        for b in boxes_raw:
            try:
                boxes.append(BoundingBox(**b))
            except (ValidationError, TypeError):
                continue

        out.append(
            ExtractedField(
                variable=key,
                display_name=spec.display_name,
                value=item.get("value"),
                unit=item.get("unit") or spec.unit,
                source_page=source_page,
                detection_state=state,
                overlay_coordinates=boxes if state != DetectionState.NOT_DETECTED else [],
                confidence_note=item.get("confidence_note"),
            )
        )

    # Partial-result handling: any variable the model omitted becomes an explicit not_detected.
    for spec in prompt.variables:
        if spec.key in seen_keys:
            continue
        out.append(
            ExtractedField(
                variable=spec.key,
                display_name=spec.display_name,
                value=None,
                unit=spec.unit,
                source_page=source_page,
                detection_state=DetectionState.NOT_DETECTED,
                overlay_coordinates=[],
                confidence_note="Model did not return this variable.",
            )
        )
    return out


def extract_from_page(
    page_image: Image.Image,
    source_page: int,
    prompt_version: str = DEFAULT_VERSION,
    model: str = DEFAULT_MODEL,
    audit_dir: Optional[Path] = None,
) -> ExtractionResult:
    prompt = get_prompt(prompt_version)
    client = Anthropic(timeout=DEFAULT_TIMEOUT_S)

    user_text = prompt.render_user_text(source_page, page_image.width, page_image.height)
    image_b64 = _encode_png(page_image)

    raw_text = ""
    error: Optional[str] = None
    last_exc: Optional[Exception] = None

    for attempt in range(DEFAULT_RETRIES + 1):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=DEFAULT_MAX_TOKENS,
                system=prompt.system,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": image_b64,
                                },
                            },
                            {"type": "text", "text": user_text},
                        ],
                    }
                ],
            )
            raw_text = "".join(
                block.text for block in resp.content if getattr(block, "type", None) == "text"
            )
            break
        except (APITimeoutError, APIError) as exc:
            last_exc = exc
            if attempt < DEFAULT_RETRIES:
                time.sleep(1.5 * (attempt + 1))
                continue
            error = f"{type(exc).__name__}: {exc}"
            break
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            error = f"{type(exc).__name__}: {exc}"
            break

    if error:
        fields = _coerce_fields({}, prompt, source_page)  # all not_detected
        result = ExtractionResult(
            fields=fields,
            prompt_version=prompt.version,
            model=model,
            source_page=source_page,
            page_image_width=page_image.width,
            page_image_height=page_image.height,
            raw_response=None,
            error=error,
        )
        _write_audit(result, page_image, audit_dir)
        return result

    try:
        parsed = _parse_json_block(raw_text)
    except (ValueError, json.JSONDecodeError) as exc:
        fields = _coerce_fields({}, prompt, source_page)
        result = ExtractionResult(
            fields=fields,
            prompt_version=prompt.version,
            model=model,
            source_page=source_page,
            page_image_width=page_image.width,
            page_image_height=page_image.height,
            raw_response=raw_text,
            error=f"JSON parse failed: {exc}",
        )
        _write_audit(result, page_image, audit_dir)
        return result

    fields = _coerce_fields(parsed, prompt, source_page)
    result = ExtractionResult(
        fields=fields,
        prompt_version=prompt.version,
        model=model,
        source_page=source_page,
        page_image_width=page_image.width,
        page_image_height=page_image.height,
        raw_response=raw_text,
        error=None,
    )
    _write_audit(result, page_image, audit_dir)
    return result


def _write_audit(result: ExtractionResult, page_image: Image.Image, audit_dir: Optional[Path]) -> None:
    """Audit store sketch (§3, §7). Every request/response gets a folder for the admin page."""
    if audit_dir is None:
        return
    audit_dir.mkdir(parents=True, exist_ok=True)
    record_id = f"{int(time.time())}_{uuid.uuid4().hex[:8]}_p{result.source_page}"
    folder = audit_dir / record_id
    folder.mkdir(parents=True, exist_ok=True)
    page_image.save(folder / "page.png")
    (folder / "result.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
