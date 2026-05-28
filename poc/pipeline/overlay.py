"""Render Detection or OCRSpan bboxes on the page image."""
from __future__ import annotations

import hashlib
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont

from .schemas import Detection, OCRSpan


def _color_for_class(symbol_class: str) -> tuple[int, int, int]:
    """Deterministic mid-saturation colour from the class name, stable across runs."""
    h = hashlib.md5(symbol_class.encode("utf-8")).digest()
    r = 60 + h[0] % 180
    g = 60 + h[1] % 180
    b = 60 + h[2] % 180
    return (r, g, b)


def _load_font(size: int) -> ImageFont.ImageFont:
    for name in ("arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _text_size(draw: ImageDraw.ImageDraw, text: str, font) -> tuple[int, int]:
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        return font.getsize(text)


def draw_detections(
    page_image: Image.Image,
    detections: Iterable[Detection],
    *,
    show_score: bool = True,
    line_width: int | None = None,
    label_font_size: int | None = None,
) -> Image.Image:
    """Render Detection bboxes on the page, auto-scaling stroke + font size
    to the image so the overlay stays readable on both 2000 px and 4500 px
    renders without manual tuning per resolution."""
    canvas = page_image.convert("RGBA").copy()
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    # auto-size to image: 5 px stroke + 20 px font at 2000 px wide,
    # scaling linearly so 4500 px gets 11 px stroke + 45 px font
    if line_width is None:
        line_width = max(3, round(canvas.width / 400))
    if label_font_size is None:
        label_font_size = max(14, round(canvas.width / 100))
    font = _load_font(label_font_size)

    for det in detections:
        color = _color_for_class(det.symbol_class)
        x0, y0, x1, y1 = det.bbox
        draw.rectangle((x0, y0, x1, y1), outline=color + (255,), width=line_width)

        label = det.symbol_class.replace("_", " ")
        if show_score:
            label = f"{label} {det.score:.2f}"

        tw, th = _text_size(draw, label, font)
        pad = max(3, label_font_size // 4)
        bx0 = x0
        by0 = max(0, y0 - th - 2 * pad)
        bx1 = bx0 + tw + 2 * pad
        by1 = by0 + th + 2 * pad
        draw.rectangle((bx0, by0, bx1, by1), fill=color + (220,))
        draw.text((bx0 + pad, by0 + pad), label, fill=(255, 255, 255, 255), font=font)

    return Image.alpha_composite(canvas, overlay).convert("RGB")


def class_colour_map(detections: Iterable[Detection]) -> dict[str, tuple[int, int, int]]:
    """Return {symbol_class: (r, g, b)} for every detected class, sorted by
    class name. Used by the Streamlit colour-legend strip and the SVG
    overlay component."""
    classes = sorted({d.symbol_class for d in detections})
    return {c: _color_for_class(c) for c in classes}


def draw_ocr_spans(
    page_image: Image.Image,
    spans: Iterable[OCRSpan],
    *,
    color: tuple[int, int, int] = (20, 132, 220),
    line_width: int = 2,
    label_font_size: int = 13,
    show_confidence: bool = True,
) -> Image.Image:
    """Render OCR span bboxes with the recognised text as the label."""
    canvas = page_image.convert("RGBA").copy()
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = _load_font(label_font_size)

    for span in spans:
        x0, y0, x1, y1 = span.bbox
        draw.rectangle((x0, y0, x1, y1), outline=color + (255,), width=line_width)
        label = f"{span.text} ({span.confidence:.2f})" if show_confidence else span.text
        tw, th = _text_size(draw, label, font)
        pad = 2
        bx0 = x0
        by0 = max(0, y0 - th - 2 * pad)
        bx1 = bx0 + tw + 2 * pad
        by1 = by0 + th + 2 * pad
        draw.rectangle((bx0, by0, bx1, by1), fill=color + (200,))
        draw.text((bx0 + pad, by0 + pad), label, fill=(255, 255, 255, 255), font=font)

    return Image.alpha_composite(canvas, overlay).convert("RGB")
