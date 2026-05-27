"""Draw extracted bounding boxes back onto the page image.

This is the visible payoff for the §3 coordinate translation layer: Claude
returns image-space coords, we render them right back on the rendered page.
Placeholder boxes are drawn dashed to honour the §7 three-state distinction.
"""
from __future__ import annotations

from typing import Iterable, List

from PIL import Image, ImageDraw, ImageFont

from .schemas import BoundingBox, DetectionState, ExtractedField


_STATE_COLOR = {
    DetectionState.DETECTED_ACTUAL: (16, 163, 74),       # green
    DetectionState.DETECTED_PLACEHOLDER: (234, 179, 8),  # amber
    DetectionState.NOT_DETECTED: (148, 163, 184),        # slate (unused; not drawn)
}


def _load_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def draw_fields(page_image: Image.Image, fields: Iterable[ExtractedField]) -> Image.Image:
    canvas = page_image.convert("RGBA").copy()
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = _load_font(max(14, canvas.width // 90))

    for field in fields:
        if field.detection_state == DetectionState.NOT_DETECTED:
            continue
        color = _STATE_COLOR[field.detection_state]
        is_placeholder = field.detection_state == DetectionState.DETECTED_PLACEHOLDER
        for box in field.overlay_coordinates:
            _draw_box(draw, box, color, dashed=is_placeholder)
            label = box.label or field.display_name
            _draw_label(draw, box, label, color, font)

    return Image.alpha_composite(canvas, overlay).convert("RGB")


def _draw_box(
    draw: ImageDraw.ImageDraw,
    box: BoundingBox,
    color: tuple,
    dashed: bool,
    width: int = 3,
) -> None:
    x0, y0 = box.x, box.y
    x1, y1 = box.x + box.w, box.y + box.h
    if not dashed:
        draw.rectangle((x0, y0, x1, y1), outline=color + (255,), width=width)
        return

    dash = 12
    gap = 8
    _dashed_line(draw, (x0, y0), (x1, y0), color, width, dash, gap)
    _dashed_line(draw, (x1, y0), (x1, y1), color, width, dash, gap)
    _dashed_line(draw, (x1, y1), (x0, y1), color, width, dash, gap)
    _dashed_line(draw, (x0, y1), (x0, y0), color, width, dash, gap)


def _dashed_line(draw, start, end, color, width, dash, gap):
    x0, y0 = start
    x1, y1 = end
    dx, dy = x1 - x0, y1 - y0
    length = (dx * dx + dy * dy) ** 0.5
    if length == 0:
        return
    ux, uy = dx / length, dy / length
    pos = 0.0
    drawing = True
    while pos < length:
        seg = dash if drawing else gap
        seg = min(seg, length - pos)
        if drawing:
            sx = x0 + ux * pos
            sy = y0 + uy * pos
            ex = x0 + ux * (pos + seg)
            ey = y0 + uy * (pos + seg)
            draw.line((sx, sy, ex, ey), fill=color + (255,), width=width)
        pos += seg
        drawing = not drawing


def _draw_label(
    draw: ImageDraw.ImageDraw,
    box: BoundingBox,
    text: str,
    color: tuple,
    font: ImageFont.ImageFont,
) -> None:
    pad = 4
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        tw, th = font.getsize(text)  # very old PIL fallback
    bx0 = box.x
    by0 = max(0, box.y - th - 2 * pad)
    bx1 = bx0 + tw + 2 * pad
    by1 = by0 + th + 2 * pad
    draw.rectangle((bx0, by0, bx1, by1), fill=color + (220,))
    draw.text((bx0 + pad, by0 + pad), text, fill=(255, 255, 255, 255), font=font)
