"""Generates 15 schematic electrical-symbol PNGs that serve as both:

  - the template-matching reference set used by pipeline/cv_detect.py (Phase 2)
  - the visual building blocks used by synth/generate_sample.py to compose
    a non-trivial synthetic electrical layout (Phase 1).

Each symbol is rendered black-on-transparent at SYMBOL_SIZE x SYMBOL_SIZE.
Shapes are kept deliberately distinct so template matching has a fair chance.

Run:
    python -m symbol_library.generate_symbols
or:
    python poc/symbol_library/generate_symbols.py
"""
from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


SYMBOL_SIZE = 80                 # square canvas in pixels
LINE = 3                         # stroke weight
BLACK = (0, 0, 0, 255)
TRANSPARENT = (0, 0, 0, 0)
OUTPUT_DIR = Path(__file__).parent


def _font(size: int) -> ImageFont.ImageFont:
    for name in ("arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _canvas() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGBA", (SYMBOL_SIZE, SYMBOL_SIZE), TRANSPARENT)
    return img, ImageDraw.Draw(img)


def _text_centered(draw: ImageDraw.ImageDraw, text: str, size: int, xy=(SYMBOL_SIZE // 2, SYMBOL_SIZE // 2)) -> None:
    font = _font(size)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    cx, cy = xy
    draw.text((cx - tw / 2 - bbox[0], cy - th / 2 - bbox[1]), text, fill=BLACK, font=font)


# ---------- symbol shape primitives ----------

def _outer_circle(draw: ImageDraw.ImageDraw, inset: int = 10) -> None:
    draw.ellipse((inset, inset, SYMBOL_SIZE - inset, SYMBOL_SIZE - inset),
                 outline=BLACK, width=LINE)


def _prongs(draw: ImageDraw.ImageDraw, cy: int, sep: int = 10, length: int = 12) -> None:
    cx = SYMBOL_SIZE // 2
    draw.line((cx - sep, cy, cx - sep, cy + length), fill=BLACK, width=LINE)
    draw.line((cx + sep, cy, cx + sep, cy + length), fill=BLACK, width=LINE)


# ---------- per-symbol renderers ----------

def single_pole_switch() -> Image.Image:
    img, d = _canvas()
    _outer_circle(d)
    _text_centered(d, "S", 36)
    return img


def two_way_switch() -> Image.Image:
    img, d = _canvas()
    _outer_circle(d)
    _text_centered(d, "S2", 28)
    return img


def dimmer() -> Image.Image:
    img, d = _canvas()
    _outer_circle(d)
    _text_centered(d, "DIM", 20)
    return img


def single_gpo() -> Image.Image:
    img, d = _canvas()
    _outer_circle(d, inset=12)
    _text_centered(d, "1", 28, xy=(SYMBOL_SIZE // 2, SYMBOL_SIZE // 2 - 4))
    _prongs(d, cy=SYMBOL_SIZE - 22)
    return img


def double_gpo() -> Image.Image:
    img, d = _canvas()
    _outer_circle(d, inset=12)
    _text_centered(d, "2", 28, xy=(SYMBOL_SIZE // 2, SYMBOL_SIZE // 2 - 4))
    _prongs(d, cy=SYMBOL_SIZE - 22, sep=8)
    _prongs(d, cy=SYMBOL_SIZE - 22, sep=16)
    return img


def wp_gpo() -> Image.Image:
    """Weatherproof GPO. Designed so it does NOT contain the visual subset of
    single_gpo (circle + prongs) — that subset collision is what forced us into
    cross-class NMS + specificity-bonus gymnastics. Here we use a hinged housing
    box with a diagonal weatherproof cover marker and an IP54 label, plus a
    filled-disc centre symbol that doesn't have the prong glyph at all.
    """
    img, d = _canvas()
    # outer hinged housing rectangle with a small hinge on the left edge
    d.rectangle((6, 6, SYMBOL_SIZE - 6, SYMBOL_SIZE - 6), outline=BLACK, width=LINE)
    # hinge marker (two short ticks on the left edge)
    d.line((4, 18, 8, 18), fill=BLACK, width=LINE)
    d.line((4, SYMBOL_SIZE - 18, 8, SYMBOL_SIZE - 18), fill=BLACK, width=LINE)
    # diagonal "cover open" indicator across the top-right corner
    d.line((SYMBOL_SIZE - 26, 10, SYMBOL_SIZE - 10, 26), fill=BLACK, width=LINE)
    # central filled disc (distinct from the open-circle + prongs of single_gpo)
    cx = cy = SYMBOL_SIZE // 2
    r = 14
    d.ellipse((cx - r, cy - r + 6, cx + r, cy + r + 6), fill=BLACK)
    # IP54 label below the disc
    _text_centered(d, "IP54", 13, xy=(SYMBOL_SIZE // 2, SYMBOL_SIZE - 16))
    return img


def ceiling_light() -> Image.Image:
    img, d = _canvas()
    _outer_circle(d, inset=12)
    # X through the circle
    inset = 18
    d.line((inset, inset, SYMBOL_SIZE - inset, SYMBOL_SIZE - inset), fill=BLACK, width=LINE)
    d.line((SYMBOL_SIZE - inset, inset, inset, SYMBOL_SIZE - inset), fill=BLACK, width=LINE)
    return img


def wall_light() -> Image.Image:
    img, d = _canvas()
    # half-circle anchored at bottom
    d.pieslice((14, 14, SYMBOL_SIZE - 14, SYMBOL_SIZE - 14), start=180, end=360,
               outline=BLACK, width=LINE)
    d.line((14, SYMBOL_SIZE // 2, SYMBOL_SIZE - 14, SYMBOL_SIZE // 2),
           fill=BLACK, width=LINE)
    # X across the half-circle
    d.line((22, 22, SYMBOL_SIZE - 22, SYMBOL_SIZE // 2 - 2), fill=BLACK, width=LINE)
    d.line((SYMBOL_SIZE - 22, 22, 22, SYMBOL_SIZE // 2 - 2), fill=BLACK, width=LINE)
    return img


def downlight() -> Image.Image:
    img, d = _canvas()
    cx = cy = SYMBOL_SIZE // 2
    r = 14
    d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=BLACK)
    return img


def ceiling_fan() -> Image.Image:
    img, d = _canvas()
    _outer_circle(d, inset=12)
    cx = cy = SYMBOL_SIZE // 2
    # four fan-blade strokes at 45 degree offsets
    for angle_deg in (15, 105, 195, 285):
        a = math.radians(angle_deg)
        r = 26
        d.line((cx, cy, cx + r * math.cos(a), cy + r * math.sin(a)),
               fill=BLACK, width=LINE)
    # hub
    d.ellipse((cx - 4, cy - 4, cx + 4, cy + 4), fill=BLACK)
    return img


def exhaust_fan() -> Image.Image:
    img, d = _canvas()
    d.rectangle((10, 10, SYMBOL_SIZE - 10, SYMBOL_SIZE - 10), outline=BLACK, width=LINE)
    cx = cy = SYMBOL_SIZE // 2
    # stylised fan: small central circle + four curved blade strokes
    d.ellipse((cx - 4, cy - 4, cx + 4, cy + 4), fill=BLACK)
    for angle_deg in (0, 90, 180, 270):
        a = math.radians(angle_deg)
        x = cx + 22 * math.cos(a)
        y = cy + 22 * math.sin(a)
        d.line((cx, cy, x, y), fill=BLACK, width=LINE)
    _text_centered(d, "EF", 14, xy=(cx, SYMBOL_SIZE - 16))
    return img


def smoke_detector() -> Image.Image:
    img, d = _canvas()
    _outer_circle(d, inset=10)
    _outer_circle(d, inset=16)
    _text_centered(d, "SD", 18)
    return img


def data_point() -> Image.Image:
    img, d = _canvas()
    # right-pointing triangle
    d.polygon([(16, 16), (16, SYMBOL_SIZE - 16), (SYMBOL_SIZE - 16, SYMBOL_SIZE // 2)],
              outline=BLACK, width=LINE)
    _text_centered(d, "D", 20, xy=(SYMBOL_SIZE // 2 - 6, SYMBOL_SIZE // 2))
    return img


def tv_point() -> Image.Image:
    img, d = _canvas()
    d.polygon([(16, 16), (16, SYMBOL_SIZE - 16), (SYMBOL_SIZE - 16, SYMBOL_SIZE // 2)],
              outline=BLACK, width=LINE)
    _text_centered(d, "TV", 16, xy=(SYMBOL_SIZE // 2 - 8, SYMBOL_SIZE // 2))
    return img


def distribution_board() -> Image.Image:
    img, d = _canvas()
    # taller rectangle representing the panel
    d.rectangle((18, 8, SYMBOL_SIZE - 18, SYMBOL_SIZE - 8), outline=BLACK, width=LINE)
    # internal breaker hatch lines
    for y in (24, 34, 44, 54, 64):
        d.line((24, y, SYMBOL_SIZE - 24, y), fill=BLACK, width=1)
    _text_centered(d, "DB", 16, xy=(SYMBOL_SIZE // 2, 16))
    return img


SYMBOLS = {
    "single_pole_switch": single_pole_switch,
    "two_way_switch": two_way_switch,
    "dimmer": dimmer,
    "single_gpo": single_gpo,
    "double_gpo": double_gpo,
    "wp_gpo": wp_gpo,
    "ceiling_light": ceiling_light,
    "wall_light": wall_light,
    "downlight": downlight,
    "ceiling_fan": ceiling_fan,
    "exhaust_fan": exhaust_fan,
    "smoke_detector": smoke_detector,
    "data_point": data_point,
    "tv_point": tv_point,
    "distribution_board": distribution_board,
}


def generate_all(output_dir: Path = OUTPUT_DIR) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for name, fn in SYMBOLS.items():
        img = fn()
        path = output_dir / f"{name}.png"
        img.save(path, "PNG")
        written[name] = path
    return written


if __name__ == "__main__":
    paths = generate_all()
    print(f"Wrote {len(paths)} symbols to {OUTPUT_DIR}:")
    for name, path in paths.items():
        print(f"  - {name:24s} -> {path.name}")
