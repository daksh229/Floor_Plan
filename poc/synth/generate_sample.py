"""Synthesises a non-trivial electrical layout PDF + ground-truth JSON.

The output is the test fixture for the rest of the pipeline. It is built from
the same symbol library used for template matching (Phase 2), with three
deliberate sources of difficulty layered on top so the demo is not a
self-fulfilling prophecy:

  1. Per-placement scale jitter (0.80x .. 1.20x)
  2. Per-placement rotation (0 / 90 / 180 / 270 degrees, weighted toward 0)
  3. Per-page Gaussian noise + subtle blur

Outputs:
  samples/synthetic_layout.pdf
  samples/synthetic_layout_ground_truth.json   (every placed symbol's class,
                                                 centre, rotation, scale, bbox)

The ground-truth file is what lets Phase 2 compute recall / precision rather
than guessing.

Run:
    python -m synth.generate_sample
"""
from __future__ import annotations

import hashlib
import json
import random
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from symbol_library.generate_symbols import SYMBOL_SIZE


SEED = 20260527
CANVAS_W, CANVAS_H = 2000, 1400         # ~landscape A4 at ~170 DPI
TITLE_BLOCK_H = 110
LEGEND_W, LEGEND_H = 420, 460

SYMBOL_DIR = Path(__file__).resolve().parent.parent / "symbol_library"
SAMPLES_DIR = Path(__file__).resolve().parent.parent.parent / "samples"

WALL_COLOR = (40, 40, 40)
ROOM_LABEL_COLOR = (60, 60, 60)
BG = (252, 252, 250)


@dataclass
class Room:
    name: str
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def cx(self) -> int:
        return (self.x0 + self.x1) // 2

    @property
    def cy(self) -> int:
        return (self.y0 + self.y1) // 2


# Realistic-ish single-storey ground-floor layout. Coordinates are absolute.
ROOMS: list[Room] = [
    Room("Bedroom 1",   90, 200,  640,  720),
    Room("Bedroom 2",   90, 720,  640, 1200),
    Room("Living",     640, 200, 1380,  820),
    Room("Kitchen",   1380, 200, 1900,  620),
    Room("Bathroom",  1380, 620, 1700,  900),
    Room("Laundry",   1700, 620, 1900,  900),
    Room("Hallway",    640, 820, 1900, 1000),
    Room("Garage",     640, 1000, 1900, 1200),
]


# Per-room symbol budget — chosen so the BOM has interesting variety.
# Format: list of (symbol_class, count)
ROOM_FIXTURES: dict[str, list[tuple[str, int]]] = {
    "Bedroom 1": [
        ("single_pole_switch", 1),
        ("double_gpo", 3),
        ("ceiling_light", 1),
        ("ceiling_fan", 1),
        ("data_point", 1),
        ("smoke_detector", 1),
    ],
    "Bedroom 2": [
        ("single_pole_switch", 1),
        ("double_gpo", 2),
        ("ceiling_light", 1),
        ("ceiling_fan", 1),
        ("data_point", 1),
    ],
    "Living": [
        ("two_way_switch", 2),
        ("double_gpo", 4),
        ("single_gpo", 1),
        ("downlight", 6),
        ("ceiling_fan", 1),
        ("tv_point", 1),
        ("data_point", 1),
    ],
    "Kitchen": [
        ("single_pole_switch", 1),
        ("dimmer", 1),
        ("double_gpo", 4),
        ("downlight", 4),
        ("smoke_detector", 1),
    ],
    "Bathroom": [
        ("single_pole_switch", 1),
        ("downlight", 2),
        ("exhaust_fan", 1),
        ("wall_light", 1),
    ],
    "Laundry": [
        ("single_pole_switch", 1),
        ("double_gpo", 1),
        ("wp_gpo", 1),
        ("exhaust_fan", 1),
        ("distribution_board", 1),
    ],
    "Hallway": [
        ("two_way_switch", 2),
        ("ceiling_light", 1),
        ("smoke_detector", 1),
    ],
    "Garage": [
        ("single_pole_switch", 1),
        ("wp_gpo", 2),
        ("ceiling_light", 2),
    ],
}


@dataclass
class Placement:
    symbol_class: str
    room: str
    cx: int
    cy: int
    scale: float
    rotation_deg: int
    bbox: tuple[int, int, int, int]   # x0, y0, x1, y1 in canvas pixels


def _font(size: int) -> ImageFont.ImageFont:
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _load_symbols() -> dict[str, Image.Image]:
    symbols: dict[str, Image.Image] = {}
    for png in sorted(SYMBOL_DIR.glob("*.png")):
        symbols[png.stem] = Image.open(png).convert("RGBA")
    return symbols


def _draw_walls(draw: ImageDraw.ImageDraw) -> None:
    for room in ROOMS:
        draw.rectangle((room.x0, room.y0, room.x1, room.y1),
                       outline=WALL_COLOR, width=5)


def _draw_room_labels(draw: ImageDraw.ImageDraw) -> None:
    # 30px (was 22) so Tesseract picks up small rooms like Bathroom and Laundry
    # where label area is crowded by symbols.
    font = _font(30)
    for room in ROOMS:
        bbox = draw.textbbox((0, 0), room.name, font=font)
        tw = bbox[2] - bbox[0]
        draw.text((room.cx - tw // 2, room.y0 + 10), room.name,
                  fill=ROOM_LABEL_COLOR, font=font)


def _legend_bbox() -> tuple[int, int, int, int]:
    lx = CANVAS_W - LEGEND_W - 30
    ly = CANVAS_H - LEGEND_H - 30
    return (lx, ly, lx + LEGEND_W, ly + LEGEND_H)


def _intersects(rect_a: tuple[int, int, int, int],
                rect_b: tuple[int, int, int, int],
                margin: int = 0) -> bool:
    ax0, ay0, ax1, ay1 = rect_a
    bx0, by0, bx1, by1 = rect_b
    return not (
        ax1 + margin < bx0 or
        ax0 - margin > bx1 or
        ay1 + margin < by0 or
        ay0 - margin > by1
    )


def _det_seed(*parts) -> int:
    """Deterministic integer seed from arbitrary inputs (str/int/tuple).

    Python 3's `hash()` randomises str hashing per process unless
    PYTHONHASHSEED is set, which silently makes the synth fixture
    non-reproducible. hashlib gives us byte-stable seeds.
    """
    h = hashlib.sha256()
    for p in parts:
        h.update(repr(p).encode("utf-8"))
        h.update(b"\x1f")
    # take first 4 bytes as an unsigned int for random.Random()
    return struct.unpack("<I", h.digest()[:4])[0]


def _draw_title_block(draw: ImageDraw.ImageDraw) -> None:
    draw.rectangle((0, 0, CANVAS_W, TITLE_BLOCK_H), outline=WALL_COLOR, width=3)
    title_font = _font(28)
    meta_font = _font(18)
    draw.text((20, 18), "PROJECT: SYNTHETIC TEST DWELLING",
              fill=WALL_COLOR, font=title_font)
    draw.text((20, 56), "Sheet: E-100 — Ground Floor Electrical Layout",
              fill=WALL_COLOR, font=meta_font)
    draw.text((20, 78), "Scale: 1:100 (indicative)  |  Drawn: synth.generate_sample.py",
              fill=WALL_COLOR, font=meta_font)
    draw.text((CANVAS_W - 280, 18), "DWG NO.  E-100",
              fill=WALL_COLOR, font=title_font)
    draw.text((CANVAS_W - 280, 56), "Rev: 1   Date: 2026-05-27",
              fill=WALL_COLOR, font=meta_font)


def _draw_legend(canvas: Image.Image, symbols: dict[str, Image.Image]) -> None:
    lx = CANVAS_W - LEGEND_W - 30
    ly = CANVAS_H - LEGEND_H - 30
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((lx, ly, lx + LEGEND_W, ly + LEGEND_H),
                   outline=WALL_COLOR, width=3, fill=(255, 255, 255))
    title_font = _font(20)
    draw.text((lx + 14, ly + 10), "LEGEND", fill=WALL_COLOR, font=title_font)
    line_font = _font(15)

    items = list(symbols.items())
    cols = 2
    col_w = LEGEND_W // cols
    row_h = 42
    for i, (name, img) in enumerate(items):
        col = i % cols
        row = i // cols
        x = lx + 12 + col * col_w
        y = ly + 46 + row * row_h
        thumb = img.copy()
        thumb.thumbnail((32, 32), Image.LANCZOS)
        canvas.paste(thumb, (x, y), thumb)
        draw.text((x + 40, y + 6), name.replace("_", " "),
                  fill=WALL_COLOR, font=line_font)


_ROOM_LABEL_BAND_PX = 60   # height of the no-symbol band at the top of each room
                            # so the room label can be OCR'd without being covered.


def _placement_positions(
    room: Room,
    count: int,
    margin: int = 36,
    exclude: tuple[int, int, int, int] | None = None,
    exclude_margin: int = 40,
) -> list[tuple[int, int]]:
    """Distribute `count` positions inside the room with margin from walls,
    refusing positions that would land inside the exclusion rectangle (the
    legend, in our case) or inside the top room-label band. Uses
    deterministic hashlib-derived seed."""
    rng = random.Random(_det_seed("placement", room.name, count, SEED))
    positions: list[tuple[int, int]] = []
    attempts = 0
    max_attempts = max(count * 80, 200)
    # No-symbol band at the top of the room so the label is OCR-readable.
    y_min = room.y0 + margin + _ROOM_LABEL_BAND_PX
    while len(positions) < count and attempts < max_attempts:
        attempts += 1
        x = rng.randint(room.x0 + margin, room.x1 - margin)
        y = rng.randint(y_min, room.y1 - margin) if y_min < room.y1 - margin \
            else rng.randint(room.y0 + margin, room.y1 - margin)
        if exclude is not None:
            ex0, ey0, ex1, ey1 = exclude
            if (ex0 - exclude_margin <= x <= ex1 + exclude_margin and
                    ey0 - exclude_margin <= y <= ey1 + exclude_margin):
                continue  # candidate is inside (or near) the legend
        if all((x - px) ** 2 + (y - py) ** 2 > 50 ** 2 for px, py in positions):
            positions.append((x, y))
    return positions


def _place_symbols(canvas: Image.Image, symbols: dict[str, Image.Image]) -> list[Placement]:
    rng = random.Random(SEED)
    placements: list[Placement] = []
    legend_rect = _legend_bbox()

    for room in ROOMS:
        fixtures = ROOM_FIXTURES.get(room.name, [])
        total = sum(c for _, c in fixtures)
        # Only apply legend exclusion to rooms whose bbox actually overlaps the legend
        # (otherwise we waste attempts in distant rooms).
        room_rect = (room.x0, room.y0, room.x1, room.y1)
        excl = legend_rect if _intersects(room_rect, legend_rect, margin=40) else None
        positions = _placement_positions(room, total, exclude=excl)
        if not positions:
            continue
        pos_iter = iter(positions)
        for symbol_class, count in fixtures:
            template = symbols.get(symbol_class)
            if template is None:
                continue
            for _ in range(count):
                try:
                    cx, cy = next(pos_iter)
                except StopIteration:
                    break
                scale = rng.uniform(0.80, 1.20)
                rotation = rng.choices([0, 90, 180, 270], weights=[6, 2, 1, 2])[0]
                sized = template.resize(
                    (max(20, int(SYMBOL_SIZE * scale)),
                     max(20, int(SYMBOL_SIZE * scale))),
                    Image.LANCZOS,
                )
                rotated = sized.rotate(rotation, expand=True, resample=Image.BICUBIC)
                w, h = rotated.size
                x0 = cx - w // 2
                y0 = cy - h // 2
                canvas.paste(rotated, (x0, y0), rotated)
                placements.append(
                    Placement(
                        symbol_class=symbol_class,
                        room=room.name,
                        cx=cx,
                        cy=cy,
                        scale=round(scale, 3),
                        rotation_deg=rotation,
                        bbox=(x0, y0, x0 + w, y0 + h),
                    )
                )
    return placements


def _apply_paper_noise(canvas: Image.Image) -> Image.Image:
    """Light blur + faint speckle so the page is not pixel-perfect clean."""
    blurred = canvas.filter(ImageFilter.GaussianBlur(radius=0.4))
    # subtle bytewise noise via PIL only (avoid numpy dep at this phase)
    rng = random.Random(SEED + 1)
    px = blurred.load()
    w, h = blurred.size
    sample = max(1, (w * h) // 4000)
    for _ in range(sample):
        x = rng.randint(0, w - 1)
        y = rng.randint(0, h - 1)
        r, g, b = px[x, y][:3]
        d = rng.randint(-12, 6)
        px[x, y] = (max(0, min(255, r + d)),
                    max(0, min(255, g + d)),
                    max(0, min(255, b + d)))
    return blurred


def generate(output_pdf: Path | None = None,
             output_ground_truth: Path | None = None) -> tuple[Path, Path]:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    if output_pdf is None:
        output_pdf = SAMPLES_DIR / "synthetic_layout.pdf"
    if output_ground_truth is None:
        output_ground_truth = SAMPLES_DIR / "synthetic_layout_ground_truth.json"

    symbols = _load_symbols()
    if not symbols:
        raise RuntimeError(
            f"No symbol PNGs found in {SYMBOL_DIR}. "
            "Run `python symbol_library/generate_symbols.py` first."
        )

    canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), BG)
    draw = ImageDraw.Draw(canvas)
    _draw_title_block(draw)
    _draw_walls(draw)
    _draw_room_labels(draw)

    placements = _place_symbols(canvas, symbols)
    _draw_legend(canvas, symbols)

    noisy = _apply_paper_noise(canvas)
    noisy.save(output_pdf, "PDF", resolution=170.0)

    counts: dict[str, int] = {}
    for p in placements:
        counts[p.symbol_class] = counts.get(p.symbol_class, 0) + 1

    lx0, ly0, lx1, ly1 = _legend_bbox()
    ground_truth = {
        "source_pdf": output_pdf.name,
        "page_index": 1,
        "canvas": {"width": CANVAS_W, "height": CANVAS_H},
        "seed": SEED,
        # Persisted so pipeline/runner can mask out the legend during detection.
        # On unknown PDFs (uploads), no mask is applied.
        "mask_regions": [
            {"name": "legend", "bbox": [lx0, ly0, lx1, ly1]},
        ],
        "placement_count": len(placements),
        "counts_by_class": dict(sorted(counts.items())),
        "placements": [asdict(p) for p in placements],
    }
    output_ground_truth.write_text(json.dumps(ground_truth, indent=2), encoding="utf-8")
    return output_pdf, output_ground_truth


if __name__ == "__main__":
    pdf, gt = generate()
    print(f"PDF          : {pdf}")
    print(f"Ground truth : {gt}")
    data = json.loads(gt.read_text(encoding="utf-8"))
    print(f"Total placed : {data['placement_count']}")
    print("Counts by class:")
    for k, v in data["counts_by_class"].items():
        print(f"  {k:24s} {v}")
