"""Generate a measurement-focused residential floor-plan PDF + ground truth.

This is the fixture for the ProCalc-measurement POC. Unlike the electrical
sample (which loads pre-rendered symbol PNGs and scatters them), this
generator draws the plan from FIRST PRINCIPLES — walls, dimension chains,
doors, windows, scale bar, title block — so we can produce arbitrary
variants and have a precise, machine-readable ground truth.

What the generator produces (per run):

  - samples/measurement_sample_<seed>.pdf
      The plan as a single-page PDF, ready to upload to the wizard
  - samples/measurement_sample_<seed>.png
      Same image as PNG (useful for quick visual inspection)
  - samples/measurement_sample_<seed>_ground_truth.json
      Per-room polygons (in mm), every wall segment, every dimension
      (value + extension-line endpoints), every door, every window,
      scale bar, total floor area in m^2. This is what the extraction
      pipeline should re-derive — useful as a regression oracle.

How to regenerate variants:
  - Change SEED at the top of the file (or pass --seed on the CLI)
  - Or change LAYOUT to one of the registered templates

Run:
    python -m synth.generate_measurement_sample
or
    python -m synth.generate_measurement_sample --seed 17 --layout 6room

Conventions:
  - All real-world dimensions are in MILLIMETRES, integers.
  - Render coordinates are in PIXELS, origin top-left.
  - Conversion: render_px = PLAN_ORIGIN_PX + (mm * RENDER_PX_PER_MM).
  - Wall thickness is rendered at REAL scale (90 mm cavity wall).
  - Dimension chains are placed OUTSIDE the building footprint and
    annotated with extension lines + tick marks per AS 1100 convention.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


# ============================================================
# Page setup
# ============================================================

# A3 landscape at 150 DPI -> 2480 x 1754 px. Big enough for plan + title block.
PAGE_PX = (2480, 1754)
BG_RGB = (252, 252, 250)  # warm cream, real-paper feel

# We draw the plan at an INTERNAL ratio (px per real-world mm). This is the
# value the dimension-driven scale calibrator would discover by reading
# the dimensions and measuring their wall lengths in pixels.
RENDER_PX_PER_MM = 0.16   # 1mm real -> 0.16 px rendered. 10000mm -> 1600 px.

# Top-left of the plan's outer wall, on the page
PLAN_ORIGIN_PX = (350, 220)

# Colours
WALL_RGB = (40, 40, 40)
DIM_LINE_RGB = (60, 60, 60)
DIM_TEXT_RGB = (40, 40, 40)
ROOM_LABEL_RGB = (60, 60, 60)
GRID_RGB = (220, 220, 220)
DOOR_RGB = (90, 90, 90)
WINDOW_RGB = (40, 40, 90)

# Wall thickness (real mm). 90 mm = standard timber-frame internal wall.
WALL_THK_MM = 90

# Dimension styling
DIM_OFFSET_PX = 70         # how far the dim line sits outside the wall
DIM_EXT_OVERSHOOT_PX = 6   # tick mark length each side
DIM_TEXT_SIZE = 22
ROOM_LABEL_SIZE = 26
ROOM_DIM_SIZE = 18
WALL_STROKE_PX = 4


# ============================================================
# Layout templates
# ============================================================

@dataclass
class Room:
    name: str
    x: int       # top-left x in mm relative to plan origin
    y: int       # top-left y in mm relative to plan origin
    w: int       # width  in mm
    h: int       # height in mm
    is_wet: bool = False  # bathroom / ensuite / laundry

@dataclass
class Window:
    """A window on an external wall.
    `wall` is one of 'top','bottom','left','right'.
    `position_mm` is the centre of the window along that wall, measured
    from the wall's start (top-left to bottom-right convention).
    `width_mm` is window opening width (typically 600..1800 mm).
    """
    wall: str
    position_mm: int
    width_mm: int
    label: str = ""

@dataclass
class Door:
    """A door — internal or external. `wall` and `position_mm` like Window.
    `is_external` controls whether it's drawn with the entry-door style.
    `is_glass` flags glass sliding doors (count toward glass_doors_windows).
    """
    wall: str
    position_mm: int
    width_mm: int = 820  # standard residential door leaf
    is_external: bool = False
    is_glass: bool = False
    label: str = ""


# --- Layout: 6-room rectangular (10m x 8m) ---
#
#       <--2500-><--2500-><--2500-><--2500->
#       +-------+--------+-------+--------+
#  4000 |       |        |       |        |
#       | BED 2 | ENSUITE| BATH  | BED 1  |
#       |       |        |       | MASTER |
#       +-------+--------+-------+--------+
#  4000 |                 |               |
#       |    LIVING       |   KITCHEN     |
#       |                 |  + DINING     |
#       +-----------------+---------------+
#       <-----5000-------><------5000---->

LAYOUT_6ROOM = {
    "name": "6room_rectangular",
    "total_w_mm": 10000,
    "total_h_mm": 8000,
    "rooms": [
        Room("BED 2",   0,    0,    2500, 4000),
        Room("ENSUITE", 2500, 0,    2500, 4000, is_wet=True),
        Room("BATH",    5000, 0,    2500, 4000, is_wet=True),
        Room("BED 1",   7500, 0,    2500, 4000),
        Room("LIVING",  0,    4000, 5000, 4000),
        Room("KITCHEN", 5000, 4000, 5000, 4000),
    ],
    "windows": [
        Window("top",    1250, 1500, "W1"),  # BED 2
        Window("top",    8750, 1500, "W2"),  # BED 1
        Window("left",   2000, 1500, "W3"),  # BED 2 side
        Window("left",   6000, 1500, "W4"),  # LIVING side
        Window("right",  2000, 1500, "W5"),  # BED 1 side
        Window("bottom", 2500, 1800, "W6"),  # LIVING
        Window("bottom", 7500, 1800, "W7"),  # KITCHEN
    ],
    "doors": [
        # External entry
        Door("left",   5500, 900, is_external=True, label="D1"),
        # Sliding glass to backyard
        Door("bottom", 5000, 2400, is_external=True, is_glass=True, label="SD1"),
    ],
}


# --- Layout: 8-room L-shaped ---
# Demonstrates polygon (non-rectangular) footprint — important for the
# total_floor_area extraction. Notch is removed from the top-right.
#
#   <--2500-><--2500-><--2500->
#   +-------+--------+--------+                      ^
#   | BED 3 |  BATH  | ENSUITE|                      |
#   |       |        |        |  4000                |
#   |       |        |        |                      |
#   +-------+--------+--------+------+----------+    v ^
#   |       |  HALL  |        |      |          |      |
#   |       |        |   BED 1 (MASTER)         | 1500 |  8500
#   | BED 2 |        |        |      |          |      |
#   +-------+--------+--------+------+          |    ^ |
#   |                                  4000     | 3000 |
#   |              LIVING                       |      |
#   |                                           |      |
#   +-----------+-------------+-----------------+      v
#   |  ENTRY    |  KITCHEN    |    DINING       |
#   |           |             |                 | 2000
#   +-----------+-------------+-----------------+
#   <-2500-><---3500---><------5000------------>
#
# (Simplified version below — full L-shape would push the file beyond v1.)
# For v1, the LAYOUT_6ROOM rectangular footprint is enough to drive the
# dimension-extraction demo. Add LAYOUT_LSHAPE in a follow-up.

LAYOUTS = {
    "6room": LAYOUT_6ROOM,
}


# ============================================================
# Geometry helpers
# ============================================================

def mm_to_px(mm: float) -> int:
    return int(round(mm * RENDER_PX_PER_MM))


def plan_to_px(x_mm: float, y_mm: float) -> tuple[int, int]:
    """Convert (x_mm, y_mm) in plan-local mm to page-px coordinates."""
    return (
        PLAN_ORIGIN_PX[0] + mm_to_px(x_mm),
        PLAN_ORIGIN_PX[1] + mm_to_px(y_mm),
    )


def room_polygon_mm(room: Room) -> list[tuple[int, int]]:
    """Rectangular room polygon in mm, clockwise from top-left."""
    return [
        (room.x,            room.y),
        (room.x + room.w,   room.y),
        (room.x + room.w,   room.y + room.h),
        (room.x,            room.y + room.h),
    ]


def room_centre_mm(room: Room) -> tuple[float, float]:
    return (room.x + room.w / 2.0, room.y + room.h / 2.0)


def room_area_m2(room: Room) -> float:
    """Internal area, ignoring wall thickness. mm² -> m²."""
    return (room.w * room.h) / 1_000_000.0


def total_floor_area_m2(rooms: list[Room]) -> float:
    return round(sum(room_area_m2(r) for r in rooms), 2)


# ============================================================
# Drawing primitives
# ============================================================

def _font(size: int) -> ImageFont.ImageFont:
    """Try a few common system fonts; fall back to default."""
    for name in ("arial.ttf", "DejaVuSans.ttf", "Helvetica.ttc"):
        try:
            return ImageFont.truetype(name, size)
        except (OSError, IOError):
            pass
    return ImageFont.load_default()


def _text_size(draw: ImageDraw.ImageDraw, text: str, font) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _draw_room_floor(draw: ImageDraw.ImageDraw, room: Room) -> None:
    """Fill room interior with a soft tint so wet areas are visually distinct."""
    fill = (235, 242, 250) if room.is_wet else (252, 252, 248)
    x0, y0 = plan_to_px(room.x, room.y)
    x1, y1 = plan_to_px(room.x + room.w, room.y + room.h)
    draw.rectangle([x0, y0, x1, y1], fill=fill)


def _draw_walls(draw: ImageDraw.ImageDraw, layout: dict) -> list[dict]:
    """Draw the outer wall + every internal partition. Returns a list of
    wall records for the ground-truth JSON.

    Walls are drawn as thick lines, not filled rectangles, to keep the
    figure clean. The thickness equals WALL_THK_MM scaled to pixels.
    """
    walls: list[dict] = []
    thk_px = max(2, mm_to_px(WALL_THK_MM))

    rooms: list[Room] = layout["rooms"]

    # Outer perimeter rectangle
    W = layout["total_w_mm"]
    H = layout["total_h_mm"]
    outer = [
        ("outer_top",    (0, 0),     (W, 0)),
        ("outer_right",  (W, 0),     (W, H)),
        ("outer_bottom", (W, H),     (0, H)),
        ("outer_left",   (0, H),     (0, 0)),
    ]
    for name, a, b in outer:
        ax, ay = plan_to_px(*a)
        bx, by = plan_to_px(*b)
        draw.line([(ax, ay), (bx, by)], fill=WALL_RGB, width=thk_px)
        walls.append({
            "id": name, "kind": "outer",
            "start_mm": list(a), "end_mm": list(b),
            "length_mm": int(math.hypot(b[0] - a[0], b[1] - a[1])),
        })

    # Internal partitions — dedup the unique room edges that aren't on the perimeter
    seen: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for r in rooms:
        edges = [
            ("h-top",    (r.x,         r.y),         (r.x + r.w, r.y)),
            ("h-bot",    (r.x,         r.y + r.h),   (r.x + r.w, r.y + r.h)),
            ("v-left",   (r.x,         r.y),         (r.x,       r.y + r.h)),
            ("v-right",  (r.x + r.w,   r.y),         (r.x + r.w, r.y + r.h)),
        ]
        for _, a, b in edges:
            # skip if it lies on the outer perimeter
            on_perimeter = (
                (a[0] == 0 and b[0] == 0) or
                (a[0] == W and b[0] == W) or
                (a[1] == 0 and b[1] == 0) or
                (a[1] == H and b[1] == H)
            )
            if on_perimeter:
                continue
            key = (a, b) if a < b else (b, a)
            if key in seen:
                continue
            seen.add(key)
            ax, ay = plan_to_px(*a)
            bx, by = plan_to_px(*b)
            draw.line([(ax, ay), (bx, by)], fill=WALL_RGB, width=thk_px)
            walls.append({
                "id": f"internal_{len(walls)}", "kind": "internal",
                "start_mm": list(a), "end_mm": list(b),
                "length_mm": int(math.hypot(b[0] - a[0], b[1] - a[1])),
            })
    return walls


def _draw_room_labels(draw: ImageDraw.ImageDraw, rooms: list[Room]) -> None:
    label_font = _font(ROOM_LABEL_SIZE)
    dim_font = _font(ROOM_DIM_SIZE)
    for r in rooms:
        cx_mm, cy_mm = room_centre_mm(r)
        cx, cy = plan_to_px(cx_mm, cy_mm)
        # Room name
        tw, th = _text_size(draw, r.name, label_font)
        draw.text((cx - tw // 2, cy - th - 6), r.name,
                  fill=ROOM_LABEL_RGB, font=label_font)
        # Room dimensions caption: "2500 x 4000" in mm
        dim_text = f"{r.w} x {r.h}"
        tw2, th2 = _text_size(draw, dim_text, dim_font)
        draw.text((cx - tw2 // 2, cy + 4), dim_text,
                  fill=ROOM_LABEL_RGB, font=dim_font)


def _draw_dimensions_along_edge(
    draw: ImageDraw.ImageDraw,
    segments_mm: list[int],
    edge: str,
    layout: dict,
) -> list[dict]:
    """Draw a chain of dimension annotations along one outer edge.

    `segments_mm` is the list of segment lengths from one end of the edge
    to the other (e.g. [2500, 2500, 2500, 2500] for the top edge of the
    6-room layout). `edge` is 'top', 'bottom', 'left', 'right'.

    Returns a list of dimension records for the ground-truth JSON.
    """
    font = _font(DIM_TEXT_SIZE)
    W = layout["total_w_mm"]
    H = layout["total_h_mm"]
    records: list[dict] = []

    # Edge runs along axis `along`, offset perpendicular to `perp`
    if edge == "top":
        along_axis = "x"
        perp_dir = -1     # offset upward
        anchor_mm = (0, 0)        # start of edge
        end_mm = (W, 0)
    elif edge == "bottom":
        along_axis = "x"
        perp_dir = 1      # offset downward
        anchor_mm = (0, H)
        end_mm = (W, H)
    elif edge == "left":
        along_axis = "y"
        perp_dir = -1     # offset leftward
        anchor_mm = (0, 0)
        end_mm = (0, H)
    elif edge == "right":
        along_axis = "y"
        perp_dir = 1      # offset rightward
        anchor_mm = (W, 0)
        end_mm = (W, H)
    else:
        return []

    # Verify the segments sum to the edge length
    total_edge_mm = W if along_axis == "x" else H
    if sum(segments_mm) != total_edge_mm:
        # Soft warning; still draw what we have
        pass

    # Compute the perpendicular offset in pixels
    offset_px = DIM_OFFSET_PX * perp_dir

    # Build the segment-boundary positions in mm along the edge
    positions = [0]
    for s in segments_mm:
        positions.append(positions[-1] + s)
    # positions is e.g. [0, 2500, 5000, 7500, 10000]

    # Draw extension lines from each segment boundary to the dim-line offset
    for pos_mm in positions:
        if along_axis == "x":
            wall_x, wall_y = plan_to_px(pos_mm, anchor_mm[1])
            ext_x, ext_y = wall_x, wall_y + offset_px
        else:
            wall_x, wall_y = plan_to_px(anchor_mm[0], pos_mm)
            ext_x, ext_y = wall_x + offset_px, wall_y
        draw.line([(wall_x, wall_y), (ext_x, ext_y)], fill=DIM_LINE_RGB, width=1)

    # Dimension line running along the edge at the offset distance
    if along_axis == "x":
        a = plan_to_px(0, anchor_mm[1]); a = (a[0], a[1] + offset_px)
        b = plan_to_px(W, anchor_mm[1]); b = (b[0], b[1] + offset_px)
    else:
        a = plan_to_px(anchor_mm[0], 0); a = (a[0] + offset_px, a[1])
        b = plan_to_px(anchor_mm[0], H); b = (b[0] + offset_px, b[1])
    draw.line([a, b], fill=DIM_LINE_RGB, width=1)

    # Tick marks at each end of the chain
    overshoot = DIM_EXT_OVERSHOOT_PX
    for endpoint in (a, b):
        if along_axis == "x":
            draw.line([(endpoint[0], endpoint[1] - overshoot),
                       (endpoint[0], endpoint[1] + overshoot)],
                      fill=DIM_LINE_RGB, width=2)
        else:
            draw.line([(endpoint[0] - overshoot, endpoint[1]),
                       (endpoint[0] + overshoot, endpoint[1])],
                      fill=DIM_LINE_RGB, width=2)

    # Dimension VALUE labels — between each pair of adjacent positions
    for i, value_mm in enumerate(segments_mm):
        mid_mm = (positions[i] + positions[i + 1]) / 2.0
        if along_axis == "x":
            mx, my = plan_to_px(mid_mm, anchor_mm[1])
            mx, my = mx, my + offset_px - (8 if perp_dir < 0 else -22)
        else:
            mx, my = plan_to_px(anchor_mm[0], mid_mm)
            mx, my = mx + offset_px - (8 if perp_dir < 0 else -8), my
        text = str(value_mm)
        tw, th = _text_size(draw, text, font)
        # Horizontal text along x-edges, rotated 90 along y-edges
        if along_axis == "x":
            draw.text((mx - tw // 2, my), text, fill=DIM_TEXT_RGB, font=font)
        else:
            # Vertical text: render onto a small image then rotate
            txt_img = Image.new("RGBA", (tw + 4, th + 4), (0, 0, 0, 0))
            ImageDraw.Draw(txt_img).text((2, 2), text, fill=DIM_TEXT_RGB + (255,), font=font)
            txt_img = txt_img.rotate(90, expand=True)
            # Paste back onto the main canvas at the correct position
            draw._image.paste(txt_img, (mx - txt_img.width // 2,
                                        my - txt_img.height // 2), txt_img)
        records.append({
            "edge": edge,
            "value_mm": int(value_mm),
            "segment_start_mm": int(positions[i]),
            "segment_end_mm":   int(positions[i + 1]),
        })

    return records


def _draw_windows(draw: ImageDraw.ImageDraw, layout: dict) -> list[dict]:
    """Draw every Window on the layout. Returns ground-truth records."""
    W = layout["total_w_mm"]
    H = layout["total_h_mm"]
    thk_px = max(2, mm_to_px(WALL_THK_MM))
    records: list[dict] = []
    font = _font(16)

    for win in layout["windows"]:
        half = win.width_mm / 2.0
        if win.wall == "top":
            a = plan_to_px(win.position_mm - half, 0)
            b = plan_to_px(win.position_mm + half, 0)
            inner_a = (a[0], a[1] + thk_px)
            inner_b = (b[0], b[1] + thk_px)
        elif win.wall == "bottom":
            a = plan_to_px(win.position_mm - half, H)
            b = plan_to_px(win.position_mm + half, H)
            inner_a = (a[0], a[1] - thk_px)
            inner_b = (b[0], b[1] - thk_px)
        elif win.wall == "left":
            a = plan_to_px(0, win.position_mm - half)
            b = plan_to_px(0, win.position_mm + half)
            inner_a = (a[0] + thk_px, a[1])
            inner_b = (b[0] + thk_px, b[1])
        elif win.wall == "right":
            a = plan_to_px(W, win.position_mm - half)
            b = plan_to_px(W, win.position_mm + half)
            inner_a = (a[0] - thk_px, a[1])
            inner_b = (b[0] - thk_px, b[1])
        else:
            continue

        # Mask the wall (paint over with bg colour) then draw the window's
        # two parallel lines representing glass
        draw.line([a, b], fill=BG_RGB, width=thk_px + 2)
        draw.line([a, b], fill=WINDOW_RGB, width=1)
        draw.line([inner_a, inner_b], fill=WINDOW_RGB, width=1)

        # Label
        mx = (a[0] + b[0]) // 2
        my = (a[1] + b[1]) // 2
        if win.label:
            if win.wall in ("top", "bottom"):
                draw.text((mx + 6, my - 8), win.label, fill=WINDOW_RGB, font=font)
            else:
                draw.text((mx - 18, my + 4), win.label, fill=WINDOW_RGB, font=font)

        # Centre point in mm
        if win.wall in ("top", "bottom"):
            centre_mm = (win.position_mm, 0 if win.wall == "top" else H)
        else:
            centre_mm = (0 if win.wall == "left" else W, win.position_mm)

        records.append({
            "label": win.label,
            "wall": win.wall,
            "width_mm": int(win.width_mm),
            "centre_mm": [int(centre_mm[0]), int(centre_mm[1])],
            "centre_px": [int(mx), int(my)],
        })
    return records


def _draw_doors(draw: ImageDraw.ImageDraw, layout: dict) -> list[dict]:
    """Draw every Door on the layout. External doors get a quarter-arc
    swing; glass sliding doors get a double-line indicator."""
    W = layout["total_w_mm"]
    H = layout["total_h_mm"]
    thk_px = max(2, mm_to_px(WALL_THK_MM))
    records: list[dict] = []
    font = _font(16)

    for d in layout["doors"]:
        half = d.width_mm / 2.0
        if d.wall == "top":
            a = plan_to_px(d.position_mm - half, 0)
            b = plan_to_px(d.position_mm + half, 0)
            into_room = (0, 1)  # arc opens downward (into room)
        elif d.wall == "bottom":
            a = plan_to_px(d.position_mm - half, H)
            b = plan_to_px(d.position_mm + half, H)
            into_room = (0, -1)
        elif d.wall == "left":
            a = plan_to_px(0, d.position_mm - half)
            b = plan_to_px(0, d.position_mm + half)
            into_room = (1, 0)
        elif d.wall == "right":
            a = plan_to_px(W, d.position_mm - half)
            b = plan_to_px(W, d.position_mm + half)
            into_room = (-1, 0)
        else:
            continue

        # Cut the wall (paint bg colour over the door opening)
        draw.line([a, b], fill=BG_RGB, width=thk_px + 2)

        if d.is_glass:
            # Glass sliding door: two parallel lines across the opening
            offset = thk_px // 2
            draw.line([a, b], fill=WINDOW_RGB, width=2)
            shift = (into_room[0] * 6, into_room[1] * 6)
            draw.line([(a[0] + shift[0], a[1] + shift[1]),
                       (b[0] + shift[0], b[1] + shift[1])],
                      fill=WINDOW_RGB, width=2)
        else:
            # Standard door: a leaf line + quarter-circle swing arc
            opening_w = abs(b[0] - a[0]) + abs(b[1] - a[1])
            # Door leaf: line from a, perpendicular into the room
            leaf_end = (a[0] + into_room[0] * opening_w,
                        a[1] + into_room[1] * opening_w)
            draw.line([a, leaf_end], fill=DOOR_RGB, width=2)
            # Quarter arc from leaf_end back to b
            # Bounding box of the full circle of radius opening_w centred at a
            arc_box = [a[0] - opening_w, a[1] - opening_w,
                       a[0] + opening_w, a[1] + opening_w]
            # Determine start/end angles
            if d.wall == "top":
                draw.arc(arc_box, start=0,   end=90,  fill=DOOR_RGB, width=1)
            elif d.wall == "bottom":
                draw.arc(arc_box, start=270, end=360, fill=DOOR_RGB, width=1)
            elif d.wall == "left":
                draw.arc(arc_box, start=270, end=360, fill=DOOR_RGB, width=1)
            else:  # right
                draw.arc(arc_box, start=180, end=270, fill=DOOR_RGB, width=1)

        mx = (a[0] + b[0]) // 2
        my = (a[1] + b[1]) // 2
        if d.label:
            draw.text((mx + 6, my + 6), d.label, fill=DOOR_RGB, font=font)

        records.append({
            "label": d.label,
            "wall": d.wall,
            "position_mm": int(d.position_mm),
            "width_mm": int(d.width_mm),
            "is_external": bool(d.is_external),
            "is_glass": bool(d.is_glass),
        })
    return records


def _draw_scale_bar(draw: ImageDraw.ImageDraw) -> dict:
    """Draw a scale bar that reads '1:50' (chosen to match our
    RENDER_PX_PER_MM = 0.16 — see header). Position: top-LEFT of the page,
    in the empty header band above the plan (mirrors the north arrow's
    top-right placement). Avoids the cramped strip between the bottom
    dimension chain and the title block.
    """
    font = _font(20)
    title_font = _font(18)

    bar_x = 80
    bar_y = 110
    bar_total_mm = 3000  # 3 metres bar — fits comfortably in the header
    bar_length_px = mm_to_px(bar_total_mm)
    bar_h = 8
    # Black/white alternating segments, every 1000 mm
    segs = bar_total_mm // 1000
    seg_w = bar_length_px // segs
    for i in range(segs):
        fill = WALL_RGB if i % 2 == 0 else BG_RGB
        x0 = bar_x + i * seg_w
        x1 = bar_x + (i + 1) * seg_w
        draw.rectangle([x0, bar_y, x1, bar_y + bar_h], outline=WALL_RGB, fill=fill)

    # Tick labels: 0, 1m, 2m, 3m
    for i in range(segs + 1):
        x = bar_x + i * seg_w
        draw.line([(x, bar_y - 4), (x, bar_y + bar_h + 4)], fill=WALL_RGB, width=1)
        lbl = "0" if i == 0 else f"{i}m"
        tw, th = _text_size(draw, lbl, font)
        draw.text((x - tw // 2, bar_y + bar_h + 6), lbl, fill=WALL_RGB, font=font)

    draw.text((bar_x, bar_y - 28), "SCALE 1:50", fill=WALL_RGB, font=title_font)
    return {
        "bbox_px": [bar_x, bar_y - 32, bar_x + bar_length_px, bar_y + bar_h + 32],
        "label": "1:50",
        "represents_mm": bar_total_mm,
        "length_px": bar_length_px,
    }


def _draw_north_arrow(draw: ImageDraw.ImageDraw) -> None:
    """North arrow in the top-right corner of the plan area."""
    cx, cy = PAGE_PX[0] - 240, 220
    radius = 36
    draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius],
                 outline=WALL_RGB, width=2)
    # Triangle pointing up
    triangle = [(cx, cy - radius + 6), (cx - 12, cy + 10), (cx + 12, cy + 10)]
    draw.polygon(triangle, fill=WALL_RGB)
    font = _font(20)
    tw, th = _text_size(draw, "N", font)
    draw.text((cx - tw // 2, cy + 12), "N", fill=WALL_RGB, font=font)


def _draw_title_block(
    draw: ImageDraw.ImageDraw, layout: dict, total_area_m2: float, seed: int
) -> None:
    """Title block at the bottom of the page."""
    title_font = _font(28)
    label_font = _font(16)
    val_font = _font(18)

    block_top = PAGE_PX[1] - 130
    block_h = 110
    draw.rectangle([20, block_top, PAGE_PX[0] - 20, block_top + block_h],
                   outline=WALL_RGB, width=2)
    # Vertical dividers
    col1 = PAGE_PX[0] // 3
    col2 = 2 * PAGE_PX[0] // 3
    draw.line([(col1, block_top), (col1, block_top + block_h)],
              fill=WALL_RGB, width=1)
    draw.line([(col2, block_top), (col2, block_top + block_h)],
              fill=WALL_RGB, width=1)

    # Column 1 — project
    draw.text((40, block_top + 12), "PROJECT", fill=WALL_RGB, font=label_font)
    draw.text((40, block_top + 38), f"Synthetic Residence #{seed:04d}",
              fill=WALL_RGB, font=title_font)
    draw.text((40, block_top + 80), "Brunswick East, VIC 3057",
              fill=WALL_RGB, font=val_font)
    # Column 2 — drawing info
    draw.text((col1 + 20, block_top + 12), "DRAWING", fill=WALL_RGB, font=label_font)
    draw.text((col1 + 20, block_top + 38), "PROPOSED GROUND FLOOR PLAN",
              fill=WALL_RGB, font=val_font)
    draw.text((col1 + 20, block_top + 64), f"DRAWING NO.  A-101",
              fill=WALL_RGB, font=val_font)
    draw.text((col1 + 20, block_top + 88), "REV.  A    SCALE  1:50",
              fill=WALL_RGB, font=val_font)
    # Column 3 — area summary (handy for the extraction GT)
    draw.text((col2 + 20, block_top + 12), "AREAS", fill=WALL_RGB, font=label_font)
    draw.text((col2 + 20, block_top + 38), f"Floor area:  {total_area_m2:.1f} m²",
              fill=WALL_RGB, font=val_font)
    draw.text((col2 + 20, block_top + 64),
              f"Rooms: {len(layout['rooms'])}  Wet: "
              f"{sum(1 for r in layout['rooms'] if r.is_wet)}",
              fill=WALL_RGB, font=val_font)
    draw.text((col2 + 20, block_top + 88),
              "POC DEMO — NOT FOR CONSTRUCTION",
              fill=WALL_RGB, font=val_font)


# ============================================================
# Top-level render
# ============================================================

def render(layout_name: str, seed: int, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    layout = LAYOUTS[layout_name]
    random.seed(seed)

    canvas = Image.new("RGB", PAGE_PX, BG_RGB)
    draw = ImageDraw.Draw(canvas)

    # Order matters: floor fills first, then walls on top, then dims/labels,
    # then windows + doors cut into the walls, then chrome.
    for r in layout["rooms"]:
        _draw_room_floor(draw, r)
    walls = _draw_walls(draw, layout)
    _draw_room_labels(draw, layout["rooms"])

    # Dimension chains along each external edge.
    # Top: 4 segments of room widths. Bottom: 2 segments (rooms merged in row 2).
    # Left: 2 segments (4000 + 4000). Right: same.
    top_segments    = [r.w for r in layout["rooms"] if r.y == 0]
    bottom_segments = [r.w for r in layout["rooms"] if r.y + r.h == layout["total_h_mm"]]
    left_segments   = sorted({r.h for r in layout["rooms"] if r.x == 0}, reverse=False)
    # Heuristic: column on the left side has rooms at y=0 and y=4000; build
    # the segment chain from those rooms' heights.
    left_rooms_in_col = sorted([r for r in layout["rooms"] if r.x == 0], key=lambda r: r.y)
    left_segments = [r.h for r in left_rooms_in_col]
    right_rooms_in_col = sorted(
        [r for r in layout["rooms"] if r.x + r.w == layout["total_w_mm"]],
        key=lambda r: r.y,
    )
    right_segments = [r.h for r in right_rooms_in_col]

    dims = (
        [{"edge": "top",    **d} for d in
         _draw_dimensions_along_edge(draw, top_segments,    "top",    layout)] +
        [{"edge": "bottom", **d} for d in
         _draw_dimensions_along_edge(draw, bottom_segments, "bottom", layout)] +
        [{"edge": "left",   **d} for d in
         _draw_dimensions_along_edge(draw, left_segments,   "left",   layout)] +
        [{"edge": "right",  **d} for d in
         _draw_dimensions_along_edge(draw, right_segments,  "right",  layout)]
    )
    # _draw_dimensions_along_edge returns records already including "edge"
    # — flatten to dedupe.
    seen_keys: set[tuple] = set()
    dims_dedup: list[dict] = []
    for d in dims:
        k = (d["edge"], d["value_mm"], d["segment_start_mm"], d["segment_end_mm"])
        if k in seen_keys:
            continue
        seen_keys.add(k)
        dims_dedup.append({k_: v_ for k_, v_ in d.items() if k_ != "edge"} | {"edge": d["edge"]})
    dims = dims_dedup

    windows = _draw_windows(draw, layout)
    doors   = _draw_doors(draw, layout)
    scale_bar = _draw_scale_bar(draw)
    _draw_north_arrow(draw)
    total_area = total_floor_area_m2(layout["rooms"])
    _draw_title_block(draw, layout, total_area, seed)

    # Save outputs
    stem = f"measurement_sample_{layout_name}_seed{seed:04d}"
    png_path = out_dir / f"{stem}.png"
    pdf_path = out_dir / f"{stem}.pdf"
    gt_path  = out_dir / f"{stem}_ground_truth.json"
    canvas.save(png_path, "PNG", optimize=True)
    canvas.save(pdf_path, "PDF", resolution=150.0)

    gt = {
        "schema_version": "measurement-v1",
        "layout_name": layout_name,
        "seed": seed,
        "page_px": list(PAGE_PX),
        "plan_origin_px": list(PLAN_ORIGIN_PX),
        "render_px_per_mm": RENDER_PX_PER_MM,
        "total_w_mm": layout["total_w_mm"],
        "total_h_mm": layout["total_h_mm"],
        "total_floor_area_m2": total_area,
        "wall_thickness_mm": WALL_THK_MM,
        "rooms": [
            {
                "name": r.name,
                "polygon_mm": room_polygon_mm(r),
                "centre_mm": list(room_centre_mm(r)),
                "centre_px": list(plan_to_px(*room_centre_mm(r))),
                "w_mm": r.w, "h_mm": r.h,
                "area_m2": round(room_area_m2(r), 2),
                "is_wet": r.is_wet,
            }
            for r in layout["rooms"]
        ],
        "walls": walls,
        "dimensions": dims,
        "windows": windows,
        "doors": doors,
        "scale_bar": scale_bar,
    }
    gt_path.write_text(json.dumps(gt, indent=2), encoding="utf-8")
    return {"png": png_path, "pdf": pdf_path, "gt": gt_path,
            "total_area_m2": total_area}


# ============================================================
# CLI
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate measurement-focused floor-plan PDF.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed (parametrises variants).")
    parser.add_argument("--layout", choices=list(LAYOUTS.keys()), default="6room",
                        help="Which layout template to render.")
    parser.add_argument("--out", type=Path, default=Path("samples"),
                        help="Output directory (relative to project root).")
    args = parser.parse_args()

    HERE = Path(__file__).resolve().parent.parent.parent
    out = HERE / args.out
    result = render(args.layout, args.seed, out)
    print(f"PDF:           {result['pdf']}")
    print(f"PNG:           {result['png']}")
    print(f"Ground truth:  {result['gt']}")
    print(f"Floor area:    {result['total_area_m2']:.1f} m²")


if __name__ == "__main__":
    main()
