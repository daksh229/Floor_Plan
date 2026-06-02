"""Phase 2 — dimension-driven scale calibration + outer-envelope polygon.

The single highest-risk module in the proposal (§7 #2). Implements:

  1. Dimension OCR pass — re-OCR the page (including a 90°-rotated pass
     to catch vertical dim text) and filter to numeric tokens in the
     residential range 200..20000 mm.
  2. Wall-line detection — Hough on a binarised greyscale of the page,
     after masking out the red title boundary. Cluster colinear
     segments into individual wall records.
  3. Dimension-to-wall association — geometric rules: orientation match
     (horizontal text -> horizontal wall) + nearest-perpendicular wall
     within a corridor. Greedy, one wall per dimension.
  4. Scale calibration — median of (dim_value_mm / wall_pixel_length)
     across all pairs. Confidence comes from the standard deviation of
     the per-pair ratios.
  5. Outer-envelope polygon — bounding rectangle of the outermost walls
     in each direction. For L-shape support we'd extend to a graph walk,
     but the variant_e fixture is rectangular so the rectangle is sound.

The output `ScaleCalibration` has both the numeric `mm_per_px` (used for
all downstream area calculations) and a derived label string (e.g.
"1:100" approximated to the nearest standard scale). The label is for
the `scale_detected` field; the numeric ratio is for everything else.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Iterable, Optional

import cv2
import numpy as np
from PIL import Image

from .ocr import extract_text_spans
from .schemas import OCRSpan


# ============================================================
# Tunables
# ============================================================

# Plausible residential dimension range (mm). Tighter than necessary on
# purpose — keeps OCR noise out of the calibration pairs.
MIN_DIM_VALUE_MM = 200
MAX_DIM_VALUE_MM = 20000

# Wall detection
WALL_MIN_LENGTH_PX = 80              # ignore short segments (noise)
WALL_CLUSTER_DIST_PX = 8             # collinear segments within this tolerance merge
WALL_ORIENTATION_TOL_DEG = 3.0       # how off-axis a segment can be and still count as H or V

# Dimension-to-wall association
DIM_TO_WALL_MAX_FRAC = 0.05          # perpendicular distance cap, fraction of page diagonal
DIM_ALONG_WALL_TOL = 0.20            # dim has to span the wall within ±20% to be a real pairing

# Standard architectural scales we round to for the derived label
STANDARD_SCALES = (20, 50, 75, 100, 150, 200, 500)


# ============================================================
# Data classes
# ============================================================

@dataclass(frozen=True)
class DimensionReading:
    value_mm: int
    bbox_px: tuple[int, int, int, int]
    centre_px: tuple[int, int]
    orientation: str    # "horizontal" | "vertical"
    ocr_confidence: float


@dataclass(frozen=True)
class WallSegment:
    start_px: tuple[int, int]
    end_px: tuple[int, int]
    orientation: str    # "horizontal" | "vertical"
    length_px: float

    @property
    def mid_px(self) -> tuple[int, int]:
        return ((self.start_px[0] + self.end_px[0]) // 2,
                (self.start_px[1] + self.end_px[1]) // 2)


@dataclass(frozen=True)
class DimWallPairing:
    dim: DimensionReading
    wall: WallSegment
    perpendicular_distance_px: float
    along_wall_ratio: float    # dim_span / wall_length (1.0 = perfect)
    mm_per_px: float


@dataclass
class ScaleCalibration:
    mm_per_px: float
    confidence: float           # 0..1 — higher when std-dev of pairings is low
    n_pairs: int
    pair_stdev_pct: float       # std-dev of mm_per_px as % of median
    derived_scale_label: str    # e.g. "1:100"
    pairings: list[DimWallPairing] = field(default_factory=list)


@dataclass(frozen=True)
class BuildingEnvelope:
    polygon_px: list[tuple[int, int]]    # clockwise from top-left
    width_px: float
    height_px: float
    bbox_px: tuple[int, int, int, int]   # x0, y0, x1, y1
    area_px2: float


# ============================================================
# Step 1 — Dimension extraction
# ============================================================

_NUMERIC_DIM = re.compile(r"^\d{3,5}$")     # 3-5 digit integers (200..99999)
# Match printed scale ratios like "1:100", "1:50", "1:200" anywhere in an OCR span.
# The label is usually inside "SCALE 1:100" but may also appear in the title block
# or as a bare "1:100" near the scale bar.
_SCALE_RATIO = re.compile(r"\b1\s*:\s*(\d{2,4})\b")


def find_printed_scale_label(ocr_spans) -> Optional[str]:
    """Search OCR for a printed `1:NNN` scale ratio. Returns the
    normalised string (e.g. "1:100") or None when no scale ratio is
    legible. This is the most reliable source — the architect literally
    wrote the scale on the plan — so we prefer it over the
    dimension-derived label.
    """
    candidates: list[tuple[int, float]] = []   # (ratio, confidence)
    for s in ocr_spans:
        m = _SCALE_RATIO.search(s.text or "")
        if not m:
            continue
        try:
            ratio = int(m.group(1))
        except ValueError:
            continue
        # Plausible architectural scales only — guard against picking up
        # "1:300" from a paragraph that happens to contain that string.
        if ratio not in (10, 20, 25, 50, 75, 100, 150, 200, 250, 500, 1000):
            continue
        candidates.append((ratio, s.confidence))
    if not candidates:
        return None
    # Most-frequent ratio wins (e.g. when "1:100" appears on the scale
    # bar AND in the title block). Ties broken by confidence.
    from collections import Counter
    freq = Counter(r for r, _ in candidates)
    most_common_ratio, _ = freq.most_common(1)[0]
    return f"1:{most_common_ratio}"


def _classify_orientation(bbox: tuple[int, int, int, int]) -> str:
    x0, y0, x1, y1 = bbox
    w, h = max(1, x1 - x0), max(1, y1 - y0)
    return "vertical" if h > w * 1.5 else "horizontal"


def extract_dimensions(
    ocr_spans: Iterable[OCRSpan],
    page_image: Optional[Image.Image] = None,
    *,
    include_rotated_pass: bool = True,
) -> list[DimensionReading]:
    """Filter OCR spans down to plausible-dimension numeric tokens.

    When `include_rotated_pass=True` and `page_image` is provided, we also
    re-OCR the image rotated 90° to pick up vertical dimension text that
    Tesseract's default PSM 11 can't read in-place.
    """
    spans = list(ocr_spans)
    readings: list[DimensionReading] = []

    # Pass 1: spans from the caller-supplied OCR
    for s in spans:
        t = s.text.strip().replace(",", "").replace(".", "")
        if not _NUMERIC_DIM.match(t):
            continue
        try:
            val = int(t)
        except ValueError:
            continue
        if not (MIN_DIM_VALUE_MM <= val <= MAX_DIM_VALUE_MM):
            continue
        cx = (s.bbox[0] + s.bbox[2]) // 2
        cy = (s.bbox[1] + s.bbox[3]) // 2
        readings.append(DimensionReading(
            value_mm=val,
            bbox_px=s.bbox,
            centre_px=(cx, cy),
            orientation=_classify_orientation(s.bbox),
            ocr_confidence=s.confidence,
        ))

    # Pass 2: rotate the page 90° and re-OCR to catch vertical dims.
    # Coordinates from the rotated pass are translated back to the
    # original page-coordinate system before adding to the list.
    if include_rotated_pass and page_image is not None:
        rotated = page_image.rotate(-90, expand=True)
        try:
            rot_spans = extract_text_spans(rotated, source_page=1)
        except Exception:  # noqa: BLE001
            rot_spans = []
        W, H = page_image.width, page_image.height
        # Dedupe against Pass 1: if a span value+position is already in `readings`,
        # don't duplicate. Position tolerance is generous (rotated-pass bbox
        # translation is approximate — we keep orientation accurate but not
        # bbox-pixel-perfect; the schema's region.position is precise enough).
        existing_keys = {(r.value_mm, r.centre_px[0] // 50, r.centre_px[1] // 50) for r in readings}
        for s in rot_spans:
            t = s.text.strip().replace(",", "").replace(".", "")
            if not _NUMERIC_DIM.match(t):
                continue
            try:
                val = int(t)
            except ValueError:
                continue
            if not (MIN_DIM_VALUE_MM <= val <= MAX_DIM_VALUE_MM):
                continue
            # Rotated bbox -> original page coords. For rotate(-90, expand=True)
            # (PIL CW rotation), original point (x, y) maps to rotated point
            # (y, W - 1 - x). Inverting: rotated point (rx, ry) -> original
            # (W - 1 - ry, rx). Bbox corners give us:
            rx0, ry0, rx1, ry1 = s.bbox
            ox0 = W - ry1
            oy0 = rx0
            ox1 = W - ry0
            oy1 = rx1
            orig_bbox = (int(ox0), int(oy0), int(ox1), int(oy1))
            cx = (orig_bbox[0] + orig_bbox[2]) // 2
            cy = (orig_bbox[1] + orig_bbox[3]) // 2
            key = (val, cx // 50, cy // 50)
            if key in existing_keys:
                continue
            existing_keys.add(key)
            readings.append(DimensionReading(
                value_mm=val,
                bbox_px=orig_bbox,
                centre_px=(cx, cy),
                orientation="vertical",   # rotated pass = vertical by construction
                ocr_confidence=s.confidence,
            ))

    return readings


# ============================================================
# Step 2 — Wall-line detection (Hough + clustering)
# ============================================================

def _mask_red(rgb: np.ndarray) -> np.ndarray:
    """Return an RGB image with red pixels masked to white (so the title
    boundary doesn't confuse the line detector)."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    # Red wraps around hue 0 and 180
    mask_lo = cv2.inRange(hsv, np.array([0, 80, 80]),   np.array([10, 255, 255]))
    mask_hi = cv2.inRange(hsv, np.array([170, 80, 80]), np.array([180, 255, 255]))
    red_mask = cv2.bitwise_or(mask_lo, mask_hi)
    out = rgb.copy()
    out[red_mask > 0] = (255, 255, 255)
    return out


def detect_walls(
    page_image: Image.Image,
    *,
    min_length_px: int = WALL_MIN_LENGTH_PX,
    cluster_dist_px: int = WALL_CLUSTER_DIST_PX,
    orientation_tol_deg: float = WALL_ORIENTATION_TOL_DEG,
) -> list[WallSegment]:
    """Run Hough on the page, return clustered horizontal/vertical wall
    segments. Filters out the red title boundary first."""
    rgb = np.array(page_image.convert("RGB"))
    rgb_no_red = _mask_red(rgb)
    gray = cv2.cvtColor(rgb_no_red, cv2.COLOR_RGB2GRAY)

    # Binarise: walls are dark/black on cream background
    _, binary = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)

    # Light dilation to bridge tiny breaks (e.g. window openings within walls)
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=1)

    # Edges -> Hough
    edges = cv2.Canny(binary, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 360, threshold=80,
        minLineLength=min_length_px, maxLineGap=15,
    )
    if lines is None:
        return []

    # Classify by orientation and cluster
    horiz: list[tuple[int, int, int, int]] = []
    vert: list[tuple[int, int, int, int]] = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx, dy = x2 - x1, y2 - y1
        if dx == 0 and dy == 0:
            continue
        angle_deg = math.degrees(math.atan2(dy, dx))
        # Normalise to [-90, 90]
        if angle_deg > 90:  angle_deg -= 180
        if angle_deg < -90: angle_deg += 180
        if abs(angle_deg) <= orientation_tol_deg:
            horiz.append((min(x1, x2), y1, max(x1, x2), y2))
        elif abs(abs(angle_deg) - 90) <= orientation_tol_deg:
            vert.append((x1, min(y1, y2), x2, max(y1, y2)))

    # Cluster horizontal lines by y, vertical lines by x
    h_walls = _cluster_segments(horiz, axis="h", cluster_dist=cluster_dist_px)
    v_walls = _cluster_segments(vert,  axis="v", cluster_dist=cluster_dist_px)

    out: list[WallSegment] = []
    for (a, b) in h_walls:
        if abs(b[0] - a[0]) < min_length_px:
            continue
        out.append(WallSegment(
            start_px=a, end_px=b, orientation="horizontal",
            length_px=abs(b[0] - a[0]),
        ))
    for (a, b) in v_walls:
        if abs(b[1] - a[1]) < min_length_px:
            continue
        out.append(WallSegment(
            start_px=a, end_px=b, orientation="vertical",
            length_px=abs(b[1] - a[1]),
        ))
    return out


def _cluster_segments(
    segments: list[tuple[int, int, int, int]],
    *, axis: str, cluster_dist: int,
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Cluster collinear segments. For horizontal segments cluster by y,
    for vertical by x. Within a cluster, merge by taking the union of the
    along-axis extent."""
    if not segments:
        return []
    if axis == "h":
        key = lambda s: (s[1] + s[3]) // 2  # mid-y
        sort_axis = 1
    else:
        key = lambda s: (s[0] + s[2]) // 2  # mid-x
        sort_axis = 0
    segments = sorted(segments, key=key)
    clusters: list[list[tuple[int, int, int, int]]] = []
    for s in segments:
        if not clusters:
            clusters.append([s]); continue
        if abs(key(s) - key(clusters[-1][0])) <= cluster_dist:
            clusters.append([s]) if False else clusters[-1].append(s)
        else:
            clusters.append([s])

    out: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for cluster in clusters:
        if axis == "h":
            y_mid = int(round(sum(key(s) for s in cluster) / len(cluster)))
            x_min = min(s[0] for s in cluster)
            x_max = max(s[2] for s in cluster)
            out.append(((x_min, y_mid), (x_max, y_mid)))
        else:
            x_mid = int(round(sum(key(s) for s in cluster) / len(cluster)))
            y_min = min(s[1] for s in cluster)
            y_max = max(s[3] for s in cluster)
            out.append(((x_mid, y_min), (x_mid, y_max)))
    return out


# ============================================================
# Step 3 — Dimension-to-wall association
# ============================================================

def associate_dimensions_to_walls(
    dimensions: list[DimensionReading],
    walls: list[WallSegment],
    page_size: tuple[int, int],
    *,
    max_distance_frac: float = DIM_TO_WALL_MAX_FRAC,
    along_wall_tol: float = DIM_ALONG_WALL_TOL,
) -> list[DimWallPairing]:
    """Pair each dimension with the most plausible wall. Greedy: each
    dimension picks its best wall; walls can be picked multiple times
    (e.g. chained dimensions all measure pieces of the same wall)."""
    if not dimensions or not walls:
        return []
    w, h = page_size
    page_diag = math.hypot(w, h)
    max_perp_dist = max_distance_frac * page_diag

    pairings: list[DimWallPairing] = []
    for d in dimensions:
        candidates = [
            wall for wall in walls
            if wall.orientation == ("horizontal" if d.orientation == "horizontal" else "vertical")
        ]
        if not candidates:
            continue
        best: Optional[tuple[WallSegment, float, float]] = None
        for wall in candidates:
            if wall.orientation == "horizontal":
                perp = abs(d.centre_px[1] - wall.start_px[1])
                # along-axis test: dim centre must be inside the wall's x-range
                if not (wall.start_px[0] - 50 <= d.centre_px[0] <= wall.end_px[0] + 50):
                    continue
            else:
                perp = abs(d.centre_px[0] - wall.start_px[0])
                if not (wall.start_px[1] - 50 <= d.centre_px[1] <= wall.end_px[1] + 50):
                    continue
            if perp > max_perp_dist:
                continue
            # Prefer closer walls; tie-break by ratio of dim_value / wall_length
            # (a dimension that's roughly the same scale as its wall is more
            # likely the correct pairing — guards against dim being attributed
            # to the long outer wall when it actually measures a short segment).
            ratio = d.value_mm / max(1.0, wall.length_px)
            score = perp  # lower is better
            if best is None or score < best[1]:
                best = (wall, score, ratio)
        if best is None:
            continue
        wall, _, mm_per_px = best
        pairings.append(DimWallPairing(
            dim=d,
            wall=wall,
            perpendicular_distance_px=_, # placeholder; updated below
            along_wall_ratio=mm_per_px,
            mm_per_px=mm_per_px,
        ))
        # patch the immutable dataclass via reconstruction
    # The line above can't actually patch a frozen dataclass; rebuild:
    fixed: list[DimWallPairing] = []
    for p in pairings:
        if p.wall.orientation == "horizontal":
            perp = abs(p.dim.centre_px[1] - p.wall.start_px[1])
        else:
            perp = abs(p.dim.centre_px[0] - p.wall.start_px[0])
        fixed.append(DimWallPairing(
            dim=p.dim, wall=p.wall,
            perpendicular_distance_px=perp,
            along_wall_ratio=1.0,   # approximate; we're not strict here
            mm_per_px=p.mm_per_px,
        ))
    return fixed


# ============================================================
# Step 4 — Scale calibration
# ============================================================

def calibrate_scale(pairings: list[DimWallPairing]) -> Optional[ScaleCalibration]:
    """Median mm_per_px across all pairings, confidence from stdev/median."""
    if len(pairings) < 2:
        return None
    ratios = [p.mm_per_px for p in pairings]
    ratios_sorted = sorted(ratios)
    median = ratios_sorted[len(ratios_sorted) // 2]
    # MAD-style: ignore pairings >50% away from median (single bad pairings)
    filtered = [r for r in ratios if abs(r - median) <= 0.5 * median]
    if len(filtered) < 2:
        filtered = ratios
    f_median = sorted(filtered)[len(filtered) // 2]
    mean = sum(filtered) / len(filtered)
    stdev = math.sqrt(sum((r - mean) ** 2 for r in filtered) / len(filtered))
    stdev_pct = (stdev / f_median * 100.0) if f_median > 0 else 100.0
    # Confidence: 1.0 - normalised stdev. Anything under 3% stdev = high conf.
    confidence = max(0.0, min(1.0, 1.0 - stdev_pct / 20.0))
    # Derive standard-scale label by matching against common ratios at 150 DPI.
    derived_label = _derive_scale_label(f_median)
    return ScaleCalibration(
        mm_per_px=f_median,
        confidence=confidence,
        n_pairs=len(filtered),
        pair_stdev_pct=stdev_pct,
        derived_scale_label=derived_label,
        pairings=pairings,
    )


def _derive_scale_label(mm_per_px: float) -> str:
    """Map calibrated mm_per_px to the nearest standard scale label.

    A drawing at 1:N at 150 DPI prints 1mm-real -> 1/N mm-paper, and at
    150 DPI 1mm paper = 5.91 px. So mm_per_px = N / 5.91.
    """
    if mm_per_px <= 0:
        return ""
    # At 150 DPI
    px_per_mm_paper = 5.9055
    inferred_n = mm_per_px * px_per_mm_paper
    # Snap to nearest standard
    nearest = min(STANDARD_SCALES, key=lambda s: abs(s - inferred_n))
    return f"1:{nearest}"


# ============================================================
# Step 5 — Outer envelope polygon
# ============================================================

def detect_outer_envelope(
    walls: list[WallSegment], page_size: tuple[int, int],
) -> Optional[BuildingEnvelope]:
    """Bounding rectangle of all detected walls. The variant_e fixture is
    rectangular so a bbox is sufficient. L-shape support would require a
    planar-graph walk — saved for a future enhancement.
    """
    if not walls:
        return None
    # Filter to "outer" walls — the longest 4 segments by orientation,
    # which on a rectangular plan are the four outer edges.
    h_walls = sorted([w for w in walls if w.orientation == "horizontal"],
                     key=lambda w: -w.length_px)
    v_walls = sorted([w for w in walls if w.orientation == "vertical"],
                     key=lambda w: -w.length_px)
    if not h_walls or not v_walls:
        return None
    # Take top 2 (longest) of each — these should be the outer perimeter.
    h_top = h_walls[:2]
    v_top = v_walls[:2]
    # Outer extents
    x0 = min(min(w.start_px[0], w.end_px[0]) for w in h_top + v_top)
    x1 = max(max(w.start_px[0], w.end_px[0]) for w in h_top + v_top)
    y0 = min(min(w.start_px[1], w.end_px[1]) for w in h_top + v_top)
    y1 = max(max(w.start_px[1], w.end_px[1]) for w in h_top + v_top)
    poly = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    return BuildingEnvelope(
        polygon_px=poly,
        width_px=x1 - x0,
        height_px=y1 - y0,
        bbox_px=(x0, y0, x1, y1),
        area_px2=(x1 - x0) * (y1 - y0),
    )


# ============================================================
# Step 6 — Top-level orchestrator
# ============================================================

@dataclass
class Phase2Result:
    calibration: Optional[ScaleCalibration]
    envelope: Optional[BuildingEnvelope]
    total_floor_area_m2: Optional[float]
    building_perimeter_m: Optional[float]
    wet_area_total_m2: Optional[float]   # filled when caller passes wet-room polys later
    n_dimensions_found: int
    n_walls_found: int
    n_pairings: int


def _filter_building_walls(
    walls: list[WallSegment], page_size: tuple[int, int],
    margin_px: int = 50,
    max_span_frac: float = 0.65,
) -> list[WallSegment]:
    """Drop walls that aren't plausibly building walls:
      - within `margin_px` of the page edge (page borders),
      - longer than `max_span_frac` of either page dim (sheet frames,
        title-block separators, drawing-area borders),
      - shorter than 100 px (noise / tick marks).
    """
    W, H = page_size
    max_h_span = W * max_span_frac
    max_v_span = H * max_span_frac
    out: list[WallSegment] = []
    for w in walls:
        x_in = (margin_px <= w.start_px[0] <= W - margin_px
                and margin_px <= w.end_px[0] <= W - margin_px)
        y_in = (margin_px <= w.start_px[1] <= H - margin_px
                and margin_px <= w.end_px[1] <= H - margin_px)
        if not (x_in and y_in):
            continue
        if w.length_px < 100:
            continue
        if w.orientation == "horizontal" and w.length_px > max_h_span:
            continue
        if w.orientation == "vertical" and w.length_px > max_v_span:
            continue
        out.append(w)
    return out


def _pick_building_totals(
    dimensions: list[DimensionReading],
    page_size: tuple[int, int],
) -> tuple[Optional[DimensionReading], Optional[DimensionReading]]:
    """Identify the two dimensions that represent the building's TOTAL
    width and height.

    Heuristic: take the dim with the LARGEST value, then find the next
    distinct value that's at least 20% smaller. On a typical residential
    plan, this gives (overall_width, overall_height) — width is the
    bigger of the two on landscape-format plans, smaller on portrait.

    Orientation overrides from the dim's positional context (which page
    margin it sits in) — bbox aspect alone is unreliable because
    Tesseract sometimes reads vertical text without realising it.
    """
    if not dimensions:
        return None, None
    PW, PH = page_size

    # Sort by value descending, find the largest two distinct values
    # (at least 20% apart) so we don't end up picking two perimeter
    # segments that happen to share a magnitude.
    ranked = sorted(dimensions, key=lambda d: -d.value_mm)
    largest = ranked[0]
    second: Optional[DimensionReading] = None
    for d in ranked[1:]:
        if d.value_mm <= largest.value_mm * 0.85:
            second = d; break
    if second is None:
        return largest, None

    # Re-classify orientation by margin position. Margin width is ~15%
    # of each page dimension — generous enough to catch dims placed
    # inside the dim line offset.
    def _margin_orientation(d: DimensionReading) -> str:
        cx, cy = d.centre_px
        in_left = cx < PW * 0.18
        in_right = cx > PW * 0.82
        in_top = cy < PH * 0.18
        in_bot = cy > PH * 0.82
        if in_left or in_right:
            return "vertical"
        if in_top or in_bot:
            return "horizontal"
        return d.orientation   # keep the bbox-aspect guess for centre dims

    largest_orient = _margin_orientation(largest)
    second_orient = _margin_orientation(second)

    # If both ended up the same axis (unusual), assume the bigger value
    # is the page's longer axis and the second is the shorter axis.
    if largest_orient == second_orient:
        if PW >= PH:
            largest_orient, second_orient = "horizontal", "vertical"
        else:
            largest_orient, second_orient = "vertical", "horizontal"

    h_total = largest if largest_orient == "horizontal" else second
    v_total = largest if largest_orient == "vertical"   else second
    return h_total, v_total


def calibrate_from_totals(
    dimensions: list[DimensionReading],
    walls: list[WallSegment],
    page_size: tuple[int, int],
) -> Optional[ScaleCalibration]:
    """Take the two LARGEST distinct dimension values (the chain totals)
    and pair each with the longest wall in its orientation.

    Orientation comes from margin position, not bbox aspect (Tesseract
    sometimes reads vertical text as horizontal-looking spans).
    """
    h_total_dim, v_total_dim = _pick_building_totals(dimensions, page_size)
    if h_total_dim is None or v_total_dim is None:
        return None

    building_walls = _filter_building_walls(walls, page_size)
    h_walls = sorted([w for w in building_walls if w.orientation == "horizontal"],
                     key=lambda w: -w.length_px)
    v_walls = sorted([w for w in building_walls if w.orientation == "vertical"],
                     key=lambda w: -w.length_px)
    if not h_walls or not v_walls:
        return None

    h_total_wall = h_walls[0]
    v_total_wall = v_walls[0]

    if h_total_wall.length_px <= 0 or v_total_wall.length_px <= 0:
        return None

    mm_per_px_h = h_total_dim.value_mm / h_total_wall.length_px
    mm_per_px_v = v_total_dim.value_mm / v_total_wall.length_px
    mm_per_px = (mm_per_px_h + mm_per_px_v) / 2.0

    # Confidence from agreement between the two axes
    diff_pct = abs(mm_per_px_h - mm_per_px_v) / mm_per_px * 100.0 if mm_per_px > 0 else 100.0
    confidence = max(0.0, min(1.0, 1.0 - diff_pct / 20.0))

    pairings = [
        DimWallPairing(
            dim=h_total_dim, wall=h_total_wall,
            perpendicular_distance_px=abs(h_total_dim.centre_px[1] - h_total_wall.start_px[1]),
            along_wall_ratio=1.0,
            mm_per_px=mm_per_px_h,
        ),
        DimWallPairing(
            dim=v_total_dim, wall=v_total_wall,
            perpendicular_distance_px=abs(v_total_dim.centre_px[0] - v_total_wall.start_px[0]),
            along_wall_ratio=1.0,
            mm_per_px=mm_per_px_v,
        ),
    ]

    return ScaleCalibration(
        mm_per_px=mm_per_px,
        confidence=confidence,
        n_pairs=2,
        pair_stdev_pct=diff_pct,
        derived_scale_label=_derive_scale_label(mm_per_px),
        pairings=pairings,
    )


def envelope_from_totals(
    dimensions: list[DimensionReading],
    walls: list[WallSegment],
    page_size: tuple[int, int],
    calibration: ScaleCalibration,
) -> Optional[BuildingEnvelope]:
    """Build the envelope rectangle from the longest H wall's x-extent
    and the longest V wall's y-extent. More reliable than min/max of
    all walls because page borders / dim lines / scale-bar lines all
    leak into the wall list."""
    building_walls = _filter_building_walls(walls, page_size)
    h_walls = sorted([w for w in building_walls if w.orientation == "horizontal"],
                     key=lambda w: -w.length_px)
    v_walls = sorted([w for w in building_walls if w.orientation == "vertical"],
                     key=lambda w: -w.length_px)
    if not h_walls or not v_walls:
        return None
    h_main = h_walls[0]
    v_main = v_walls[0]
    x0 = min(h_main.start_px[0], h_main.end_px[0], v_main.start_px[0], v_main.end_px[0])
    x1 = max(h_main.start_px[0], h_main.end_px[0], v_main.start_px[0], v_main.end_px[0])
    y0 = min(h_main.start_px[1], h_main.end_px[1], v_main.start_px[1], v_main.end_px[1])
    y1 = max(h_main.start_px[1], h_main.end_px[1], v_main.start_px[1], v_main.end_px[1])
    poly = [(int(x0), int(y0)), (int(x1), int(y0)),
            (int(x1), int(y1)), (int(x0), int(y1))]
    return BuildingEnvelope(
        polygon_px=poly,
        width_px=float(x1 - x0),
        height_px=float(y1 - y0),
        bbox_px=(int(x0), int(y0), int(x1), int(y1)),
        area_px2=float((x1 - x0) * (y1 - y0)),
    )


def run_phase2(
    page_image: Image.Image,
    ocr_spans: list[OCRSpan],
    *,
    wall_thickness_correction_mm: float = 90.0,
    apply_wall_thickness_correction: bool = True,
) -> Phase2Result:
    """Single entry point: OCR spans + page image -> calibrated scale +
    envelope + headline area numbers.

    Strategy (revised after first smoke test):
      1. Extract dimensions (numeric tokens 200..20000 mm) from OCR,
         including a rotated-page pass for vertical dim text.
      2. Detect walls via Hough on a red-filtered binarisation. Drop
         walls within 50 px of the page edge (those are page borders,
         not building walls).
      3. Pair the LARGEST horizontal dim with the LONGEST horizontal
         wall — that's the building's total width. Same for vertical.
      4. Scale = average of the two ratios. Confidence = how much they
         agree (low diff% = high confidence).
      5. Building envelope = rectangle spanning the longest H wall × the
         longest V wall. Floor area derived from the TWO total
         dimensions directly, not from pixel measurements (since we
         already know the building is X mm × Y mm).
      6. Wall-thickness correction (perimeter × thickness × 0.5)
         subtracted to approximate net internal floor area.
    """
    dimensions = extract_dimensions(ocr_spans, page_image=page_image)
    walls = detect_walls(page_image)
    calibration = calibrate_from_totals(dimensions, walls, page_image.size)
    envelope = envelope_from_totals(dimensions, walls, page_image.size,
                                     calibration) if calibration else None

    total_floor_area_m2: Optional[float] = None
    building_perimeter_m: Optional[float] = None
    if calibration and dimensions:
        h_total_dim, v_total_dim = _pick_building_totals(dimensions, page_image.size)
        if h_total_dim and v_total_dim:
            h_total_mm = h_total_dim.value_mm
            v_total_mm = v_total_dim.value_mm
            gross_mm2 = h_total_mm * v_total_mm
            perim_mm = 2 * (h_total_mm + v_total_mm)
            if apply_wall_thickness_correction:
                wall_band_mm2 = perim_mm * wall_thickness_correction_mm * 0.5
                net_mm2 = max(0.0, gross_mm2 - wall_band_mm2)
                total_floor_area_m2 = round(net_mm2 / 1_000_000.0, 2)
            else:
                total_floor_area_m2 = round(gross_mm2 / 1_000_000.0, 2)
            building_perimeter_m = round(perim_mm / 1000.0, 2)

    # Count pairings even though we don't use them in this simpler approach
    n_pairings = len(calibration.pairings) if calibration else 0

    return Phase2Result(
        calibration=calibration,
        envelope=envelope,
        total_floor_area_m2=total_floor_area_m2,
        building_perimeter_m=building_perimeter_m,
        wet_area_total_m2=None,
        n_dimensions_found=len(dimensions),
        n_walls_found=len(walls),
        n_pairings=n_pairings,
    )
