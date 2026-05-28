"""Multi-scale, multi-rotation template matching for electrical symbols.

Each symbol PNG (alpha-transparent) is the reference template. For every
combination of scale and rotation, run cv2.matchTemplate against the target
page, threshold per class, then NMS within each class to dedupe overlapping
hits across scales / rotations.

We composite each template onto the same cream background as the synth PDF
before matching, so dense TM_CCOEFF_NORMED works correctly without needing
mask-based matching (which is finicky across OpenCV versions).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image

from .schemas import Detection


# Templates are composited onto WHITE before matching. This is intentional:
# a 3-unit difference between cream (252,252,250) and white (255,255,255) is
# negligible for TM_CCOEFF_NORMED but white is the universally-correct choice
# for real-world PDFs where backgrounds vary. (Was cream-tuned to the synth
# fixture; that coupling is the flaw we're fixing here.)
DEFAULT_BG_RGB: tuple[int, int, int] = (255, 255, 255)
DEFAULT_SCALES: tuple[float, ...] = (0.70, 0.85, 1.00, 1.15, 1.30)
DEFAULT_ROTATIONS: tuple[int, ...] = (0, 90, 180, 270)
DEFAULT_THRESHOLD: float = 0.60
DEFAULT_NMS_IOU: float = 0.45          # 0.30 was killing adjacent same-class symbols
CROSS_CLASS_NMS_IOU: float = 0.55      # 0.45 was killing legitimate neighbour pairs

# Per-class thresholds tuned against the synthetic fixture.
# Composite symbols (wp_gpo, distribution_board) match weaker than their
# simpler cousins; we lower their threshold and lean on cross-class NMS to
# resolve "single_gpo inside wp_gpo" style collisions in favour of the more
# specific class.
PER_CLASS_THRESHOLD: dict[str, float] = {
    "downlight": 0.82,           # solid filled circle, very generic
    "wall_light": 0.58,
    "single_pole_switch": 0.60,
    "smoke_detector": 0.50,
    "dimmer": 0.68,
    "data_point": 0.62,
    "tv_point": 0.62,
    "single_gpo": 0.75,          # heavily over-fires inside double_gpo
    "ceiling_fan": 0.70,
    "wp_gpo": 0.60,              # new distinctive design scores well; was 0.42 for old design
    "exhaust_fan": 0.55,
    "double_gpo": 0.58,
    "ceiling_light": 0.55,
    "two_way_switch": 0.55,
}

# Classes that score *higher* should beat classes that score lower in cross-class
# NMS, but specificity matters: a wp_gpo hit should beat an overlapping single_gpo
# hit even if absolute scores are close. These bonuses are added to the score
# only during cross-class NMS ranking (not stored).
CROSS_CLASS_SPECIFICITY_BONUS: dict[str, float] = {
    "wp_gpo": 0.10,
    "double_gpo": 0.05,
    "distribution_board": 0.05,
    "two_way_switch": 0.03,
    "exhaust_fan": 0.03,
    "smoke_detector": 0.03,
    "dimmer": 0.04,           # specific "DIM" label — protect from NMS by neighbours
    "wp_general_purpose_outlet": 0.10,  # alias for wp_gpo when user pack uses long name
}


@dataclass
class _Raw:
    symbol_class: str
    score: float
    bbox: tuple[int, int, int, int]
    rotation_deg: int
    scale: float


# ---------- helpers ----------

def _pil_to_gray(img: Image.Image) -> np.ndarray:
    arr = np.array(img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)


def _composite_on_bg(rgba: Image.Image, bg: tuple[int, int, int]) -> Image.Image:
    bg_img = Image.new("RGB", rgba.size, bg)
    bg_img.paste(rgba, (0, 0), rgba)
    return bg_img


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    iw = max(0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _nms_within_class(raws: list[_Raw], iou_thresh: float) -> list[_Raw]:
    by_class: dict[str, list[_Raw]] = {}
    for r in raws:
        by_class.setdefault(r.symbol_class, []).append(r)
    kept: list[_Raw] = []
    for cls_dets in by_class.values():
        cls_dets.sort(key=lambda d: d.score, reverse=True)
        survivors: list[_Raw] = []
        for d in cls_dets:
            if all(_iou(d.bbox, s.bbox) <= iou_thresh for s in survivors):
                survivors.append(d)
        kept.extend(survivors)
    return kept


def _nms_across_classes(raws: list[_Raw], iou_thresh: float) -> list[_Raw]:
    """Suppress overlapping detections of different classes (e.g. single_gpo
    swallowed by an overlapping wp_gpo). Ranking is score + specificity bonus
    so visually-more-specific symbols win ties.
    """
    def rank(r: _Raw) -> float:
        return r.score + CROSS_CLASS_SPECIFICITY_BONUS.get(r.symbol_class, 0.0)

    ordered = sorted(raws, key=rank, reverse=True)
    survivors: list[_Raw] = []
    for d in ordered:
        if all(_iou(d.bbox, s.bbox) <= iou_thresh for s in survivors):
            survivors.append(d)
    return survivors


# ---------- template preparation ----------

def _resolve_symbol_pngs(
    symbol_dir: Path | None = None,
    symbol_pngs: Iterable[Path] | None = None,
) -> list[Path]:
    """Returns the list of PNG files to use as templates, deduped by stem.

    If both args are provided, files in `symbol_pngs` shadow same-named files
    in `symbol_dir` (so user-supplied templates override built-ins).
    """
    found: dict[str, Path] = {}
    if symbol_dir is not None:
        for png in sorted(symbol_dir.glob("*.png")):
            found[png.stem] = png
    if symbol_pngs is not None:
        for png in symbol_pngs:
            found[Path(png).stem] = Path(png)
    return [found[k] for k in sorted(found)]


def _prepare_templates(
    symbol_pngs: list[Path],
    scales: Iterable[float],
    rotations: Iterable[int],
    bg: tuple[int, int, int],
) -> dict[str, list[tuple[float, int, np.ndarray]]]:
    """Returns {class_name: [(scale, rotation_deg, gray_template), ...]}"""
    templates: dict[str, list[tuple[float, int, np.ndarray]]] = {}
    for png in symbol_pngs:
        sym = Image.open(png).convert("RGBA")
        variants: list[tuple[float, int, np.ndarray]] = []
        for s in scales:
            w = max(20, int(sym.width * s))
            h = max(20, int(sym.height * s))
            resized = sym.resize((w, h), Image.LANCZOS)
            for rot in rotations:
                rotated = resized.rotate(rot, expand=True, resample=Image.BICUBIC)
                gray = _pil_to_gray(_composite_on_bg(rotated, bg))
                variants.append((s, rot, gray))
        templates[png.stem] = variants
    return templates


# ---------- entry point ----------

def _bbox_centre(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    x0, y0, x1, y1 = bbox
    return ((x0 + x1) // 2, (y0 + y1) // 2)


def _is_in_mask(
    bbox: tuple[int, int, int, int],
    mask_regions: Iterable[tuple[int, int, int, int]],
) -> bool:
    """True if the detection's centre is inside any of the mask rectangles
    (legend, title block, etc.)."""
    cx, cy = _bbox_centre(bbox)
    for mx0, my0, mx1, my1 in mask_regions:
        if mx0 <= cx <= mx1 and my0 <= cy <= my1:
            return True
    return False


def detect_symbols(
    page_image: Image.Image,
    symbol_dir: Path | None = None,
    *,
    symbol_pngs: Iterable[Path] | None = None,
    source_page: int = 1,
    scales: Iterable[float] = DEFAULT_SCALES,
    rotations: Iterable[int] = DEFAULT_ROTATIONS,
    default_threshold: float = DEFAULT_THRESHOLD,
    per_class_threshold: dict[str, float] | None = None,
    nms_iou: float = DEFAULT_NMS_IOU,
    bg_rgb: tuple[int, int, int] = DEFAULT_BG_RGB,
    mask_regions: Iterable[tuple[int, int, int, int]] | None = None,
) -> list[Detection]:
    """Run template matching across all (class, scale, rotation) combinations and
    dedupe with class-wise NMS. Returns a list of `Detection` ordered by class
    then descending score.

    `mask_regions` is an optional iterable of (x0, y0, x1, y1) rectangles in
    page-image px. Any detection whose centre falls inside one of these is
    discarded after NMS. Used to suppress legend / title-block self-detections
    when those region coords are known (e.g. from the synth fixture's GT).
    """
    thresholds = dict(PER_CLASS_THRESHOLD)
    if per_class_threshold:
        thresholds.update(per_class_threshold)

    masks = tuple(mask_regions) if mask_regions else ()

    target_gray = _pil_to_gray(page_image)
    target_h, target_w = target_gray.shape

    pngs = _resolve_symbol_pngs(symbol_dir=symbol_dir, symbol_pngs=symbol_pngs)
    if not pngs:
        raise ValueError(
            "No symbol templates available. Pass symbol_dir or symbol_pngs."
        )
    templates = _prepare_templates(pngs, scales, rotations, bg_rgb)

    raws: list[_Raw] = []
    for cls, variants in templates.items():
        threshold = thresholds.get(cls, default_threshold)
        for scale, rot, tmpl in variants:
            th, tw = tmpl.shape
            if th >= target_h or tw >= target_w:
                continue
            result = cv2.matchTemplate(target_gray, tmpl, cv2.TM_CCOEFF_NORMED)
            ys, xs = np.where(result >= threshold)
            if len(xs) == 0:
                continue
            for y, x in zip(ys.tolist(), xs.tolist()):
                raws.append(
                    _Raw(
                        symbol_class=cls,
                        score=float(result[y, x]),
                        bbox=(int(x), int(y), int(x + tw), int(y + th)),
                        rotation_deg=int(rot),
                        scale=float(scale),
                    )
                )

    kept = _nms_within_class(raws, nms_iou)
    kept = _nms_across_classes(kept, CROSS_CLASS_NMS_IOU)
    if masks:
        kept = [d for d in kept if not _is_in_mask(d.bbox, masks)]
    kept.sort(key=lambda d: (d.symbol_class, -d.score))

    return [
        Detection(
            symbol_class=r.symbol_class,
            bbox=r.bbox,
            score=round(r.score, 4),
            rotation_deg=r.rotation_deg,
            scale=r.scale,
            source_page=source_page,
        )
        for r in kept
    ]
