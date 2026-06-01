"""YOLOv8 ONNX backend for electrical-symbol detection.

Drop-in replacement for `cv_detect.detect_symbols()`. Returns the same
`list[Detection]` so the rest of the pipeline (RAG -> BOM assembler) is
backend-agnostic.

The ONNX file is loaded once per process and reused across calls.

Class index ordering MUST match `training_data/master.json` categories
(ids 0..14, alphabetical-by-name). Mismatching this would silently map
labels to the wrong symbol class with high confidence — guard rails are
class-count-check + an explicit hard-coded list below.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

from .schemas import Detection


# Hard-coded to fail loud if the ONNX has a different number of classes.
# Order matches training_data/master.json categories array (id -> name).
YOLO_CLASS_NAMES: tuple[str, ...] = (
    "ceiling_fan",        # 0
    "ceiling_light",      # 1
    "data_point",         # 2
    "dimmer",             # 3
    "distribution_board", # 4
    "double_gpo",         # 5
    "downlight",          # 6
    "exhaust_fan",        # 7
    "single_gpo",         # 8
    "single_pole_switch", # 9
    "smoke_detector",     # 10
    "tv_point",           # 11
    "two_way_switch",     # 12
    "wall_light",         # 13
    "wp_gpo",             # 14
)

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "best.onnx"
DEFAULT_CONF: float = 0.25
DEFAULT_IOU: float = 0.45
INPUT_SIZE: int = 1024  # must match training imgsz


# Module-level cache so re-running detect_symbols() in the Streamlit loop
# doesn't reload the 12 MB graph each time.
_session_cache: dict[str, ort.InferenceSession] = {}


def _load_session(model_path: Path) -> ort.InferenceSession:
    key = str(model_path.resolve())
    sess = _session_cache.get(key)
    if sess is None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"YOLO ONNX model not found at {model_path}. "
                f"Train via training_model.md and copy best.onnx to project root."
            )
        sess = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        # Schema guard: output channels must be 4 + len(YOLO_CLASS_NAMES)
        out_shape = sess.get_outputs()[0].shape
        expected_ch = 4 + len(YOLO_CLASS_NAMES)
        if out_shape[1] != expected_ch:
            raise ValueError(
                f"ONNX output channel count {out_shape[1]} != expected "
                f"{expected_ch} (4 bbox + {len(YOLO_CLASS_NAMES)} classes). "
                f"YOLO_CLASS_NAMES is stale relative to the trained model."
            )
        _session_cache[key] = sess
    return sess


def _letterbox(
    img: np.ndarray, new_size: int = INPUT_SIZE, color: int = 114
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize + pad image to a square of side `new_size`, preserving aspect.

    Returns (letterboxed_img, scale, (pad_x, pad_y)) so callers can map
    detections back to the original image coordinate space.
    """
    h, w = img.shape[:2]
    scale = min(new_size / w, new_size / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_x = (new_size - new_w) // 2
    pad_y = (new_size - new_h) // 2
    canvas = np.full((new_size, new_size, 3), color, dtype=np.uint8)
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return canvas, scale, (pad_x, pad_y)


def _preprocess(pil_img: Image.Image) -> tuple[np.ndarray, float, tuple[int, int]]:
    rgb = np.array(pil_img.convert("RGB"))
    canvas, scale, pad = _letterbox(rgb)
    chw = canvas.transpose(2, 0, 1).astype(np.float32) / 255.0
    return chw[np.newaxis, ...], scale, pad


def _postprocess(
    output: np.ndarray,
    scale: float,
    pad: tuple[int, int],
    orig_w: int,
    orig_h: int,
    conf_thresh: float,
    iou_thresh: float,
    per_class_threshold: Optional[dict[str, float]],
) -> list[tuple[str, float, tuple[int, int, int, int]]]:
    """Decode YOLOv8 raw output -> list of (class_name, score, (x0,y0,x1,y1))."""
    # output: (1, 4+nc, N) -> transpose to (N, 4+nc)
    preds = output[0].T  # (N, 19)
    boxes_xywh = preds[:, :4]
    cls_scores = preds[:, 4:]  # (N, nc), already sigmoid-applied by YOLOv8 export

    # For each anchor, pick the best class
    cls_ids = np.argmax(cls_scores, axis=1)
    confidences = cls_scores[np.arange(cls_scores.shape[0]), cls_ids]

    # Initial confidence filter
    keep = confidences >= conf_thresh
    if not keep.any():
        return []
    boxes_xywh = boxes_xywh[keep]
    confidences = confidences[keep]
    cls_ids = cls_ids[keep]

    # Per-class threshold override (apply after global filter)
    if per_class_threshold:
        survives = np.ones(len(confidences), dtype=bool)
        for i, (cid, conf) in enumerate(zip(cls_ids, confidences)):
            name = YOLO_CLASS_NAMES[cid]
            thr = per_class_threshold.get(name)
            if thr is not None and conf < thr:
                survives[i] = False
        if not survives.any():
            return []
        boxes_xywh = boxes_xywh[survives]
        confidences = confidences[survives]
        cls_ids = cls_ids[survives]

    # xywh (centre, width, height) -> xyxy on the 1024x1024 padded canvas
    cx, cy, w, h = boxes_xywh.T
    x0 = cx - w / 2
    y0 = cy - h / 2
    x1 = cx + w / 2
    y1 = cy + h / 2
    boxes_xyxy_padded = np.stack([x0, y0, x1, y1], axis=1)

    # Per-class NMS (class-aware so an overlap between two classes doesn't get suppressed).
    # cv2.dnn.NMSBoxesBatched is class-aware but needs xywh; easier to do per-class loop.
    final: list[tuple[str, float, tuple[int, int, int, int]]] = []
    pad_x, pad_y = pad
    for cid in np.unique(cls_ids):
        mask = cls_ids == cid
        cls_boxes = boxes_xyxy_padded[mask]
        cls_scores_ = confidences[mask]
        # cv2.dnn.NMSBoxes expects (x, y, w, h) xywh
        nms_in = [
            [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])]
            for b in cls_boxes
        ]
        idxs = cv2.dnn.NMSBoxes(
            nms_in, cls_scores_.astype(float).tolist(), conf_thresh, iou_thresh
        )
        if len(idxs) == 0:
            continue
        idxs = np.array(idxs).flatten()
        for i in idxs:
            x0p, y0p, x1p, y1p = cls_boxes[i]
            # Undo letterbox: subtract pad then divide by scale
            ox0 = int(round((x0p - pad_x) / scale))
            oy0 = int(round((y0p - pad_y) / scale))
            ox1 = int(round((x1p - pad_x) / scale))
            oy1 = int(round((y1p - pad_y) / scale))
            # Clamp to image bounds
            ox0 = max(0, min(orig_w - 1, ox0))
            oy0 = max(0, min(orig_h - 1, oy0))
            ox1 = max(0, min(orig_w - 1, ox1))
            oy1 = max(0, min(orig_h - 1, oy1))
            if ox1 <= ox0 or oy1 <= oy0:
                continue
            final.append(
                (YOLO_CLASS_NAMES[int(cid)], float(cls_scores_[i]), (ox0, oy0, ox1, oy1))
            )
    return final


def detect_symbols(
    page_image: Image.Image,
    symbol_dir: Path | None = None,             # accepted for API compat; ignored
    *,
    symbol_pngs: Iterable[Path] | None = None,  # accepted for API compat; ignored
    source_page: int = 1,
    default_threshold: float = DEFAULT_CONF,
    per_class_threshold: dict[str, float] | None = None,
    nms_iou: float = DEFAULT_IOU,
    mask_regions: Iterable[tuple[int, int, int, int]] | None = None,
    model_path: Path | None = None,
    # The following are accepted for cv_detect API compatibility but ignored:
    scales: Iterable[float] | None = None,
    rotations: Iterable[int] | None = None,
    bg_rgb: tuple[int, int, int] | None = None,
) -> list[Detection]:
    """Run YOLOv8 inference and return `Detection` objects.

    Parameters are kept compatible with `cv_detect.detect_symbols()` so the
    runner can hot-swap backends. Template-matching-specific args (scales,
    rotations, bg_rgb, symbol_dir/pngs) are accepted and ignored — YOLO
    learned scale + rotation invariance from training augmentation.

    `mask_regions` is honoured: detections whose centre falls inside a
    masked rectangle are dropped (used to suppress legend / title-block
    self-detections when GT for those regions is known).
    """
    model_path = model_path or DEFAULT_MODEL_PATH
    sess = _load_session(model_path)

    blob, scale, pad = _preprocess(page_image)
    output = sess.run(None, {sess.get_inputs()[0].name: blob})[0]

    decoded = _postprocess(
        output,
        scale=scale,
        pad=pad,
        orig_w=page_image.width,
        orig_h=page_image.height,
        conf_thresh=default_threshold,
        iou_thresh=nms_iou,
        per_class_threshold=per_class_threshold,
    )

    masks = tuple(mask_regions) if mask_regions else ()

    detections: list[Detection] = []
    for cls_name, score, bbox in decoded:
        if masks:
            cx = (bbox[0] + bbox[2]) // 2
            cy = (bbox[1] + bbox[3]) // 2
            if any(
                mx0 <= cx <= mx1 and my0 <= cy <= my1
                for mx0, my0, mx1, my1 in masks
            ):
                continue
        detections.append(
            Detection(
                symbol_class=cls_name,
                bbox=bbox,
                score=round(min(1.0, max(-1.0, score)), 4),
                rotation_deg=0,    # YOLO is rotation-invariant; not exposed
                scale=1.0,         # YOLO is scale-invariant; not exposed
                source_page=source_page,
            )
        )
    detections.sort(key=lambda d: (d.symbol_class, -d.score))
    return detections
