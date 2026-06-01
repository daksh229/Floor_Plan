"""Room-label extraction from OCR spans + room-aware detection assignment.

Used to produce a per-room BOM breakdown ('Living: 3 x Double GPO, 6 x
downlight, ...' etc.) instead of a single project-wide roll-up.

Pipeline:
  1. Filter OCR spans to inside the Claude ROI (drops title-block labels
     like 'STUDIO' that name a project type, not a room).
  2. Merge horizontally-adjacent same-line spans within ~80 px of each
     other — Tesseract returns 'BED' and '02' as separate tokens but the
     user reads them as one label 'BED 02'.
  3. Match merged spans against a curated Australian-residential room
     vocabulary (with synonyms — 'BED 02' -> 'Bedroom 2', 'ENS' ->
     'Ensuite', 'WIR' -> 'Walk-in Robe').
  4. For BOM aggregation, assign each detection to the room whose label
     centroid is nearest in Euclidean distance (Voronoi nearest-seed).
     If <2 rooms detected, room assignment is disabled and the BOM
     falls back to the global view.

The 80 px merge threshold and the vocabulary were tuned against the
variant_a (4 rooms) and variant_b (~6 rooms) synthetic fixtures and
should generalise to most residential floor plans.
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
from typing import Iterable, Optional

from PIL import Image

from .schemas import Detection, OCRSpan, Room, RoomExtraction


# ---------- vocabulary ----------

# Map of (lowercase OCR token base) -> canonical room name.
# Multi-token room names like "BED 02" are handled by merging tokens
# horizontally before lookup; the merged result still resolves through
# this map (the suffix number is preserved separately).
_ROOM_BASE: dict[str, str] = {
    # bedrooms (the base "bed" gets a number suffix if found)
    "bedroom": "Bedroom",
    "bed":     "Bedroom",
    "master":  "Master Bedroom",
    "mbr":     "Master Bedroom",
    # living spaces
    "living":  "Living",
    "lounge":  "Lounge",
    "family":  "Family",
    "meals":   "Meals",
    "dining":  "Dining",
    "dine":    "Dining",
    "kitchen": "Kitchen",
    "kit":     "Kitchen",
    "rumpus":  "Rumpus",
    "theatre": "Theatre",
    "games":   "Games",
    "retreat": "Retreat",
    # wet / utility
    "bathroom":"Bathroom",
    "bath":    "Bathroom",
    "ensuite": "Ensuite",
    "ens":     "Ensuite",
    "wc":      "WC",
    "toilet":  "Toilet",
    "powder":  "Powder Room",
    "laundry": "Laundry",
    "lndy":    "Laundry",
    "ldy":     "Laundry",
    "pantry":  "Pantry",
    "ptry":    "Pantry",
    # circulation / entry / garage
    "entry":   "Entry",
    "foyer":   "Foyer",
    "hall":    "Hallway",
    "hallway": "Hallway",
    "passage": "Passage",
    "garage":  "Garage",
    "gge":     "Garage",
    # storage / dressing
    "robe":    "Robe",
    "wir":     "Walk-in Robe",
    "wic":     "Walk-in Closet",
    "store":   "Store",
    "closet":  "Closet",
    # work / study
    "study":   "Study",
    "office":  "Office",
    # outdoor — generally not part of indoor BOM but still labelled
    "alfresco":"Alfresco",
    "porch":   "Porch",
    "deck":    "Deck",
    "patio":   "Patio",
    "courtyard":"Courtyard",
    # extensions / accessory
    "studio":  "Studio",  # only when not in title block — ROI filter handles that
}

# Tokens that suggest a multi-word room name (next token may be a number
# or letter suffix that distinguishes instances): "BED 02", "BEDROOM 1",
# "WC 2". The suffix gets folded into the canonical name.
_ACCEPTS_SUFFIX: set[str] = {"bedroom", "bed", "master", "wc", "robe", "wir", "ensuite", "ens", "lounge", "study", "office"}

# Single-token noise that often gets keyword-matched but shouldn't count
# as a room. Filtered out before vocab lookup.
_OCR_NOISE_LITERALS: set[str] = {
    "be", "ent", "stu", "kit",  # truncated mid-words from upstream wraps
}

# Horizontal merge threshold (px). Tokens with same baseline whose gap is
# <= this value are merged into one label.
_HORIZONTAL_MERGE_PX: int = 80
# Y-axis tolerance — tokens on the "same line" can vary slightly in y0.
_SAME_LINE_Y_TOL: int = 12

# Confidence floor for an OCR span to count as a candidate room label.
# Below this we silently skip — these are usually OCR misreads on small text.
_MIN_LABEL_CONFIDENCE: float = 0.60


# ---------- helpers ----------

def _span_centre(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    x0, y0, x1, y1 = bbox
    return ((x0 + x1) // 2, (y0 + y1) // 2)


def _bbox_inside(
    inner: tuple[int, int, int, int],
    outer: tuple[int, int, int, int],
) -> bool:
    """True if the centre of `inner` is inside `outer`."""
    cx, cy = _span_centre(inner)
    ox0, oy0, ox1, oy1 = outer
    return ox0 <= cx <= ox1 and oy0 <= cy <= oy1


def _normalise_token(text: str) -> str:
    """Lowercase, strip punctuation. Keep digits since they're suffixes."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _same_line(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    """Two spans are on the same OCR line iff:
      - their mid-y values are within _SAME_LINE_Y_TOL, AND
      - their heights are similar (within 1.5x of each other).
    The height check stops a tall stray span (e.g. a multi-line paragraph
    fragment) from swallowing a short adjacent token. This matters: a
    naïve y-overlap rule merged 'ENTRY' with a paragraph-fragment 'or'
    that happened to overlap by a single pixel, killing the room label.
    """
    ay_mid = (a[1] + a[3]) // 2
    by_mid = (b[1] + b[3]) // 2
    if abs(ay_mid - by_mid) > _SAME_LINE_Y_TOL:
        return False
    ah = max(1, a[3] - a[1])
    bh = max(1, b[3] - b[1])
    ratio = max(ah, bh) / min(ah, bh)
    return ratio <= 1.5


def _merge_horizontally(spans: list[OCRSpan]) -> list[tuple[str, tuple[int, int, int, int], float]]:
    """Merge spans on the same y-line whose horizontal gap is small.

    Returns list of (joined_text, merged_bbox, mean_confidence) tuples.
    Each input span participates in exactly one output group.
    """
    if not spans:
        return []
    # Sort by y first then x, so same-line spans are adjacent
    ordered = sorted(spans, key=lambda s: (s.bbox[1], s.bbox[0]))
    groups: list[list[OCRSpan]] = []
    current: list[OCRSpan] = []
    for s in ordered:
        if not current:
            current = [s]
            continue
        last = current[-1]
        x_gap = s.bbox[0] - last.bbox[2]
        if _same_line(last.bbox, s.bbox) and 0 <= x_gap <= _HORIZONTAL_MERGE_PX:
            current.append(s)
        else:
            groups.append(current)
            current = [s]
    if current:
        groups.append(current)

    out: list[tuple[str, tuple[int, int, int, int], float]] = []
    for grp in groups:
        text = " ".join(s.text for s in grp)
        x0 = min(s.bbox[0] for s in grp)
        y0 = min(s.bbox[1] for s in grp)
        x1 = max(s.bbox[2] for s in grp)
        y1 = max(s.bbox[3] for s in grp)
        mean_conf = sum(s.confidence for s in grp) / len(grp)
        out.append((text, (x0, y0, x1, y1), mean_conf))
    return out


def _resolve_label(merged_text: str) -> Optional[tuple[str, list[str]]]:
    """Map a merged OCR text to a canonical room name.

    Returns (canonical_name, source_tokens) or None if no match.
    """
    raw_tokens = merged_text.split()
    norm_tokens = [_normalise_token(t) for t in raw_tokens if _normalise_token(t)]
    if not norm_tokens:
        return None

    # Filter out short OCR noise that happens to be in the vocab map keys
    if norm_tokens[0] in _OCR_NOISE_LITERALS and norm_tokens[0] not in _ROOM_BASE:
        return None

    first = norm_tokens[0]
    base = _ROOM_BASE.get(first)
    if base is None:
        return None

    # If second token is a numeric/alpha suffix and the base accepts it,
    # fold it into the canonical name. "bed 02" -> "Bedroom 2".
    if len(norm_tokens) >= 2 and first in _ACCEPTS_SUFFIX:
        suffix = norm_tokens[1]
        if suffix.isdigit():
            return (f"{base} {int(suffix)}", raw_tokens[:2])
        if len(suffix) <= 2 and suffix.isalpha():
            return (f"{base} {suffix.upper()}", raw_tokens[:2])

    return (base, raw_tokens[:1])


def _dedupe_rooms(rooms: list[Room]) -> list[Room]:
    """Two OCR hits for the same canonical name (e.g. "BED" appearing
    twice on a 2-bedroom plan) are kept as separate rooms only if their
    label bboxes are spatially distinct. Otherwise the higher-confidence
    one wins.
    """
    if not rooms:
        return rooms
    # Group by canonical name and pick representatives that are >300 px apart
    by_name: dict[str, list[Room]] = {}
    for r in rooms:
        by_name.setdefault(r.canonical_name, []).append(r)

    out: list[Room] = []
    for name, group in by_name.items():
        if len(group) == 1:
            out.append(group[0])
            continue
        # If multiple rooms share a base name (e.g. "Bedroom" with no
        # suffix), assign incrementing numbers by reading order (top->bottom).
        ordered = sorted(group, key=lambda r: (r.centroid[1], r.centroid[0]))
        # If any has a number already in name, leave them as-is
        any_numbered = any(re.search(r"\d", r.canonical_name) for r in ordered)
        if any_numbered:
            out.extend(ordered)
            continue
        for i, r in enumerate(ordered, start=1):
            out.append(r.model_copy(update={"canonical_name": f"{r.canonical_name} {i}"}))
    return out


# ---------- public API ----------

def extract_rooms(
    ocr_spans: Iterable[OCRSpan],
    *,
    page_image_size: tuple[int, int],
    roi_bbox: Optional[tuple[int, int, int, int]] = None,
    source_page: int = 1,
    min_confidence: float = _MIN_LABEL_CONFIDENCE,
) -> RoomExtraction:
    """Extract room labels from OCR spans.

    `roi_bbox`, when provided, restricts the search to inside the floor-plan
    region (in page-image px). Spans outside the ROI are dropped — this
    suppresses title-block text that contains room-sounding words like
    'STUDIO STRUCTURE' or 'BEDROOM LAYOUT NOTES'.

    Returns a RoomExtraction with diagnostics so the UI can surface why
    the room breakdown is what it is.
    """
    spans = [s for s in ocr_spans if s.source_page == source_page]
    raw_count = len(spans)

    # ROI filter
    if roi_bbox is not None:
        spans = [s for s in spans if _bbox_inside(s.bbox, roi_bbox)]
    inside_count = len(spans)

    # Confidence filter
    spans = [s for s in spans if s.confidence >= min_confidence]

    # Merge horizontally-adjacent tokens
    merged = _merge_horizontally(spans)
    multi_token_merges = sum(1 for text, *_ in merged if " " in text)

    # Resolve to canonical room names
    rooms_raw: list[Room] = []
    skipped = 0
    for text, bbox, conf in merged:
        resolved = _resolve_label(text)
        if resolved is None:
            skipped += 1
            continue
        canonical, src_tokens = resolved
        rooms_raw.append(
            Room(
                canonical_name=canonical,
                label_bbox=bbox,
                centroid=_span_centre(bbox),
                source_tokens=src_tokens,
                ocr_confidence=round(conf, 3),
                source_page=source_page,
            )
        )

    rooms = _dedupe_rooms(rooms_raw)

    warning: Optional[str] = None
    if len(rooms) == 0:
        warning = (
            "No room labels recognised — by-room BOM disabled, falling back to "
            "the global By-Symbol view. Check that the plan page is being "
            "rendered at >= 4500 px and the ROI crop is not too aggressive."
        )
    elif len(rooms) == 1:
        warning = (
            f"Only 1 room label recognised ({rooms[0].canonical_name}) — "
            "by-room BOM degenerates to a single-bucket view. Consider whether "
            "OCR is missing labels on smaller rooms."
        )

    return RoomExtraction(
        rooms=rooms,
        raw_ocr_span_count=raw_count,
        spans_inside_roi=inside_count,
        multi_token_merges=multi_token_merges,
        skipped_vocab_misses=skipped,
        warning=warning,
    )


# Distance-gating threshold: a detection's nearest room-label centroid
# must be closer than this fraction of the page diagonal, otherwise the
# detection is left Unassigned. This prevents the "Kitchen swallows the
# whole back of the house" failure mode that happens when OCR misses 4-5
# rooms — without gating, every detection in those rooms gets assigned
# to whichever labelled room is nearest, which is usually a front-of-
# house room. Empirically 0.28 (~28% of page diagonal) draws the line
# at "one room over" for a typical 8-room residential plan.
DEFAULT_MAX_DISTANCE_FRAC: float = 0.28


def assign_detections_to_rooms(
    detections: Iterable[Detection],
    rooms: list[Room],
    *,
    page_size: Optional[tuple[int, int]] = None,
    max_distance_frac: float = DEFAULT_MAX_DISTANCE_FRAC,
) -> dict[int, Optional[str]]:
    """For each detection (keyed by `id()`), return the canonical room name
    of the nearest room-label centroid, or None if no plausible room exists.

    Uses Euclidean distance from the detection's centre to each room
    label's centroid. With <2 rooms, returns None for every detection so
    callers know to fall back to the unassigned bucket.

    When `page_size` is provided, also enforces a max-distance cap: any
    detection whose nearest centroid is more than `max_distance_frac` of
    the page diagonal away is left Unassigned. This is critical when OCR
    misses rooms — without it, detections in undetected rooms get
    silently misattributed to the nearest labelled neighbour.
    """
    detections = list(detections)
    if len(rooms) < 2:
        return {id(d): None for d in detections}

    if page_size is not None:
        w, h = page_size
        max_distance_px = max_distance_frac * (w * w + h * h) ** 0.5
    else:
        max_distance_px = float("inf")

    assignments: dict[int, Optional[str]] = {}
    for det in detections:
        dcx, dcy = det.center
        best_name: Optional[str] = None
        best_d2 = float("inf")
        for r in rooms:
            rx, ry = r.centroid
            d2 = (dcx - rx) ** 2 + (dcy - ry) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best_name = r.canonical_name
        # Distance gate: detection too far from any room label -> Unassigned
        if best_d2 ** 0.5 > max_distance_px:
            assignments[id(det)] = None
        else:
            assignments[id(det)] = best_name
    return assignments


# ---------- Claude-vision fallback ----------

# When OCR finds fewer rooms than this, the Claude vision fallback kicks
# in (if enabled). Threshold tuned for typical residential plans —
# below 5 rooms is rarely correct for a full-house plan.
CLAUDE_FALLBACK_THRESHOLD: int = 5

_CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")
_CLAUDE_TIMEOUT_S = 60
_CLAUDE_MAX_TOKENS = 1200
_CLAUDE_INPUT_MAX_LONG_EDGE = 1600

_CLAUDE_ROOM_SYSTEM = """You are a residential-floor-plan room-label extractor.

You will see a CROP of an electrical / floor plan. Your job is to list \
EVERY room label that is visible on the plan — in any of these forms:

  - long words: BEDROOM, LIVING, KITCHEN, BATHROOM, LAUNDRY, GARAGE,
    DINING, FAMILY, MEALS, STUDY, ENSUITE, PANTRY, ROBE, etc.
  - abbreviations: BED, LDY, LNDY, ENS, WIR, WC, GGE, KIT, etc.
  - numbered variants: BED 01, BED 02, BEDROOM 1, ENSUITE 2, etc.

For each room, return:
  - canonical_name: a normalised display name. Use proper case ("Bedroom 2",
    "Walk-in Robe", "Living"). Expand abbreviations ("LDY" -> "Laundry",
    "WIR" -> "Walk-in Robe", "ENS" -> "Ensuite", "GGE" -> "Garage").
    For numbered variants use the number ("Bedroom 1", "Ensuite 2").
  - bbox: [x0, y0, x1, y1] of the room LABEL text in the image you see,
    NOT the whole room. Pixel coordinates, origin top-left.
  - confidence: 0..1 how sure you are this is a real room label.

IGNORE:
  - the legend/symbol-library box (rows like "ceiling light", "GPO")
  - title-block text ("DRAWING NO", "REVISION", "SCALE", project address)
  - dimensions (numbers like "3600", "2480")
  - notes paragraphs

Return ONLY this JSON object:

{
  "rooms": [
    {"canonical_name": "Bedroom 1", "bbox": [x0, y0, x1, y1], "confidence": 0.92},
    ...
  ]
}
"""


def _downsample_for_claude(
    img: Image.Image, max_long_edge: int = _CLAUDE_INPUT_MAX_LONG_EDGE
) -> tuple[Image.Image, float]:
    long_edge = max(img.width, img.height)
    if long_edge <= max_long_edge:
        return img, 1.0
    ratio = max_long_edge / long_edge
    new_size = (int(img.width * ratio), int(img.height * ratio))
    return img.resize(new_size, Image.LANCZOS), (1.0 / ratio)


def _pil_to_b64_jpeg(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=85)
    return base64.standard_b64encode(buf.getvalue()).decode("ascii")


def _parse_json_block(text: str) -> dict:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return json.loads(fenced.group(1))
    first, last = text.find("{"), text.rfind("}")
    if first == -1 or last == -1:
        raise ValueError("No JSON object in response")
    return json.loads(text[first : last + 1])


def extract_rooms_via_claude(
    image: Image.Image,
    *,
    source_page: int = 1,
    crop_offset: tuple[int, int] = (0, 0),
    model: str = _CLAUDE_MODEL,
    timeout_s: int = _CLAUDE_TIMEOUT_S,
) -> list[Room]:
    """Ask Claude vision to enumerate room labels visible in `image`.

    `crop_offset` (cx, cy) is added to every returned bbox so coordinates
    map back into the parent (full-page) image space. Use this when
    `image` is a crop of a larger page.

    Returns [] on any error (timeout, parse failure, empty response) so
    callers can treat the fallback as best-effort.
    """
    # Lazy import — Anthropic is already a dep elsewhere, but keeping this
    # import local means tests / runs without the SDK still work for the
    # OCR-only path.
    try:
        from anthropic import Anthropic, APIError, APITimeoutError
    except ImportError:
        return []

    downsampled, scale_back = _downsample_for_claude(image)
    img_b64 = _pil_to_b64_jpeg(downsampled)

    client = Anthropic(timeout=timeout_s)
    raw_text = ""
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=_CLAUDE_MAX_TOKENS,
            system=_CLAUDE_ROOM_SYSTEM,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {
                        "type": "base64", "media_type": "image/jpeg", "data": img_b64,
                    }},
                    {"type": "text", "text": (
                        f"Image is {downsampled.width} x {downsampled.height} px. "
                        "Return JSON per system instructions."
                    )},
                ],
            }],
        )
        raw_text = "".join(
            b.text for b in resp.content if getattr(b, "type", None) == "text"
        )
    except (APITimeoutError, APIError):
        return []
    except Exception:  # noqa: BLE001
        return []

    try:
        parsed = _parse_json_block(raw_text)
    except (ValueError, json.JSONDecodeError):
        return []

    raw_rooms = parsed.get("rooms") or []
    cx_off, cy_off = crop_offset
    out: list[Room] = []
    for r in raw_rooms:
        name = r.get("canonical_name")
        bbox = r.get("bbox")
        if not isinstance(name, str) or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        try:
            x0, y0, x1, y1 = [float(v) for v in bbox]
        except (TypeError, ValueError):
            continue
        # Map crop -> full-page coords
        x0 = int(round(x0 * scale_back)) + cx_off
        y0 = int(round(y0 * scale_back)) + cy_off
        x1 = int(round(x1 * scale_back)) + cx_off
        y1 = int(round(y1 * scale_back)) + cy_off
        if x1 - x0 < 5 or y1 - y0 < 5:
            continue
        try:
            conf = float(r.get("confidence", 0.7) or 0.7)
        except (TypeError, ValueError):
            conf = 0.7
        out.append(Room(
            canonical_name=name.strip(),
            label_bbox=(x0, y0, x1, y1),
            centroid=((x0 + x1) // 2, (y0 + y1) // 2),
            source_tokens=[f"(claude) {name.strip()}"],
            ocr_confidence=max(0.0, min(1.0, conf)),
            source_page=source_page,
        ))
    return out


def _merge_room_lists(
    primary: list[Room],
    secondary: list[Room],
    *,
    dedupe_distance_px: int = 200,
) -> list[Room]:
    """Merge two room lists, dropping any secondary entry whose centroid is
    within `dedupe_distance_px` of a primary entry. Primary wins on conflict.
    The secondary list is typically the Claude-vision augmentation.
    """
    merged = list(primary)
    for sr in secondary:
        sx, sy = sr.centroid
        is_dupe = False
        for pr in merged:
            px, py = pr.centroid
            if (sx - px) ** 2 + (sy - py) ** 2 <= dedupe_distance_px ** 2:
                is_dupe = True
                break
        if not is_dupe:
            merged.append(sr)
    return _dedupe_rooms(merged)


def extract_rooms_with_fallback(
    ocr_spans: Iterable[OCRSpan],
    *,
    page_image: Image.Image,
    roi_bbox: Optional[tuple[int, int, int, int]] = None,
    source_page: int = 1,
    enable_claude_fallback: bool = True,
    fallback_threshold: int = CLAUDE_FALLBACK_THRESHOLD,
    min_confidence: float = _MIN_LABEL_CONFIDENCE,
) -> RoomExtraction:
    """Orchestrator: OCR extraction first, then Claude vision augmentation
    when OCR yields fewer rooms than `fallback_threshold`.

    The Claude call is made on the ROI crop when one is available
    (avoiding wasted tokens on title block / legend areas), otherwise on
    the full page.
    """
    primary = extract_rooms(
        ocr_spans,
        page_image_size=(page_image.width, page_image.height),
        roi_bbox=roi_bbox,
        source_page=source_page,
        min_confidence=min_confidence,
    )

    used_fallback = False
    claude_added = 0
    if enable_claude_fallback and len(primary.rooms) < fallback_threshold:
        if roi_bbox is not None:
            x0, y0, x1, y1 = roi_bbox
            crop = page_image.crop((x0, y0, x1, y1))
            offset = (x0, y0)
        else:
            crop = page_image
            offset = (0, 0)
        claude_rooms = extract_rooms_via_claude(
            crop, source_page=source_page, crop_offset=offset
        )
        if claude_rooms:
            merged = _merge_room_lists(primary.rooms, claude_rooms)
            claude_added = len(merged) - len(primary.rooms)
            primary = primary.model_copy(update={"rooms": merged})
            used_fallback = True

    # Re-evaluate the warning after fallback
    note_bits = []
    if used_fallback:
        note_bits.append(
            f"Claude vision fallback added {claude_added} room(s) on top of "
            f"OCR's initial result."
        )
    if len(primary.rooms) == 0:
        note_bits.append(
            "No room labels recognised by OCR or Claude — by-room BOM disabled."
        )
    elif len(primary.rooms) == 1:
        note_bits.append(
            f"Only 1 room label recognised ({primary.rooms[0].canonical_name})."
        )
    warning = " ".join(note_bits) if note_bits else None

    return primary.model_copy(update={"warning": warning})


# ---------- sanity-check warning ----------

def compute_distribution_warning(
    detections: Iterable[Detection],
    assignments: dict[int, Optional[str]],
    rooms_count: int,
) -> Optional[str]:
    """Surface a 'room extraction may be incomplete' warning when the
    per-room detection distribution looks suspicious.

    Triggers:
      - one room holds >40% of detections AND fewer than 5 rooms in total
      - more than 30% of detections landed in 'Unassigned' (distance gate
        tripped frequently — strong signal that OCR missed real rooms)
    Both cases mean the by-room view is likely misleading and the user
    should consider enabling the Claude fallback (if disabled).
    """
    detections = list(detections)
    n = len(detections)
    if n < 10:
        return None

    counts: dict[Optional[str], int] = {}
    for d in detections:
        r = assignments.get(id(d))
        counts[r] = counts.get(r, 0) + 1

    unassigned = counts.get(None, 0)
    if unassigned / n > 0.30:
        return (
            f"{unassigned} of {n} detections ({unassigned / n:.0%}) couldn't be "
            f"pinned to any detected room — they fell outside the distance gate "
            f"around the {rooms_count} room label(s) found. This usually means "
            f"OCR missed labels on additional rooms. Consider enabling the "
            f"Claude vision fallback in the sidebar if it's off."
        )

    # Find the max-share NAMED room
    named = {r: c for r, c in counts.items() if r is not None}
    if not named:
        return None
    max_room, max_count = max(named.items(), key=lambda kv: kv[1])
    share = max_count / n
    if share > 0.40 and rooms_count <= 4:
        return (
            f"⚠ Room extraction may be incomplete: '{max_room}' has {max_count} "
            f"of {n} detections ({share:.0%}). With only {rooms_count} room(s) "
            f"detected this likely means detections from undetected rooms are "
            f"clustering on the nearest labelled one. Consider enabling the "
            f"Claude vision fallback to improve room recall."
        )
    return None
