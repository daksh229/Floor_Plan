"""Phase 1 measurement extractors — 12 fields that answer the client schema.

What each extractor produces:

  Room-derived (Section A):
    - room_count          int
    - bedroom_count       int + multi_region bboxes
    - bathroom_count      int + multi_region bboxes
    - per_room_areas      list of per-room records (Phase 1 = name+bbox+is_wet;
                                                    Phase 2 fills area_m2)

  OCR-label-derived (Section B):
    - window_count            int + multi_region points
    - external_door_count     int + multi_region points
    - glass_doors_windows     int + multi_region points

  Vision-derived (Section C):
    - scale_detected      string + confidence (Claude on a scale-bar crop)

  Phase-2 placeholders (Section D):
    - total_floor_area    float (m²) + polygon
    - wet_area_total      float (m²)
    - building_perimeter  float (m)
    - building_envelope   polygon

The Phase-2 placeholders emit `coordinate_status="placeholder"` with a
note explaining what's missing. Phase 2 (dimension-driven calibration +
polygon construction) promotes them to "actual".
"""
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Optional

from PIL import Image

from .dimension_calibrator import Phase2Result, find_printed_scale_label, run_phase2
from .measurement_schemas import (
    ExtractionField,
    MeasurementOutput,
    ScaleDetected,
    make_bbox_coords,
    make_multi_region_coords,
    make_point_coords,
    make_polygon_coords,
    new_document_id,
    new_extraction_id,
    not_detected_field,
    placeholder_field,
)
from .ocr import extract_text_spans, TesseractNotInstalledError
from .room_extractor import extract_rooms_with_fallback


# Room name patterns. Bedrooms get matched on the canonical-name prefix
# "Bedroom" or "Master". Bathrooms include the AU residential terms
# (Bathroom, Ensuite, WC, Toilet, Powder Room) but NOT Laundry — laundry
# counts as a wet area but not a bathroom for the `bathroom_count` field.
# ============================================================
# Per-room dimensions — Phase 8
# ============================================================
#
# Residential plans label each room with its interior dimensions
# (e.g. "BED 1 / 3200 x 3600"). Tesseract usually reads these as two
# adjacent numeric spans on the same y-line, with the 'x' character
# dropped (small + thin). We detect those pairs here and snap them
# to the nearest room label, producing an EXACT area instead of the
# per-room placeholder Phase 1 emitted.

_NUMERIC_TOKEN = re.compile(r"^\d{3,5}$")
# Width range for a residential room dimension (mm). 200 mm = a WC
# stall; 12000 mm = a generous open-plan room. Anything outside is
# probably a wall dimension or a misread.
_MIN_ROOM_DIM_MM = 1000
_MAX_ROOM_DIM_MM = 15000
# Pair gap threshold (px). Two numeric spans this close on the same
# y-band form a candidate "W x H" pair.
_DIM_PAIR_X_GAP_PX = 80
# Snap-to-room threshold: a dim pair has to be within this distance
# of a room's centroid to be assigned. Tuned for variant_e's render
# resolution (~0.105 px/mm); a 1.5 m radius caps spurious matches.
_DIM_TO_ROOM_MAX_DIST_PX = 250


def _find_room_dimension_pairs(ocr_spans) -> list[dict]:
    """Find pairs of numeric OCR spans on the same line that look like
    "W x H" room dimensions (mm)."""
    numeric: list = []
    for s in ocr_spans:
        t = s.text.strip().replace(",", "")
        if _NUMERIC_TOKEN.match(t):
            try:
                v = int(t)
            except ValueError:
                continue
            if _MIN_ROOM_DIM_MM <= v <= _MAX_ROOM_DIM_MM:
                numeric.append((v, s))

    # Bucket by mid-y to find same-line groups
    bands: dict[int, list] = {}
    for v, s in numeric:
        y_mid = (s.bbox[1] + s.bbox[3]) // 2
        bands.setdefault(y_mid // 8, []).append((v, s))

    pairs: list[dict] = []
    for items in bands.values():
        if len(items) < 2:
            continue
        items.sort(key=lambda x: x[1].bbox[0])
        for i in range(len(items) - 1):
            (w, sa), (h, sb) = items[i], items[i + 1]
            x_gap = sb.bbox[0] - sa.bbox[2]
            if 5 < x_gap < _DIM_PAIR_X_GAP_PX:
                cx = (sa.bbox[0] + sb.bbox[2]) // 2
                cy = (sa.bbox[1] + sb.bbox[3]) // 2
                pairs.append({
                    "w_mm": w, "h_mm": h,
                    "centre_px": (cx, cy),
                    "bbox_px": (sa.bbox[0], min(sa.bbox[1], sb.bbox[1]),
                                 sb.bbox[2], max(sa.bbox[3], sb.bbox[3])),
                })
    return pairs


def _assign_dimensions_to_rooms(
    pairs: list[dict],
    rooms: list,
) -> dict[str, dict]:
    """For each room, find the nearest dimension pair within the
    snap threshold. Greedy — each pair maps to at most one room."""
    used: set[int] = set()
    out: dict[str, dict] = {}
    for r in rooms:
        rcx, rcy = r.centroid
        best_i: Optional[int] = None
        best_d: float = float("inf")
        for i, p in enumerate(pairs):
            if i in used:
                continue
            pcx, pcy = p["centre_px"]
            d = ((rcx - pcx) ** 2 + (rcy - pcy) ** 2) ** 0.5
            if d < best_d and d <= _DIM_TO_ROOM_MAX_DIST_PX:
                best_d = d
                best_i = i
        if best_i is not None:
            used.add(best_i)
            out[r.canonical_name] = pairs[best_i]
    return out


_BEDROOM_NAMES = re.compile(r"^(Bedroom|Master Bedroom|Bed\s*\d*)", re.IGNORECASE)
_BATHROOM_NAMES = re.compile(r"^(Bathroom|Ensuite|WC|Toilet|Powder Room|Bath)$", re.IGNORECASE)
_WET_NAMES = re.compile(r"^(Bathroom|Bath|Ensuite|WC|Toilet|Powder Room|Laundry|Ldy)$", re.IGNORECASE)
# Outdoor / non-indoor spaces. These are returned by the room extractor's
# vocabulary but shouldn't count toward `room_count` because they aren't
# inside the building envelope — they don't contribute floor area or BOM.
_OUTDOOR_NAMES = re.compile(r"^(Courtyard|Alfresco|Porch|Deck|Patio|Light\s*Court|Verandah|Balcony)$", re.IGNORECASE)

# OCR label patterns. Drawings label windows W1, W2, ... ; doors D1, D2, ... ;
# sliding glass doors SD1, SD2, ... (or GD/SGD on some sets).
# OCR commonly confuses '1' with 'l'/'I' and '0' with 'O' — the [\dlIoO]+
# character class catches those misreads while staying strict on the prefix.
_WINDOW_LABEL = re.compile(r"^[Ww][\dlIoO]+$")
_DOOR_LABEL = re.compile(r"^[Dd][\dlIoO]+$")
_GLASS_DOOR_LABEL = re.compile(r"^(SD|SGD|GD|Sd|sd)[\dlIoO]+$")

# Threshold for triggering the Claude-vision fallback on window/door
# labels. A typical residential plan has 4+ windows; if OCR returns
# fewer, the labels are probably below Tesseract's resolution and we
# should let Claude look at the cropped page instead.
_WIN_DOOR_FALLBACK_THRESHOLD = 4


# =========================================================================
# Section A — Room-derived extractors
# =========================================================================

def _build_room_extractions(
    rooms: list,           # list[Room] from room_extractor
    *,
    page_image_size: tuple[int, int],
    source_page: int,
    ocr_spans=None,        # Phase 8 — for per-room dimension matching
) -> list[ExtractionField]:
    """Emit room_count, bedroom_count, bathroom_count, per_room_areas.

    Phase 8 — when `ocr_spans` is provided, the per_room_areas field
    is populated with EXACT m² values derived from the room-dimension
    labels printed on the plan (e.g. "3200 x 3600" next to "BED 1").
    These graduate from `placeholder` to `actual`.
    """
    fields: list[ExtractionField] = []

    # Phase 8: pull room dimensions from OCR
    room_dim_pairs: dict[str, dict] = {}
    if ocr_spans:
        pairs = _find_room_dimension_pairs(ocr_spans)
        room_dim_pairs = _assign_dimensions_to_rooms(pairs, rooms)

    # Filter outdoor spaces (Courtyard, Light Court, Alfresco, ...) out of
    # the indoor room count — they contribute neither floor area nor BOM.
    indoor_rooms = [r for r in rooms if not _OUTDOOR_NAMES.match(r.canonical_name)]
    outdoor_rooms = [r for r in rooms if _OUTDOOR_NAMES.match(r.canonical_name)]
    bedrooms = [r for r in indoor_rooms if _BEDROOM_NAMES.match(r.canonical_name)]
    bathrooms = [r for r in indoor_rooms if _BATHROOM_NAMES.match(r.canonical_name)]
    wets = [r for r in indoor_rooms if _WET_NAMES.match(r.canonical_name)]

    # --- room_count (indoor only) ---
    n_indoor = len(indoor_rooms)
    outdoor_note = (
        f" Plus {len(outdoor_rooms)} outdoor space(s) "
        f"({', '.join(r.canonical_name for r in outdoor_rooms)}) excluded."
        if outdoor_rooms else ""
    )
    fields.append(ExtractionField(
        field_key="room_count",
        field_label="Total Rooms",
        value=n_indoor,
        unit="count",
        source_page=source_page,
        coordinates=None,
        coordinate_status="actual" if indoor_rooms else "not_detected",
        confidence=0.95 if indoor_rooms else 0.0,
        notes=(f"{n_indoor} indoor room label(s) extracted.{outdoor_note}"
               if indoor_rooms else "No indoor room labels found."),
    ))

    # --- bedroom_count ---
    if bedrooms:
        regions = [
            {"label": r.canonical_name, "type": "bounding_box",
             "bbox": list(r.label_bbox)}
            for r in bedrooms
        ]
        fields.append(ExtractionField(
            field_key="bedroom_count",
            field_label="Bedrooms",
            value=len(bedrooms),
            unit="count",
            source_page=source_page,
            coordinates=make_multi_region_coords(regions, page_image_size),
            coordinate_status="actual",
            confidence=0.92,
            notes=f"{len(bedrooms)} bedroom(s) by name match against the AU "
                  "residential vocabulary.",
        ))
    else:
        fields.append(not_detected_field(
            field_key="bedroom_count", field_label="Bedrooms",
            unit="count", source_page=source_page,
            reason="No room with name matching Bedroom/Master/Bed N pattern.",
        ))

    # --- wet_room_count (loose: bathrooms + laundry + WC) ---
    # Separate from bathroom_count because the trade scopes are different:
    # bathroom_count drives bathroom-specific plumbing & tiling line items;
    # wet_room_count is the all-up wet-area count used by the schedule
    # table (which lumps laundry in with bathrooms).
    if wets:
        regions = [
            {"label": r.canonical_name, "type": "bounding_box",
             "bbox": list(r.label_bbox)}
            for r in wets
        ]
        fields.append(ExtractionField(
            field_key="wet_room_count",
            field_label="Wet Rooms (incl. laundry)",
            value=len(wets),
            unit="count",
            source_page=source_page,
            coordinates=make_multi_region_coords(regions, page_image_size),
            coordinate_status="actual",
            confidence=0.92,
            notes=(f"{len(wets)} wet room(s) including laundry. "
                   "Use this for area-schedule alignment; use `bathroom_count` "
                   "for strict bathroom-plumbing scope."),
        ))
    else:
        fields.append(not_detected_field(
            field_key="wet_room_count", field_label="Wet Rooms",
            unit="count", source_page=source_page,
            reason="No wet-area rooms found.",
        ))

    # --- bathroom_count ---
    if bathrooms:
        regions = [
            {"label": r.canonical_name, "type": "bounding_box",
             "bbox": list(r.label_bbox)}
            for r in bathrooms
        ]
        fields.append(ExtractionField(
            field_key="bathroom_count",
            field_label="Bathrooms / Ensuites",
            value=len(bathrooms),
            unit="count",
            source_page=source_page,
            coordinates=make_multi_region_coords(regions, page_image_size),
            coordinate_status="actual",
            confidence=0.92,
            notes=f"{len(bathrooms)} wet room(s) identified (Bath/Ensuite/WC/Toilet/Powder).",
        ))
    else:
        fields.append(not_detected_field(
            field_key="bathroom_count", field_label="Bathrooms / Ensuites",
            unit="count", source_page=source_page,
            reason="No room with name matching Bathroom/Ensuite/WC/Toilet/Powder.",
        ))

    # --- per_room_areas: Phase 8 fills area_m2 from OCR'd room dimensions ---
    if rooms:
        room_records = []
        n_with_area = 0
        for r in rooms:
            dims = room_dim_pairs.get(r.canonical_name)
            area_m2 = None
            w_mm = None
            h_mm = None
            if dims is not None:
                w_mm = dims["w_mm"]
                h_mm = dims["h_mm"]
                area_m2 = round((w_mm * h_mm) / 1_000_000.0, 2)
                n_with_area += 1
            room_records.append({
                "name": r.canonical_name,
                "label_bbox_px": list(r.label_bbox),
                "is_wet": bool(_WET_NAMES.match(r.canonical_name)),
                "is_outdoor": bool(_OUTDOOR_NAMES.match(r.canonical_name)),
                "w_mm": w_mm,
                "h_mm": h_mm,
                "area_m2": area_m2,
            })
        # Status: outdoor rooms (Courtyard, Alfresco) don't have dimension
        # labels — count them as expected misses. Status is "actual" when
        # every INDOOR room has an OCR'd area.
        n_indoor_total = sum(
            1 for r in room_records if not r["is_outdoor"]
        )
        n_indoor_with_area = sum(
            1 for r in room_records
            if not r["is_outdoor"] and r["area_m2"] is not None
        )
        if n_indoor_with_area == n_indoor_total and n_indoor_with_area > 0:
            status = "actual"
            conf = 0.90
            outdoor_note = ""
            if n_indoor_total != len(rooms):
                outdoor_note = (f" {len(rooms) - n_indoor_total} outdoor space(s) "
                                f"correctly excluded from area extraction.")
            notes = (f"All {n_indoor_with_area} indoor room areas derived from "
                     f"on-plan dimension labels (e.g. '3200 x 3600' next to "
                     f"'BED 1').{outdoor_note}")
        elif n_indoor_with_area > 0:
            status = "placeholder"
            conf = 0.60
            notes = (f"{n_indoor_with_area} of {n_indoor_total} indoor rooms "
                     f"have OCR'd dimension labels — others left as area_m2=null. "
                     f"Run higher-DPI OCR or supply via manual edit.")
        else:
            status = "placeholder"
            conf = 0.40
            notes = (f"{len(rooms)} room(s) with label bboxes but no on-plan "
                     "dimension labels were found by OCR. area_m2 null on every row.")
        fields.append(ExtractionField(
            field_key="per_room_areas",
            field_label="Per-Room Breakdown",
            value=room_records,
            unit="list",
            source_page=source_page,
            coordinates=None,
            coordinate_status=status,
            confidence=conf,
            notes=notes,
        ))
    else:
        fields.append(not_detected_field(
            field_key="per_room_areas", field_label="Per-Room Breakdown",
            unit="list", source_page=source_page,
            reason="No rooms extracted.",
        ))

    return fields


# =========================================================================
# Section B — OCR-label-derived extractors
# =========================================================================

def _claude_find_window_door_labels(
    page_image, source_page: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Claude-vision fallback when OCR can't read tiny W*/D*/SD* glyphs.

    Returns (windows, doors, glass_doors) where each entry is a dict
    {"label": "W1", "position": (cx, cy)} matching the shape we'd build
    from an OCRSpan.

    Returns ([], [], []) on any error so the caller can fall back to
    whatever OCR found.
    """
    try:
        import base64, io, json, os, re as _re
        from anthropic import Anthropic, APIError, APITimeoutError
    except ImportError:
        return [], [], []

    # Bump resolution: window/door labels are tiny blue text at typical
    # plan scale (often <10 px on a downsampled image). 2400 max-edge
    # gives Claude readable glyphs at a still-reasonable token cost.
    long_edge = max(page_image.width, page_image.height)
    max_edge = 2400
    if long_edge > max_edge:
        ratio = max_edge / long_edge
        from PIL import Image as _Image
        small = page_image.resize(
            (int(page_image.width * ratio), int(page_image.height * ratio)),
            _Image.LANCZOS,
        )
        scale_back = long_edge / max(small.width, small.height)
    else:
        small, scale_back = page_image, 1.0

    buf = io.BytesIO(); small.convert("RGB").save(buf, format="JPEG", quality=85)
    img_b64 = base64.standard_b64encode(buf.getvalue()).decode("ascii")

    system = (
        "You are a construction-drawing opening finder. Count every "
        "WINDOW, every external DOOR, and every SLIDING GLASS DOOR on "
        "the architectural floor plan you're shown. Identify them by "
        "VISUAL CONVENTION:\n\n"
        "  - Window:           a GAP in an external wall with two parallel "
        "lines (or a single thin rectangle) crossing it. Often labelled "
        "W1, W2, W3, etc. nearby, but count by visual appearance whether "
        "or not the label is legible.\n"
        "  - Door (external):  a GAP in a wall with a quarter-circle arc "
        "(door swing) drawn from one corner of the opening. Often "
        "labelled D1, D2, ... .\n"
        "  - Sliding glass:    a GAP in a wall with TWO PARALLEL HEAVY "
        "LINES across it (heavier than window glass lines), often labelled "
        "SD1, SGD1, GD1.\n\n"
        "For each opening, return its centre position in PIXEL "
        "coordinates of the image you see, top-left origin. If you can "
        "see a label nearby, include it; otherwise use a generic label "
        "like 'W?' or 'D?'.\n\n"
        "Return ONLY this JSON:\n"
        "{\"windows\":     [{\"label\": \"W1\", \"position\": [x, y]}, ...],\n"
        " \"doors\":       [{\"label\": \"D1\", \"position\": [x, y]}, ...],\n"
        " \"glass_doors\": [{\"label\": \"SD1\", \"position\": [x, y]}, ...]}\n\n"
        "If none visible, return empty arrays. Never invent."
    )

    try:
        client = Anthropic(timeout=60)
        resp = client.messages.create(
            model=os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6"),
            max_tokens=2000,
            system=system,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {
                        "type": "base64", "media_type": "image/jpeg", "data": img_b64,
                    }},
                    {"type": "text", "text": (
                        f"Image is {small.width} x {small.height} px. "
                        "Return JSON per system instructions."
                    )},
                ],
            }],
        )
        raw_text = "".join(
            b.text for b in resp.content if getattr(b, "type", None) == "text"
        )
    except (APITimeoutError, APIError, Exception):  # noqa: BLE001
        return [], [], []

    try:
        fenced = _re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, _re.DOTALL)
        if fenced:
            parsed = json.loads(fenced.group(1))
        else:
            first, last = raw_text.find("{"), raw_text.rfind("}")
            parsed = json.loads(raw_text[first : last + 1])
    except (ValueError, json.JSONDecodeError):
        return [], [], []

    def _scale_back(items):
        out = []
        for it in items or []:
            label = str(it.get("label", "")).strip()
            pos = it.get("position")
            if not label or not isinstance(pos, (list, tuple)) or len(pos) != 2:
                continue
            try:
                cx = int(round(float(pos[0]) * scale_back))
                cy = int(round(float(pos[1]) * scale_back))
            except (TypeError, ValueError):
                continue
            out.append({"label": label, "position": [cx, cy]})
        return out

    return (
        _scale_back(parsed.get("windows")),
        _scale_back(parsed.get("doors")),
        _scale_back(parsed.get("glass_doors")),
    )


def _build_window_door_extractions(
    spans,                 # list[OCRSpan]
    *,
    page_image,            # PIL.Image — needed if we fall back to Claude vision
    page_image_size: tuple[int, int],
    source_page: int,
    enable_claude_fallback: bool = True,
) -> list[ExtractionField]:
    """Count W*/D*/SD* labels and emit their centre points.

    OCR-first: walk OCRSpans, match against fuzzy W*/D*/SD* patterns
    (accepting common '1'<->'l'/'I' misreads). When total count falls
    below `_WIN_DOOR_FALLBACK_THRESHOLD`, fire a single Claude-vision
    call against the page crop to recover labels Tesseract couldn't
    read at this resolution.
    """
    windows: list = []
    doors: list = []
    glass_doors: list = []

    # OCR pass
    for s in spans:
        t = s.text.strip()
        # IMPORTANT: check SD pattern BEFORE D pattern, since 'D' is a
        # prefix of 'SD' — without ordering 'SD1' would match _DOOR_LABEL.
        if _GLASS_DOOR_LABEL.match(t):
            glass_doors.append({
                "label": t.upper(),
                "position": [(s.bbox[0] + s.bbox[2]) // 2, (s.bbox[1] + s.bbox[3]) // 2],
            })
        elif _WINDOW_LABEL.match(t):
            windows.append({
                "label": t.upper(),
                "position": [(s.bbox[0] + s.bbox[2]) // 2, (s.bbox[1] + s.bbox[3]) // 2],
            })
        elif _DOOR_LABEL.match(t):
            doors.append({
                "label": t.upper(),
                "position": [(s.bbox[0] + s.bbox[2]) // 2, (s.bbox[1] + s.bbox[3]) // 2],
            })

    # Claude fallback when OCR was sparse
    used_fallback = False
    if enable_claude_fallback and (len(windows) + len(doors) + len(glass_doors)) < _WIN_DOOR_FALLBACK_THRESHOLD:
        c_win, c_door, c_glass = _claude_find_window_door_labels(page_image, source_page)
        if c_win or c_door or c_glass:
            # Replace OCR results entirely with Claude's — they're more reliable
            # at this resolution. Merging the two risks double-counting.
            windows = c_win
            doors = c_door
            glass_doors = c_glass
            used_fallback = True

    fields: list[ExtractionField] = []

    def _entry_to_region(entry: dict) -> dict:
        """Each entry is already {label, position}; wrap for multi_region."""
        return {"label": entry["label"], "type": "point", "position": entry["position"]}

    source_note = " (Claude vision fallback fired — OCR couldn't read labels at this resolution)" if used_fallback else " (OCR)"

    # When all three opening fields end up empty, the not_detected notes
    # should point readers at the schedule backfill in `combined_runner`
    # — that's the architectural fix, not "Tesseract failed".
    schedule_backfill_note = (
        "Page-5 visual detection of W*/D*/SD* labels often fails at "
        "residential render scale (they're tiny blue glyphs). Run via "
        "`run_combined_pipeline()` so the Phase-5 schedule reader can "
        "backfill these counts from the page-6 window/door schedule "
        "(authoritative source)."
    )

    # --- window_count ---
    if windows:
        fields.append(ExtractionField(
            field_key="window_count",
            field_label="Windows",
            value=len(windows),
            unit="count",
            source_page=source_page,
            coordinates=make_multi_region_coords(
                [_entry_to_region(e) for e in windows], page_image_size,
            ),
            coordinate_status="actual",
            confidence=0.88 if not used_fallback else 0.82,
            notes=f"{len(windows)} window label(s) (W1..Wn) detected{source_note}.",
        ))
    else:
        fields.append(not_detected_field(
            field_key="window_count", field_label="Windows",
            unit="count", source_page=source_page,
            reason="No W* labels found in OCR or Claude-vision fallback. "
                   + schedule_backfill_note,
        ))

    # --- external_door_count ---
    if doors:
        fields.append(ExtractionField(
            field_key="external_door_count",
            field_label="External Doors",
            value=len(doors),
            unit="count",
            source_page=source_page,
            coordinates=make_multi_region_coords(
                [_entry_to_region(e) for e in doors], page_image_size,
            ),
            coordinate_status="actual",
            confidence=0.85 if not used_fallback else 0.80,
            notes=f"{len(doors)} door label(s) (D1..Dn) detected{source_note}.",
        ))
    else:
        fields.append(not_detected_field(
            field_key="external_door_count", field_label="External Doors",
            unit="count", source_page=source_page,
            reason="No D* labels found in OCR or Claude-vision fallback. "
                   + schedule_backfill_note,
        ))

    # --- glass_doors_windows = windows + glass_doors (combined per client schema) ---
    glass_total_count = len(windows) + len(glass_doors)
    if glass_total_count:
        fields.append(ExtractionField(
            field_key="glass_doors_windows",
            field_label="Glass Doors & Windows",
            value=glass_total_count,
            unit="count",
            source_page=source_page,
            coordinates=make_multi_region_coords(
                [_entry_to_region(e) for e in windows + glass_doors], page_image_size,
            ),
            coordinate_status="actual",
            confidence=0.87 if not used_fallback else 0.81,
            notes=(f"{glass_total_count} glass opening(s): {len(windows)} window(s) + "
                   f"{len(glass_doors)} sliding glass door(s){source_note}."),
        ))
    else:
        fields.append(not_detected_field(
            field_key="glass_doors_windows", field_label="Glass Doors & Windows",
            unit="count", source_page=source_page,
            reason="No W*/SD* labels found in OCR or Claude-vision fallback. "
                   + schedule_backfill_note,
        ))

    return fields


# =========================================================================
# Section C — Phase 2 placeholders
# =========================================================================

def _build_phase2_fields(
    page_image_size: tuple[int, int],
    source_page: int,
    phase2: Optional[Phase2Result],
    n_bathrooms: int,
    n_wet_rooms: int,
    per_room_areas: Optional[list[dict]] = None,    # Phase 8 — exact areas
) -> list[ExtractionField]:
    """Emit the four Phase-2 fields. When `phase2` is None or its
    calibration failed, fall back to `placeholder` records explaining
    why. When successful, fields become `actual` with the dimension-
    derived numbers."""
    fields: list[ExtractionField] = []

    if phase2 is None or phase2.calibration is None:
        # Phase 2 didn't run or couldn't calibrate — emit placeholders
        reason = ("Dimension-driven calibration did not complete "
                  f"(dims found: {phase2.n_dimensions_found if phase2 else 0}, "
                  f"walls found: {phase2.n_walls_found if phase2 else 0}). "
                  "Plan may not have legible perimeter dimensions.")
        fields.extend([
            placeholder_field(field_key="total_floor_area",
                              field_label="Total Floor Area",
                              unit="m2", source_page=source_page, reason=reason),
            placeholder_field(field_key="wet_area_total",
                              field_label="Wet Area Total",
                              unit="m2", source_page=source_page, reason=reason),
            placeholder_field(field_key="building_perimeter",
                              field_label="Building Perimeter",
                              unit="m", source_page=source_page, reason=reason),
            placeholder_field(field_key="building_envelope",
                              field_label="Building Envelope",
                              unit="polygon", source_page=source_page, reason=reason),
        ])
        return fields

    cal = phase2.calibration
    conf = cal.confidence  # axes-agreement confidence, [0..1]

    # --- total_floor_area_m2 ---
    if phase2.total_floor_area_m2 is not None:
        fields.append(ExtractionField(
            field_key="total_floor_area",
            field_label="Total Floor Area",
            value=phase2.total_floor_area_m2,
            unit="m2",
            source_page=source_page,
            coordinates=(
                make_polygon_coords(phase2.envelope.polygon_px, page_image_size)
                if phase2.envelope else None
            ),
            coordinate_status="actual",
            confidence=conf,
            notes=(f"Derived from dimension chain totals "
                   f"({cal.pairings[0].dim.value_mm} mm x "
                   f"{cal.pairings[1].dim.value_mm} mm = "
                   f"{phase2.total_floor_area_m2} m^2 net after wall-thickness "
                   f"correction). Calibrated at {cal.mm_per_px:.4f} mm/px, "
                   f"axes agree to within {cal.pair_stdev_pct:.1f}%."),
        ))
    else:
        fields.append(placeholder_field(
            field_key="total_floor_area", field_label="Total Floor Area",
            unit="m2", source_page=source_page,
            reason="Calibration succeeded but envelope detection failed.",
        ))

    # --- wet_area_total_m2 ---
    # Phase 8 — sum the wet rooms' actual areas from per_room_areas when
    # available. Falls back to the 10%-of-total heuristic when per-room
    # dims weren't OCR'd.
    wet_rooms_with_area = []
    if per_room_areas:
        wet_rooms_with_area = [
            r for r in per_room_areas
            if r.get("is_wet") and not r.get("is_outdoor")
            and r.get("area_m2") is not None
        ]

    if wet_rooms_with_area:
        wet_total = round(sum(r["area_m2"] for r in wet_rooms_with_area), 2)
        wet_room_names = ", ".join(r["name"] for r in wet_rooms_with_area)
        fields.append(ExtractionField(
            field_key="wet_area_total",
            field_label="Wet Area Total",
            value=wet_total,
            unit="m2",
            source_page=source_page,
            coordinates=None,
            coordinate_status="actual",
            confidence=0.90,
            notes=(f"Sum of {len(wet_rooms_with_area)} wet room(s) "
                   f"({wet_room_names}) using per-room dimensions from "
                   f"the plan's room labels."),
        ))
    elif (phase2.total_floor_area_m2 is not None and n_wet_rooms > 0):
        wet_fraction_estimate = 0.10
        wet_area_est = round(phase2.total_floor_area_m2 * wet_fraction_estimate, 2)
        fields.append(ExtractionField(
            field_key="wet_area_total",
            field_label="Wet Area Total",
            value=wet_area_est,
            unit="m2",
            source_page=source_page,
            coordinates=None,
            coordinate_status="placeholder",
            confidence=0.45,
            notes=(f"Rough estimate: {wet_fraction_estimate:.0%} of total floor "
                   f"area (per-room dimensions not OCR'd). Estimator should "
                   f"sight-check."),
        ))
    else:
        fields.append(placeholder_field(
            field_key="wet_area_total", field_label="Wet Area Total",
            unit="m2", source_page=source_page,
            reason="No wet rooms found AND no per-room dimensions OCR'd.",
        ))

    # --- building_perimeter ---
    if phase2.building_perimeter_m is not None:
        fields.append(ExtractionField(
            field_key="building_perimeter",
            field_label="Building Perimeter",
            value=phase2.building_perimeter_m,
            unit="m",
            source_page=source_page,
            coordinates=None,
            coordinate_status="actual",
            confidence=conf,
            notes=(f"2 * (width + height) = 2 * ("
                   f"{cal.pairings[0].dim.value_mm} + "
                   f"{cal.pairings[1].dim.value_mm}) mm = "
                   f"{phase2.building_perimeter_m} m."),
        ))
    else:
        fields.append(placeholder_field(
            field_key="building_perimeter", field_label="Building Perimeter",
            unit="m", source_page=source_page,
            reason="Perimeter requires both width and height dims.",
        ))

    # --- building_envelope polygon ---
    if phase2.envelope:
        env = phase2.envelope
        n_vertices = len(env.polygon_px)
        # Use the dim totals (from cal.pairings) to express the envelope
        # in mm — more useful than the raw px width/height.
        w_mm = cal.pairings[0].dim.value_mm
        h_mm = cal.pairings[1].dim.value_mm
        # Human-readable summary as the value field so the UI doesn't
        # show "None". The actual polygon vertices stay in coordinates.
        summary_value = (
            f"{w_mm} × {h_mm} mm rectangle ({n_vertices}-vertex polygon, "
            f"{env.width_px:.0f} × {env.height_px:.0f} px on page)"
        )
        fields.append(ExtractionField(
            field_key="building_envelope",
            field_label="Building Envelope",
            value=summary_value,
            unit="polygon",
            source_page=source_page,
            coordinates=make_polygon_coords(
                env.polygon_px, page_image_size,
            ),
            coordinate_status="actual",
            confidence=conf * 0.85,
            notes=(f"Outer-wall rectangle, {w_mm} × {h_mm} mm. Vertices in "
                   f"`coordinates.points`. Envelope detection is the weakest "
                   f"piece of Phase 2 — L-shaped footprints would need a "
                   f"planar-graph walk; this is a bounding rectangle of the "
                   f"longest H wall × longest V wall."),
        ))
    else:
        fields.append(placeholder_field(
            field_key="building_envelope", field_label="Building Envelope",
            unit="polygon", source_page=source_page,
            reason="Envelope detection failed despite calibration.",
        ))

    return fields


# =========================================================================
# Top-level orchestrator
# =========================================================================

def extract_measurements(
    page_image: Image.Image,
    *,
    source_page: int,
    source_file: str,
    pdf_path: Optional[Path] = None,
    enable_claude_scale_reader: bool = True,
    enable_claude_room_fallback: bool = True,
) -> MeasurementOutput:
    """Run Phase 1 extraction on a single architectural-plan page.

    Returns a fully-populated `MeasurementOutput` matching the client
    schema, with `coordinate_status` honestly reflecting which fields
    have actual coordinates vs placeholders.

    `enable_claude_scale_reader` controls whether we call Claude vision
    for the scale label. When off, `scale_detected` is left as a
    not_detected field — the caller can fill it later (e.g. from
    dimension calibration in Phase 2).
    """
    t_start = time.perf_counter()
    errors: list[str] = []
    extractions: list[ExtractionField] = []
    scale_detected: Optional[ScaleDetected] = None

    # --- OCR pass (deterministic) ---
    spans = []
    try:
        spans = extract_text_spans(page_image, source_page=source_page)
    except TesseractNotInstalledError as exc:
        errors.append(f"OCR unavailable: {exc}")

    # --- Room extraction (with Claude vision fallback when sparse) ---
    rooms = []
    try:
        re_result = extract_rooms_with_fallback(
            spans,
            page_image=page_image,
            roi_bbox=None,
            source_page=source_page,
            enable_claude_fallback=enable_claude_room_fallback,
        )
        rooms = re_result.rooms
    except Exception as exc:  # noqa: BLE001
        errors.append(f"Room extraction failed: {type(exc).__name__}: {exc}")

    # --- Build the Section A fields (room-derived) ---
    extractions.extend(_build_room_extractions(
        rooms, page_image_size=page_image.size, source_page=source_page,
        ocr_spans=spans,
    ))

    # --- Build the Section B fields (OCR-label-derived + Claude fallback) ---
    extractions.extend(_build_window_door_extractions(
        spans,
        page_image=page_image,
        page_image_size=page_image.size,
        source_page=source_page,
        enable_claude_fallback=enable_claude_room_fallback,
    ))

    # --- Section C field: scale_detected (Claude vision) ---
    if enable_claude_scale_reader:
        try:
            from .scale_reader import read_scale
            scale_detected = read_scale(page_image)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Scale reader failed: {type(exc).__name__}: {exc}")
            scale_detected = None
    if scale_detected is not None:
        extractions.append(ExtractionField(
            field_key="scale_detected",
            field_label="Drawing Scale",
            value=scale_detected.value,
            unit="ratio",
            source_page=source_page,
            coordinates=None,
            coordinate_status="actual",
            confidence=scale_detected.confidence,
            notes=f"Read from scale bar (Claude vision). Source: {scale_detected.source}.",
        ))
    else:
        extractions.append(not_detected_field(
            field_key="scale_detected", field_label="Drawing Scale",
            unit="ratio", source_page=source_page,
            reason="Scale-bar reader returned nothing. Phase 2 dimension "
                   "calibration will provide the derived ratio.",
        ))

    # --- Section D: Phase 2 — dimension-driven calibration ---
    phase2: Optional[Phase2Result] = None
    try:
        phase2 = run_phase2(page_image, spans)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"Phase 2 calibration failed: {type(exc).__name__}: {exc}")

    # Count wet/bathroom rooms for the wet_area heuristic
    indoor = [r for r in rooms if not _OUTDOOR_NAMES.match(r.canonical_name)]
    n_bathrooms = sum(1 for r in indoor if _BATHROOM_NAMES.match(r.canonical_name))
    n_wet = sum(1 for r in indoor if _WET_NAMES.match(r.canonical_name))

    # Phase 8 — pull the per_room_areas value (now with real m²) so
    # the Phase 2 wet_area_total can sum actual room areas.
    per_room = None
    for f in extractions:
        if f.field_key == "per_room_areas" and isinstance(f.value, list):
            per_room = f.value
            break

    extractions.extend(_build_phase2_fields(
        page_image_size=page_image.size,
        source_page=source_page,
        phase2=phase2,
        n_bathrooms=n_bathrooms,
        n_wet_rooms=n_wet,
        per_room_areas=per_room,
    ))

    # Scale-label resolution priority (most-reliable first):
    #   1. Printed "1:NNN" text on the plan, read by OCR. Most reliable
    #      because it's what the architect wrote.
    #   2. Claude vision scale-bar reader.
    #   3. Dimension-derived ratio. Numerically accurate but the
    #      label-snap formula is rendering-DPI dependent and often wrong.
    #
    # We always run #1 if OCR is available. It overrides everything.
    printed_scale_label = find_printed_scale_label(spans) if spans else None
    if printed_scale_label:
        scale_detected = ScaleDetected(
            value=printed_scale_label,
            source="ocr_label",
            confidence=0.95,
        )
        for i, f in enumerate(extractions):
            if f.field_key == "scale_detected":
                extractions[i] = ExtractionField(
                    field_key="scale_detected",
                    field_label="Drawing Scale",
                    value=printed_scale_label,
                    unit="ratio",
                    source_page=source_page,
                    coordinates=None,
                    coordinate_status="actual",
                    confidence=0.95,
                    notes=(f"Read directly from the printed scale label on "
                           f"the plan via OCR (e.g. 'SCALE 1:100'). "
                           "Most reliable source for the scale ratio."),
                )
                break
    elif scale_detected is None and phase2 and phase2.calibration:
        # Fallback: dimension-derived label (note its limitation in the message)
        scale_detected = ScaleDetected(
            value=phase2.calibration.derived_scale_label,
            source="dimension_chain",
            confidence=phase2.calibration.confidence * 0.7,   # discount
        )
        for i, f in enumerate(extractions):
            if f.field_key == "scale_detected":
                extractions[i] = ExtractionField(
                    field_key="scale_detected",
                    field_label="Drawing Scale",
                    value=scale_detected.value,
                    unit="ratio",
                    source_page=source_page,
                    coordinates=None,
                    coordinate_status="actual",
                    confidence=scale_detected.confidence,
                    notes=(f"Derived from dimension chain (no printed '1:NNN' "
                           f"label found in OCR). Calibrated mm_per_px = "
                           f"{phase2.calibration.mm_per_px:.4f}, axes agree to "
                           f"within {phase2.calibration.pair_stdev_pct:.1f}%. "
                           f"NOTE: the snap-to-standard-scale assumes 150 DPI "
                           f"printing — the underlying ratio is correct but "
                           f"the label string may be off if the PDF was "
                           f"rendered at non-standard DPI."),
                )
                break

    # --- Status: success if no errors and at least one actual field;
    #     partial if some fields ok but errors present; failed if nothing extracted.
    actual_count = sum(1 for f in extractions if f.coordinate_status == "actual")
    if errors and actual_count == 0:
        status = "failed"
    elif errors:
        status = "partial"
    else:
        status = "success"

    elapsed_ms = int((time.perf_counter() - t_start) * 1000)

    return MeasurementOutput(
        document_id=new_document_id(),
        extraction_id=new_extraction_id(),
        status=status,
        source_file=source_file,
        extractions=extractions,
        errors=errors,
        processing_time_ms=elapsed_ms,
        model_version="techuz-procalc-v0.1",
        scale_detected=scale_detected,
    )
