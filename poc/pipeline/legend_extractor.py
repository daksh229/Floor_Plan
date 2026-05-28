"""Read a legend page from a construction drawing and turn each row into a
LegendRow ready for user-confirmed ingestion into the symbol library.

Strategy (tuned against samples/synthetic_procalc_style_electrical_set.pdf):
  1. Render the requested page at a high scale (default 2.5x) so the small
     table text is large enough for Tesseract.
  2. Run Tesseract with PSM 6 (uniform block of text) over the full page.
  3. Locate the legend's column headers by keyword anchor — 'Symbol',
     'Canonical', 'Alias', 'Unit', 'Indicative' — and use their x-positions
     to define column ranges.
  4. Cluster text spans below the header row by y to get body rows.
  5. For each row: join text in each column's x-range; crop the symbol
     image from the Symbol column's x-range at that row's y.
  6. Parse unit + indicative cost; emit a LegendRow plus a list of warnings.

The output is *editable* — the Streamlit UI / CLI surface lets the user
correct any OCR errors before writing to the library.
"""
from __future__ import annotations

import base64
import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pypdfium2 as pdfium
import pytesseract
from PIL import Image

from .ocr import _ensure_configured
from .rag import get_builtin_entry
from .schemas import LegendExtraction, LegendRow


# Map common legend-derived slugs to canonical built-in library keys, so
# cost/unit backfill works even when the user's legend uses a slightly
# different name. (e.g. "weatherproof_gpo" in a card-grid legend resolves
# to "wp_gpo" in the built-in library.)
_BUILTIN_KEY_ALIASES: dict[str, str] = {
    "weatherproof_gpo": "wp_gpo",
    "wp_general_purpose_outlet": "wp_gpo",
    "single_pole_switch_1_gang": "single_pole_switch",
    "two_way_switch_intermediate": "two_way_switch",
    "dimmer_switch": "dimmer",
    "trailing_edge_dimmer": "dimmer",
    "single_general_purpose_outlet": "single_gpo",
    "double_general_purpose_outlet": "double_gpo",
    "ceiling_oyster_light": "ceiling_light",
    "ceiling_batten_light": "ceiling_light",
    "wall_mounted_light": "wall_light",
    "recessed_led_downlight": "downlight",
    "led_downlight": "downlight",
    "ceiling_sweep_fan": "ceiling_fan",
    "bathroom_exhaust_fan": "exhaust_fan",
    "photoelectric_smoke_alarm": "smoke_detector",
    "smoke_alarm": "smoke_detector",
    "data_outlet": "data_point",
    "rj45_outlet": "data_point",
    "ethernet_outlet": "data_point",
    "tv_outlet": "tv_point",
    "tv_antenna_outlet": "tv_point",
    "consumer_unit": "distribution_board",
    "switchboard": "distribution_board",
    "main_switchboard": "distribution_board",
}


def _builtin_for(library_key: str) -> dict | None:
    """Look up a built-in library entry, applying alias remapping first."""
    canonical = _BUILTIN_KEY_ALIASES.get(library_key, library_key)
    return get_builtin_entry(canonical)


# Column header keywords we anchor on. Matching is case-insensitive substring.
_HEADER_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("symbol", "Symbol"),
    ("canonical", "Canonical item"),
    ("alias", "Alias/spec phrase"),
    ("unit", "Unit"),
    ("indicative", "Indicative AUD"),
)

DEFAULT_RENDER_SCALE = 2.5
DEFAULT_TARGET_SYMBOL_SIZE = 80    # match symbol_library/*.png standard size
DEFAULT_TARGET_W = 1786            # for resolving render scale when caller wants
DEFAULT_PSM = 6


@dataclass
class _Span:
    text: str
    x: int
    y: int
    w: int
    h: int
    conf: int

    @property
    def x_right(self) -> int:
        return self.x + self.w

    @property
    def cx(self) -> int:
        return self.x + self.w // 2

    @property
    def cy(self) -> int:
        return self.y + self.h // 2


# ---------- OCR helpers ----------

def _tesseract_spans(img: Image.Image, psm: int = DEFAULT_PSM,
                     min_conf: int = 25) -> list[_Span]:
    _ensure_configured()
    data = pytesseract.image_to_data(
        img, config=f"--psm {psm}", output_type=pytesseract.Output.DICT,
    )
    spans: list[_Span] = []
    n = len(data["text"])
    for i in range(n):
        text = (data["text"][i] or "").strip()
        if not text:
            continue
        try:
            conf = int(float(data["conf"][i]))
        except (TypeError, ValueError):
            continue
        if conf < min_conf:
            continue
        spans.append(_Span(
            text=text,
            x=int(data["left"][i]),
            y=int(data["top"][i]),
            w=int(data["width"][i]),
            h=int(data["height"][i]),
            conf=conf,
        ))
    return spans


def _find_header(spans: list[_Span]) -> Optional[dict[str, _Span]]:
    """Locate the column-header row by finding the y where the most header
    keywords appear together (within a ~40 px band)."""
    candidates: list[tuple[str, _Span]] = []
    for s in spans:
        t = s.text.lower()
        for kw, _label in _HEADER_KEYWORDS:
            if kw in t:
                candidates.append((kw, s))
                break

    if not candidates:
        return None

    # group by y proximity
    candidates.sort(key=lambda kv: kv[1].cy)
    best_group: list[tuple[str, _Span]] = []
    for i, (kw, s) in enumerate(candidates):
        group = [(kw, s)]
        for kw2, s2 in candidates[i + 1:]:
            if abs(s2.cy - s.cy) <= 40:
                group.append((kw2, s2))
            else:
                break
        # take by distinct keywords found
        distinct = {k for k, _ in group}
        if len(distinct) > len({k for k, _ in best_group}):
            best_group = group
    if len(best_group) < 3:  # need at least 3 of 5 columns to anchor reliably
        return None
    out: dict[str, _Span] = {}
    for kw, s in best_group:
        if kw not in out:
            out[kw] = s
    return out


_MAX_LAST_COLUMN_WIDTH = 400  # cap for the rightmost (indicative) column so
                              # adjacent panels (OCR notes etc.) don't bleed in


def _column_ranges(header: dict[str, _Span], page_w: int) -> dict[str, tuple[int, int]]:
    """Convert header anchors into (x_start, x_end) ranges per column.

    The "Symbol" column starts ~near the left edge (no anchor to the left of
    it). Each other column starts at its header's x-left; each column ends
    just before the next column's x-left. The rightmost column is capped to
    ~400 px so a sibling panel to the right of the table (e.g. an OCR-notes
    box) doesn't get pulled into its values."""
    ordered = ["symbol", "canonical", "alias", "unit", "indicative"]
    present = [k for k in ordered if k in header]
    starts: dict[str, int] = {}
    for k in present:
        starts[k] = header[k].x - 6
    # Always have a 'symbol' start, even if its header wasn't detected — use
    # 'canonical' minus the canonical-header width as a fallback for the icon
    # column left edge.
    if "symbol" not in starts and "canonical" in starts:
        starts["symbol"] = max(0, starts["canonical"] - 280)
        present = ["symbol"] + present

    ranges: dict[str, tuple[int, int]] = {}
    for i, k in enumerate(present):
        if i + 1 < len(present):
            end = starts[present[i + 1]] - 6
        else:
            end = min(page_w, starts[k] + _MAX_LAST_COLUMN_WIDTH)
        ranges[k] = (starts[k], end)
    return ranges


def _row_clusters(spans: list[_Span], header_y: int, max_y: int,
                  row_gap: int = 50, x_start: int = 0, x_end: int | None = None,
                  ) -> list[list[_Span]]:
    """Group spans below the header into rows by y-clustering.

    `row_gap` defaults to 50 so that a tall symbol-as-text span at the top of
    a row (Tesseract reading the icon as 'O' or 'OD' a few pixels above the
    canonical-name text) still clusters with the row's text spans.

    `x_start` / `x_end` restrict clustering to a horizontal band so the
    right-side notes panel doesn't pollute the row count.
    """
    if x_end is None:
        x_end = float("inf")
    body = [s for s in spans
            if s.cy > header_y + 20 and s.cy < max_y
            and x_start <= s.cx <= x_end]
    body.sort(key=lambda s: s.cy)
    rows: list[list[_Span]] = []
    for s in body:
        if rows and abs(s.cy - rows[-1][0].cy) <= row_gap:
            rows[-1].append(s)
        else:
            rows.append([s])
    return rows


def _spans_in_column(row: list[_Span], x_start: int, x_end: int) -> list[_Span]:
    return sorted(
        [s for s in row if x_start <= s.cx <= x_end],
        key=lambda s: s.x,
    )


def _join_text(spans: list[_Span]) -> str:
    return " ".join(s.text for s in spans).strip()


# ---------- parsing ----------

_CURRENCY_RE = re.compile(r"[^\d.\-]")


def _parse_cost(text: str) -> float:
    if not text:
        return 0.0
    cleaned = _CURRENCY_RE.sub("", text)
    if not cleaned or cleaned in {".", "-", "-."}:
        return 0.0
    try:
        return round(float(cleaned), 2)
    except ValueError:
        return 0.0


def _slugify(text: str) -> str:
    """canonical_name -> snake_case library_key."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "unnamed_symbol"


def _unique_slug(base: str, seen: set[str]) -> str:
    if base not in seen:
        return base
    n = 2
    while f"{base}_{n}" in seen:
        n += 1
    return f"{base}_{n}"


# Tokens that the icon-glyph misread can produce when it bleeds into the
# card name line. If a card's first token matches one of these AND the
# remainder of the name still reads sensibly, we drop the noise token.
_OCR_NOISE_PREFIXES = {
    "aan", "ann", "ll", "lli", "lll", "om", "we", "wp", "ss", "tt",
    "co", "po", "po2", "le", "ot", "oo", "cc", "ce", "ce2",
    "do", "fo", "rr", "uu", "nn", "an",
}


def _strip_ocr_noise_prefix(name: str) -> str:
    """Drop a leading icon-misread token like 'aan' from 'aan Wall Light'.

    Only strips if (a) first token matches a known noise pattern AND
    (b) removing it leaves >=2 tokens of >=3 chars each.
    """
    parts = name.strip().split()
    if len(parts) < 2:
        return name
    first = parts[0].lower().strip("()[]{}.,")
    if first not in _OCR_NOISE_PREFIXES:
        return name
    remaining = parts[1:]
    if len(remaining) < 1 or not any(len(p) >= 3 for p in remaining):
        return name
    return " ".join(remaining)


# ---------- symbol crop ----------

def _tight_crop_dark_pixels(img: Image.Image, dark_threshold: int = 180,
                            pad: int = 4) -> Image.Image:
    """Trim whitespace around the icon's dark pixels so the template is
    the symbol's actual silhouette, not symbol + padding.

    Without this, template-matching at scale 1.0 wastes the surrounding
    whitespace and effectively means "match a symbol of true_size pixels
    against a target region of (true_size + padding) pixels" — which biases
    detection toward larger plan symbols and misses the small ones.
    """
    rgb = img.convert("RGB")
    px = rgb.load()
    w, h = rgb.size
    min_x, min_y, max_x, max_y = w, h, -1, -1
    for y in range(h):
        for x in range(w):
            r, g, b = px[x, y]
            if (r + g + b) // 3 < dark_threshold:
                if x < min_x: min_x = x
                if y < min_y: min_y = y
                if x > max_x: max_x = x
                if y > max_y: max_y = y
    if max_x < min_x or max_y < min_y:
        return img  # no dark pixels found — return as-is
    x0 = max(0, min_x - pad)
    y0 = max(0, min_y - pad)
    x1 = min(w, max_x + pad + 1)
    y1 = min(h, max_y + pad + 1)
    # keep aspect 1:1 — pad the shorter side so resize doesn't squash
    w2 = x1 - x0
    h2 = y1 - y0
    if w2 != h2:
        side = max(w2, h2)
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        x0 = max(0, cx - side // 2)
        y0 = max(0, cy - side // 2)
        x1 = min(w, x0 + side)
        y1 = min(h, y0 + side)
    return img.crop((x0, y0, x1, y1))


def _crop_symbol(page: Image.Image, icon_cx: int, row_cy: int,
                 target_size: int, row_spacing: int,
                 ) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """Crop a square symbol icon centred on (icon_cx, row_cy), tight-crop
    to the symbol's dark-pixel silhouette, then resize to target_size.

    The initial crop window is sized to ~80 % of the row spacing so adjacent
    rows don't leak in vertically. The tight-crop step then removes the
    whitespace around the icon so the template is the symbol itself, not
    symbol + padding — which matters for matching small plan symbols.
    """
    side = max(50, int(row_spacing * 0.85))
    half = side // 2
    x0 = max(0, icon_cx - half)
    x1 = min(page.width, icon_cx + half)
    y0 = max(0, row_cy - half)
    y1 = min(page.height, row_cy + half)
    crop = page.crop((x0, y0, x1, y1)).convert("RGBA")
    # square it in case we hit a page edge
    s = min(crop.width, crop.height)
    if crop.width != crop.height:
        ox = (crop.width - s) // 2
        oy = (crop.height - s) // 2
        crop = crop.crop((ox, oy, ox + s, oy + s))
    # tight-crop to dark pixels so template = symbol, not symbol + padding
    crop = _tight_crop_dark_pixels(crop)
    crop = crop.resize((target_size, target_size), Image.LANCZOS)
    return crop, (x0, y0, x1, y1)


def _resolve_icon_cx(symbol_spans_in_row: list[_Span],
                     symbol_x_range: tuple[int, int],
                     header_symbol_span: _Span | None) -> int:
    """Pick the most likely icon x-centre for this row.

    1. If Tesseract picked up any text inside the symbol column (often it
       interprets an open circle as 'O', a switch as 'S', etc.) use the
       centroid of those spans.
    2. Otherwise fall back to the symbol-column-header's centre.
    3. Last resort: middle of the column x-range.
    """
    if symbol_spans_in_row:
        return sum(s.cx for s in symbol_spans_in_row) // len(symbol_spans_in_row)
    if header_symbol_span is not None:
        return header_symbol_span.cx
    return sum(symbol_x_range) // 2


# ---------- card-grid fallback ----------

_TITLE_KEYWORDS = ("ELECTRICAL SYMBOL LIBRARY", "SYMBOL LIBRARY", "LEGEND")

# Words that indicate the start of a section AFTER the legend grid (extraction
# notes, plan-intelligence summary, title block, etc.). Used to cap the body
# region so card-grid clustering doesn't pull in bottom-of-page noise.
_SECTION_BREAK_KEYWORDS = {
    "EXTRACTION", "NOTES", "PLAN", "INTELLIGENCE", "SUMMARY",
    "IMPORTANT", "REVISION", "DRAWING", "ADDRESS", "SCALE",
    "CLIENT", "WEIGHT", "STYLE",
}


def _find_title_anchor(spans: list[_Span]) -> _Span | None:
    """Locate the legend's section title (e.g. 'ELECTRICAL SYMBOL LIBRARY').

    Prefers the topmost (smallest cy) candidate among the keywords so we
    don't accidentally anchor on a title-block 'PROJECT' or similar far
    down the page.
    """
    candidates: list[_Span] = []
    upper = [(s, s.text.upper()) for s in spans]
    by_text = {t: s for s, t in upper}
    for s, t in upper:
        # two-word: SYMBOL + LIBRARY adjacent in Y
        if t == "SYMBOL":
            for ns, nt in upper:
                if nt == "LIBRARY" and abs(ns.cy - s.cy) <= 20:
                    candidates.append(s)
                    break
        elif t == "LEGEND":
            candidates.append(s)
    if not candidates:
        # fallback to LIBRARY alone (only if SYMBOL didn't appear)
        if "LIBRARY" in by_text:
            candidates.append(by_text["LIBRARY"])
    if not candidates:
        return None
    candidates.sort(key=lambda s: s.cy)
    return candidates[0]


def _find_next_section_break(spans: list[_Span], after_y: int,
                             page_h: int) -> int:
    """Find the y of the next section-header below `after_y`. Returns page_h
    if no break detected. Used as the max_y for body extraction."""
    breaks = [s.cy for s in spans
              if s.cy > after_y + 50 and s.text.upper() in _SECTION_BREAK_KEYWORDS]
    if not breaks:
        return page_h
    return min(breaks) - 20


def _cluster_columns(spans: list[_Span], page_w: int,
                     min_gap: int = 200) -> list[tuple[int, int]]:
    """Detect column x-ranges from name-text spans by gap analysis.

    Returns a list of (x_start, x_end) per detected column, left-to-right.
    """
    if not spans:
        return []
    # collect leftmost x of each span as candidate column starts
    xs = sorted(s.x for s in spans)
    # naive 1-D clustering: split where gap > min_gap
    columns: list[list[int]] = [[xs[0]]]
    for x in xs[1:]:
        if x - columns[-1][-1] > min_gap:
            columns.append([x])
        else:
            columns[-1].append(x)
    # convert clusters to (start, end) — end of column N = start of column N+1 minus a margin
    ranges: list[tuple[int, int]] = []
    starts = [c[0] - 10 for c in columns]
    for i, start in enumerate(starts):
        end = (starts[i + 1] - 10) if i + 1 < len(starts) else page_w
        ranges.append((start, end))
    return ranges


def _extract_card_grid(
    spans: list[_Span],
    page: Image.Image,
    pdf_name: str,
    page_index: int,
    target_symbol_size: int,
    warnings: list[str],
) -> list[LegendRow]:
    """Extract legend rows from a card-grid layout (no column headers).

    Strategy:
      - Restrict the search region to roughly the top 70 % of the page (so
        the bottom title block / POC notes don't pollute clustering). If a
        title anchor like 'ELECTRICAL SYMBOL LIBRARY' is detected and is
        well above that, use it instead.
      - Detect columns from the x-distribution of bold name-spans.
      - Within each column, cluster cards by y-gap.
      - For each card, the first line of text is the canonical name and
        any following sub-line is the alias.
      - Skip any "card" whose only text is a single section-header word
        (defensive against false positives).
    """
    # Region of interest: title (if at top of page) -> 70% page height
    title = _find_title_anchor(spans)
    page_h = page.height
    region_top = 100
    if title is not None and title.cy < page_h * 0.4:
        region_top = title.cy + 30
    region_bottom = int(page_h * 0.70)
    body_max_y = _find_next_section_break(spans, after_y=region_top, page_h=region_bottom)
    region_bottom = min(region_bottom, body_max_y)

    body = [s for s in spans
            if region_top < s.cy < region_bottom and s.text.strip()
            and s.text.upper() not in _SECTION_BREAK_KEYWORDS]
    if not body:
        warnings.append(
            f"Card-grid fallback: no text in region y=[{region_top}, {region_bottom}]."
        )
        return []

    # "Name" spans are taller (bold) — use them to detect column structure
    name_candidates = [s for s in body if s.h >= 18]
    if len(name_candidates) < 3:
        name_candidates = body
    columns = _cluster_columns(name_candidates, page.width, min_gap=200)
    if not columns or len(columns) < 2:
        warnings.append(
            f"Card-grid fallback: detected {len(columns)} column(s); need at least 2."
        )
        return []

    # Per-column row clustering
    cards: list[tuple[int, int, list[_Span]]] = []  # (col_idx, row_y, spans)
    for col_idx, (cx_start, cx_end) in enumerate(columns):
        col_spans = sorted(
            [s for s in body if cx_start <= s.x < cx_end],
            key=lambda s: s.cy,
        )
        # cluster into cards by y-gap: spans within ~60 px = same card
        col_cards: list[list[_Span]] = []
        for s in col_spans:
            if col_cards and s.cy - col_cards[-1][-1].cy <= 60:
                col_cards[-1].append(s)
            else:
                col_cards.append([s])
        for c in col_cards:
            if len(c) < 2:  # bare-icon rows or noise — skip
                continue
            row_y = sum(sp.cy for sp in c) // len(c)
            cards.append((col_idx, row_y, c))

    # estimate row spacing (within a column) for icon-crop sizing
    same_col = sorted([c for c in cards if c[0] == 0], key=lambda c: c[1])
    if len(same_col) >= 2:
        gaps = [same_col[i + 1][1] - same_col[i][1] for i in range(len(same_col) - 1)]
        row_spacing = int(sum(gaps) / len(gaps))
    else:
        row_spacing = 150

    # order cards top-to-bottom, then left-to-right
    cards.sort(key=lambda c: (c[1] // (row_spacing // 2 or 1), c[0]))

    out: list[LegendRow] = []
    seen: set[str] = set()
    for idx, (col_idx, row_y, card_spans) in enumerate(cards):
        # split card spans into "lines" by y; first line = name, rest = alias
        card_spans = sorted(card_spans, key=lambda s: (s.cy, s.x))
        lines: list[list[_Span]] = []
        for s in card_spans:
            if lines and abs(s.cy - lines[-1][0].cy) <= 12:
                lines[-1].append(s)
            else:
                lines.append([s])
        name_line = lines[0]
        alias_lines = lines[1:]
        canonical = " ".join(s.text for s in sorted(name_line, key=lambda s: s.x)).strip()
        alias = " ".join(
            s.text for ln in alias_lines for s in sorted(ln, key=lambda s: s.x)
        ).strip()
        if not canonical or len(canonical) < 2:
            continue

        # Strip OCR-noise prefixes: when the icon glyph leaks into the
        # name line ('aan Wall Light', '(we) Weatherproof GPO'), the first
        # token is short, lowercase, and doesn't look like an English word.
        # If removing it leaves a meaningful name, do so.
        canonical = _strip_ocr_noise_prefix(canonical)

        raw_slug = _slugify(canonical)
        # Normalise via the built-in alias map so 'Dimmer switch' and
        # 'Dimmer' collapse to the same key ('dimmer'), preventing duplicate
        # classes in the user pack that produce two parallel detection
        # streams against the same plan symbols.
        canonical_slug = _BUILTIN_KEY_ALIASES.get(raw_slug, raw_slug)
        slug = _unique_slug(canonical_slug, seen)
        seen.add(slug)

        # icon position: prefer Tesseract-picked-up icon-as-text spans
        # (e.g. WP GPO icon read as '(we)', dimmer icon read as '(om)').
        # Indicators: leftmost span is much taller than the bulk text
        # (icons render >=40 px tall) OR there's a substantial x-gap
        # between leftmost and the next span. Otherwise fall back to an
        # offset from the leftmost text (icon physically sits to its left).
        spans_by_x = sorted(card_spans, key=lambda s: s.x)
        leftmost = spans_by_x[0]
        gap = (spans_by_x[1].x - (leftmost.x + leftmost.w)
               if len(spans_by_x) > 1 else 0)
        tall_for_icon = leftmost.h >= 40
        short_text = len(leftmost.text) <= 5
        if (tall_for_icon or gap > 30) and short_text:
            icon_cx = leftmost.cx
        else:
            icon_cx = max(0, leftmost.x - 90)
        crop, bbox = _crop_symbol(page, icon_cx, row_y, target_symbol_size, row_spacing)

        # Card-grid legends have no cost/unit columns; backfill from the
        # built-in library when the user's symbol key matches one of our
        # 15 built-ins (directly or via _BUILTIN_KEY_ALIASES). User can
        # still override in the Tab 6 preview before ingesting.
        cost = 0.0
        unit = "ea"
        bi = _builtin_for(slug)
        if bi is not None:
            cost = float(bi.get("indicative_cost_aud", 0.0))
            unit = bi.get("unit", "ea") or "ea"

        out.append(LegendRow(
            library_key=slug,
            canonical_name=canonical,
            alias=alias,
            unit=unit,
            indicative_cost_aud=cost,
            symbol_bbox=bbox,
            source_pdf=pdf_name,
            source_page=page_index,
            row_index=idx,
            symbol_image_b64=_png_to_b64(crop),
        ))
    return out


def _png_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ---------- entry point ----------

def render_page(pdf_path: Path, page_index_one_based: int,
                render_scale: float = DEFAULT_RENDER_SCALE) -> Image.Image:
    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        page = pdf[page_index_one_based - 1]
        base_scale = DEFAULT_TARGET_W / page.get_size()[0]
        return page.render(scale=base_scale * render_scale).to_pil()
    finally:
        pdf.close()


def extract_legend(
    pdf_path: Path,
    page_index_one_based: int,
    *,
    render_scale: float = DEFAULT_RENDER_SCALE,
    target_symbol_size: int = DEFAULT_TARGET_SYMBOL_SIZE,
    psm: int = DEFAULT_PSM,
    max_rows: int = 40,
) -> LegendExtraction:
    page = render_page(pdf_path, page_index_one_based, render_scale=render_scale)
    spans = _tesseract_spans(page, psm=psm)

    warnings: list[str] = []
    extraction = LegendExtraction(
        source_pdf=pdf_path.name,
        source_page=page_index_one_based,
        page_image_width=page.width,
        page_image_height=page.height,
        rows=[],
        warnings=warnings,
    )

    header = _find_header(spans)
    if header is None:
        # Tabular extractor failed (no column headers detected) — try the
        # card-grid layout used by variant_b / variant_c (each symbol is in
        # its own labelled card; no Unit / Cost columns).
        warnings.append(
            "Tabular column headers not found — falling back to card-grid layout."
        )
        rows = _extract_card_grid(
            spans, page, pdf_path.name, page_index_one_based,
            target_symbol_size, warnings,
        )
        extraction.rows = rows
        if not rows:
            warnings.append(
                "Neither tabular nor card-grid layout matched this page. "
                "Use a page that contains a legend (table or grid of symbols + names)."
            )
        return extraction

    header_y = max(h.cy for h in header.values())
    ranges = _column_ranges(header, page.width)
    if "symbol" not in ranges or "canonical" not in ranges:
        warnings.append("Symbol or Canonical column could not be located.")
        return extraction

    # body region: from just below header to either the next big header below,
    # or the page bottom. We trim at the first 'LINE WEIGHT' / 'STYLE SAMPLES'
    # marker if present (it's the section after the table in our sample).
    end_y = page.height
    for s in spans:
        if s.cy > header_y + 40 and ("STYLE" in s.text.upper() or "WEIGHT" in s.text.upper()):
            end_y = min(end_y, s.y - 5)

    # Constrain row-clustering to the table's horizontal band so spans from
    # the adjacent OCR-notes / extraction-summary panel don't get pulled in.
    table_x_start = min(r[0] for r in ranges.values())
    table_x_end = max(r[1] for r in ranges.values())
    raw_rows = _row_clusters(
        spans, header_y=header_y, max_y=end_y,
        x_start=table_x_start, x_end=table_x_end,
    )
    if not raw_rows:
        warnings.append("No body rows found below the legend header.")
        return extraction

    # estimate average row spacing in source-px so we can size crops without
    # leaking adjacent rows into them
    row_centres = [sum(s.cy for s in row) / len(row) for row in raw_rows]
    if len(row_centres) >= 2:
        gaps = [row_centres[i + 1] - row_centres[i] for i in range(len(row_centres) - 1)]
        row_spacing = int(sum(gaps) / len(gaps))
    else:
        row_spacing = 75

    seen: set[str] = set()
    out_rows: list[LegendRow] = []
    for row_idx, row in enumerate(raw_rows[:max_rows]):
        row_cy = sum(s.cy for s in row) // len(row)

        canonical_spans = _spans_in_column(row, *ranges["canonical"])
        canonical_name = _join_text(canonical_spans)
        if not canonical_name:
            warnings.append(f"Row at y={row_cy}: blank canonical name; skipped.")
            continue

        alias = _join_text(_spans_in_column(row, *ranges.get("alias", (0, 0)))) \
            if "alias" in ranges else ""
        unit = _join_text(_spans_in_column(row, *ranges.get("unit", (0, 0)))) or "ea" \
            if "unit" in ranges else "ea"
        cost_text = _join_text(_spans_in_column(row, *ranges.get("indicative", (0, 0)))) \
            if "indicative" in ranges else ""
        cost = _parse_cost(cost_text)
        if "indicative" in ranges and not cost_text:
            warnings.append(f"Row '{canonical_name}': cost cell empty.")

        raw_slug = _slugify(canonical_name)
        # Normalise via the built-in alias map so e.g. 'dimmer_switch'
        # collapses to 'dimmer'. Prevents duplicate classes in the user pack.
        canonical_slug = _BUILTIN_KEY_ALIASES.get(raw_slug, raw_slug)
        slug = _unique_slug(canonical_slug, seen)
        seen.add(slug)

        # Backfill cost from built-in library if Tesseract returned a blank
        # cost cell (this also covers card-grid layouts which have no cost
        # column at all — handled in _extract_card_grid below).
        if cost == 0.0:
            bi = _builtin_for(slug)
            if bi is not None:
                cost = float(bi.get("indicative_cost_aud", 0.0))

        symbol_spans_in_row = _spans_in_column(row, *ranges["symbol"])
        icon_cx = _resolve_icon_cx(
            symbol_spans_in_row, ranges["symbol"], header.get("symbol"),
        )
        crop, bbox = _crop_symbol(
            page, icon_cx, row_cy, target_symbol_size, row_spacing,
        )

        out_rows.append(LegendRow(
            library_key=slug,
            canonical_name=canonical_name,
            alias=alias,
            unit=unit,
            indicative_cost_aud=cost,
            symbol_bbox=bbox,
            source_pdf=pdf_path.name,
            source_page=page_index_one_based,
            row_index=row_idx,
            symbol_image_b64=_png_to_b64(crop),
        ))

    extraction.rows = out_rows
    return extraction
