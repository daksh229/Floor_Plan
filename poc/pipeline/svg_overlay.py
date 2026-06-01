"""SVG vector overlay over the raster page image, embedded in an HTML
component with pan + zoom.

Why: when you ctrl-+ in the browser on a plain `st.image()`, the underlying
raster gets upscaled — bboxes and labels go pixelated. Here the page is
still raster (PDF render), but bboxes and labels are SVG. SVG strokes /
text stay crisp at any zoom level.

The component embeds the page image as base64 in an `<img>` tag and overlays
absolutely-positioned `<svg>` shapes on top. A tiny inline pan-zoom JS
implementation (no external library) lets the user drag and scroll-zoom
inside the component box.

Used by:
  - app._tab_cv() — replaces / augments the plain st.image() overlay.
"""
from __future__ import annotations

import base64
import io
import json
from typing import Iterable

from PIL import Image

from .schemas import Detection


def _img_to_base64(img: Image.Image, fmt: str = "JPEG", quality: int = 90) -> str:
    """JPEG with quality 90 keeps file size sensible while staying visually
    indistinguishable from the source PNG for the purposes of detection review."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format=fmt, quality=quality, optimize=True)
    return base64.standard_b64encode(buf.getvalue()).decode("ascii")


def _colour_for_class(symbol_class: str) -> str:
    # Same deterministic colour as pipeline.overlay._color_for_class but
    # emitted as CSS rgb() string instead of an (r,g,b) tuple.
    import hashlib
    h = hashlib.md5(symbol_class.encode("utf-8")).digest()
    r = 60 + h[0] % 180
    g = 60 + h[1] % 180
    b = 60 + h[2] % 180
    return f"rgb({r},{g},{b})"


def build_svg_overlay_html(
    page_image: Image.Image,
    detections: Iterable[Detection],
    *,
    component_height_px: int = 720,
    show_score: bool = True,
) -> str:
    """Return a complete HTML snippet (string) suitable for
    `streamlit.components.v1.html(..., height=component_height_px+40)`.

    The result is a fixed-size scrollable container with the page as a
    raster background and bboxes/labels as SVG. Drag to pan; mouse-wheel
    to zoom (centred on cursor); double-click to reset.
    """
    img_w, img_h = page_image.width, page_image.height
    img_b64 = _img_to_base64(page_image)

    dets = list(detections)
    # SVG strokes and labels — stroke-width is in SVG user units (== image px)
    # but stays crisp under CSS transform because SVG re-rasterises on zoom
    stroke = max(3, round(img_w / 400))
    font_px = max(12, round(img_w / 110))
    label_pad = max(3, font_px // 3)

    svg_elems: list[str] = []
    for d in dets:
        x0, y0, x1, y1 = d.bbox
        c = _colour_for_class(d.symbol_class)
        label = d.symbol_class.replace("_", " ")
        if show_score:
            label = f"{label} {d.score:.2f}"
        text_w_est = int(len(label) * font_px * 0.55)
        text_h_est = font_px + 2 * label_pad
        ly0 = max(0, y0 - text_h_est)
        ly1 = ly0 + text_h_est
        # Wrap the bbox + label-bg + label-text in a <g class="detection"
        # data-class="..."> so the JS highlight logic can target detections
        # by class as a single visual unit (legend hover -> dim non-matching).
        svg_elems.append(
            f'<g class="detection" data-class="{_escape_attr(d.symbol_class)}">'
            f'<rect x="{x0}" y="{y0}" width="{x1 - x0}" height="{y1 - y0}" '
            f'fill="none" stroke="{c}" stroke-width="{stroke}" />'
            f'<rect x="{x0}" y="{ly0}" width="{text_w_est + 2 * label_pad}" '
            f'height="{text_h_est}" fill="{c}" fill-opacity="0.85" />'
            f'<text x="{x0 + label_pad}" y="{ly1 - label_pad - 2}" '
            f'font-family="system-ui, sans-serif" font-size="{font_px}" '
            f'font-weight="600" fill="white">{_escape_text(label)}</text>'
            f'</g>'
        )

    svg = (
        f'<svg id="vec-svg" xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {img_w} {img_h}" '
        f'style="position:absolute;top:0;left:0;width:100%;height:100%;'
        f'pointer-events:none;">'
        + "".join(svg_elems)
        + "</svg>"
    )

    # Inline pan-zoom JS — no external libraries. Tracks (scale, tx, ty),
    # applies CSS transform to the inner content wrapper.
    js = """
<script>
(function() {
  const root = document.getElementById('vec-root');
  const inner = document.getElementById('vec-inner');
  const reset = document.getElementById('vec-reset');
  if (!root || !inner) return;
  let scale = 1.0, tx = 0, ty = 0;
  let dragging = false, lastX = 0, lastY = 0;
  function apply() {
    inner.style.transform = `translate(${tx}px, ${ty}px) scale(${scale})`;
  }
  function resetView() { scale = 1.0; tx = 0; ty = 0; apply(); }
  resetView();

  root.addEventListener('wheel', (e) => {
    e.preventDefault();
    const rect = root.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const oldScale = scale;
    const k = e.deltaY < 0 ? 1.15 : (1 / 1.15);
    const newScale = Math.max(0.2, Math.min(10, scale * k));
    // zoom around cursor: keep (mx,my) fixed in image coords
    tx = mx - ((mx - tx) * (newScale / oldScale));
    ty = my - ((my - ty) * (newScale / oldScale));
    scale = newScale;
    apply();
  }, { passive: false });

  root.addEventListener('mousedown', (e) => {
    dragging = true; lastX = e.clientX; lastY = e.clientY;
    root.style.cursor = 'grabbing';
  });
  window.addEventListener('mousemove', (e) => {
    if (!dragging) return;
    tx += (e.clientX - lastX);
    ty += (e.clientY - lastY);
    lastX = e.clientX; lastY = e.clientY;
    apply();
  });
  window.addEventListener('mouseup', () => {
    dragging = false; root.style.cursor = 'grab';
  });
  root.addEventListener('dblclick', resetView);
  if (reset) reset.addEventListener('click', resetView);
  root.style.cursor = 'grab';
})();
</script>
""".strip()

    # Per-class counts power the legend's count badges. Sorted desc by count
    # so the highest-cardinality classes (typically downlights / GPOs) are
    # at the top where they're easiest to reach.
    class_counts: dict[str, int] = {}
    for d in dets:
        class_counts[d.symbol_class] = class_counts.get(d.symbol_class, 0) + 1
    legend_classes = sorted(class_counts.keys(), key=lambda c: (-class_counts[c], c))

    legend_rows_html = "".join(
        f'<div class="legend-row" data-class="{_escape_attr(c)}" '
        f'title="Hover to isolate · Click to lock · Esc to clear">'
        f'<span class="legend-swatch" style="background:{_colour_for_class(c)};"></span>'
        f'<span class="legend-name">{_escape_text(c.replace("_", " "))}</span>'
        f'<span class="legend-count">{class_counts[c]}</span>'
        f'</div>'
        for c in legend_classes
    )

    # Interactive-legend CSS + JS. The pattern:
    #   - All <g.detection> groups have a stable data-class attribute
    #   - Hovering a .legend-row injects a <style> rule that dims every
    #     non-matching detection group and every non-matching legend row
    #   - Clicking a row toggles "locked" mode — the dim survives mouse
    #     movement until the same row is clicked again or Esc is pressed
    # We use injected CSS (not per-element class toggling) because there
    # can be hundreds of detection groups; a single rule update is O(1).
    interactive_css = """
<style>
  .bom-overlay-grid {
    display: grid;
    grid-template-columns: 1fr 240px;
    gap: 10px;
    align-items: start;
  }
  .legend-panel {
    border: 1px solid #ccc;
    border-radius: 4px;
    background: #fafafa;
    padding: 6px;
    max-height: __HEIGHT__px;
    overflow-y: auto;
    font-family: system-ui, sans-serif;
    font-size: 12px;
  }
  .legend-header {
    display:flex; justify-content:space-between; align-items:center;
    font-weight:600; color:#444; padding:4px 6px 8px;
    border-bottom:1px solid #ddd; margin-bottom:4px;
  }
  .legend-clear {
    font-size:11px; padding:2px 8px; cursor:pointer; border-radius:3px;
    border:1px solid #999; background:#fff; color:#555;
  }
  .legend-clear:hover { background:#eee; }
  .legend-row {
    display:flex; align-items:center; padding:5px 6px; margin:2px 0;
    border-radius:3px; cursor:pointer; user-select:none;
    transition: background 0.1s, opacity 0.15s;
  }
  .legend-row:hover { background:#e8e8e8; }
  .legend-row.locked {
    background:#dfe9f5; outline:2px solid #4a7bbd; outline-offset:-2px;
  }
  .legend-swatch {
    display:inline-block; width:14px; height:14px;
    border:1px solid #777; border-radius:3px; margin-right:8px;
    flex-shrink:0;
  }
  .legend-name { flex:1; color:#222; }
  .legend-count {
    background:#666; color:#fff; font-weight:600;
    padding:1px 7px; border-radius:9px; font-size:11px; min-width:18px;
    text-align:center;
  }
  .legend-row.locked .legend-count { background:#4a7bbd; }
  /* When any class is active, dim all detections + rows by default; the
     dynamic <style> below un-dims the matching set. */
  #vec-svg.has-active g.detection { opacity: 0.12; transition: opacity 0.15s; }
  #legend-panel.has-active .legend-row { opacity: 0.45; }
  /* Cursor hint when locked */
  .legend-panel[data-locked="true"] .legend-row { cursor: pointer; }
</style>
""".replace("__HEIGHT__", str(component_height_px))

    interactive_js = """
<script>
(function() {
  const svg = document.getElementById('vec-svg');
  const legend = document.getElementById('legend-panel');
  if (!svg || !legend) return;
  const rows = legend.querySelectorAll('.legend-row');
  const clearBtn = legend.querySelector('.legend-clear');
  let lockedClass = null;
  const dyn = document.createElement('style');
  document.head.appendChild(dyn);

  function setActive(cls) {
    if (!cls) {
      svg.classList.remove('has-active');
      legend.classList.remove('has-active');
      dyn.textContent = '';
      return;
    }
    svg.classList.add('has-active');
    legend.classList.add('has-active');
    // Quote-safety: data-class values come from server-escaped slugs
    // (snake_case, only [a-z0-9_]), so a plain attribute selector is safe.
    dyn.textContent =
      '#vec-svg.has-active g.detection[data-class="' + cls + '"] { opacity: 1; }\\n' +
      '#legend-panel.has-active .legend-row[data-class="' + cls + '"] { opacity: 1; }';
  }

  rows.forEach(row => {
    row.addEventListener('mouseenter', () => {
      if (!lockedClass) setActive(row.dataset.class);
    });
    row.addEventListener('mouseleave', () => {
      if (!lockedClass) setActive(null);
    });
    row.addEventListener('click', () => {
      const target = row.dataset.class;
      if (lockedClass === target) {
        // Toggle off
        lockedClass = null;
        row.classList.remove('locked');
        legend.removeAttribute('data-locked');
        setActive(null);
      } else {
        // Replace lock
        legend.querySelectorAll('.legend-row.locked').forEach(r => r.classList.remove('locked'));
        lockedClass = target;
        row.classList.add('locked');
        legend.setAttribute('data-locked', 'true');
        setActive(lockedClass);
      }
    });
  });

  if (clearBtn) {
    clearBtn.addEventListener('click', () => {
      lockedClass = null;
      legend.querySelectorAll('.legend-row.locked').forEach(r => r.classList.remove('locked'));
      legend.removeAttribute('data-locked');
      setActive(null);
    });
  }

  // Esc clears lock (mirrors the clear button)
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && lockedClass) {
      lockedClass = null;
      legend.querySelectorAll('.legend-row.locked').forEach(r => r.classList.remove('locked'));
      legend.removeAttribute('data-locked');
      setActive(null);
    }
  });
})();
</script>
""".strip()

    html = f"""
{interactive_css}
<div style="font-family:system-ui,sans-serif;">
  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px;">
    <div style="font-size:13px;color:#555;">
      <strong>Vector overlay</strong> — drag to pan · scroll to zoom ·
      double-click to reset · <em>hover legend to isolate a class · click to lock</em>
    </div>
    <button id="vec-reset" style="font-size:12px;padding:4px 10px;cursor:pointer;
      border:1px solid #999;background:#f5f5f5;border-radius:4px;">
      Reset view
    </button>
  </div>
  <div class="bom-overlay-grid">
    <div id="vec-root"
         style="position:relative;width:100%;height:{component_height_px}px;
                overflow:hidden;border:1px solid #ccc;background:#eee;
                user-select:none;">
      <div id="vec-inner"
           style="position:absolute;top:0;left:0;width:100%;height:100%;
                  transform-origin:0 0;">
        <div style="position:relative;width:100%;height:100%;">
          <img src="data:image/jpeg;base64,{img_b64}"
               style="width:100%;height:100%;object-fit:contain;display:block;"
               draggable="false" />
          {svg}
        </div>
      </div>
    </div>
    <div id="legend-panel" class="legend-panel">
      <div class="legend-header">
        <span>Detected classes ({len(legend_classes)})</span>
        <button class="legend-clear" title="Esc">Clear</button>
      </div>
      {legend_rows_html}
    </div>
  </div>
</div>
{js}
{interactive_js}
"""
    return html


def _escape_attr(s: str) -> str:
    """Escape for use inside double-quoted HTML attributes. Class slugs are
    already snake_case ASCII so this is mostly defensive."""
    return (
        s.replace("&", "&amp;").replace('"', "&quot;")
        .replace("<", "&lt;").replace(">", "&gt;")
    )


def _escape_text(s: str) -> str:
    return (
        s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        .replace('"', "&quot;")
    )
