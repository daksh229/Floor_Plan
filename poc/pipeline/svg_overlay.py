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
        # bbox
        svg_elems.append(
            f'<rect x="{x0}" y="{y0}" width="{x1 - x0}" height="{y1 - y0}" '
            f'fill="none" stroke="{c}" stroke-width="{stroke}" />'
        )
        # label background
        svg_elems.append(
            f'<rect x="{x0}" y="{ly0}" width="{text_w_est + 2 * label_pad}" '
            f'height="{text_h_est}" fill="{c}" fill-opacity="0.85" />'
        )
        # label text
        svg_elems.append(
            f'<text x="{x0 + label_pad}" y="{ly1 - label_pad - 2}" '
            f'font-family="system-ui, sans-serif" font-size="{font_px}" '
            f'font-weight="600" fill="white">{_escape_text(label)}</text>'
        )

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
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

    legend_classes = sorted({d.symbol_class for d in dets})
    legend_html = "".join(
        f'<span style="display:inline-flex;align-items:center;margin:0 10px 4px 0;'
        f'font-family:system-ui,sans-serif;font-size:12px;color:#333;">'
        f'<span style="display:inline-block;width:12px;height:12px;'
        f'background:{_colour_for_class(c)};border:1px solid #888;'
        f'margin-right:4px;border-radius:2px;"></span>'
        f'<code>{c}</code></span>'
        for c in legend_classes
    )

    html = f"""
<div style="font-family:system-ui,sans-serif;">
  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px;">
    <div style="font-size:13px;color:#555;">
      <strong>Vector overlay</strong> — drag to pan · scroll to zoom ·
      double-click to reset
    </div>
    <button id="vec-reset" style="font-size:12px;padding:4px 10px;cursor:pointer;
      border:1px solid #999;background:#f5f5f5;border-radius:4px;">
      Reset view
    </button>
  </div>
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
  <div style="margin-top:8px;">{legend_html}</div>
</div>
{js}
"""
    return html


def _escape_text(s: str) -> str:
    return (
        s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        .replace('"', "&quot;")
    )
