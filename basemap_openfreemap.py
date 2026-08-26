"""Render OpenFreeMap vector styles to PNG (no API key, labels stripped)."""

from __future__ import annotations

import base64
import io
import json
import math
import urllib.request
from typing import Any

from PIL import Image

OPENFREEMAP_STYLE_IDS = {
    "Positron": "positron",
    "Voyager": "bright",
    "DarkMatter": "dark",
    "Bright": "bright",
    "Liberty": "liberty",
    "Fiord": "fiord",
}

OPENFREEMAP_ATTRIBUTION = (
    "© OpenStreetMap contributors © OpenMapTiles © OpenFreeMap (MapLibre)"
)

_STYLE_CACHE: dict[str, dict[str, Any]] = {}


def fetch_nolabel_style(style_id: str) -> dict[str, Any]:
    if style_id in _STYLE_CACHE:
        return _STYLE_CACHE[style_id]

    url = f"https://tiles.openfreemap.org/styles/{style_id}"
    req = urllib.request.Request(url, headers={"User-Agent": "MappingSiteExport/1.0"})
    with urllib.request.urlopen(req, timeout=90) as resp:
        style = json.load(resp)
    style["layers"] = [layer for layer in style["layers"] if layer.get("type") != "symbol"]
    _STYLE_CACHE[style_id] = style
    return style


def _sub_bounds(
    bbox_wgs84: tuple[float, float, float, float],
    col: int,
    row: int,
    cols: int,
    rows: int,
) -> tuple[float, float, float, float]:
    west, south, east, north = bbox_wgs84
    lon_step = (east - west) / cols
    lat_step = (north - south) / rows
    tile_west = west + col * lon_step
    tile_east = west + (col + 1) * lon_step
    tile_north = north - row * lat_step
    tile_south = north - (row + 1) * lat_step
    return tile_west, tile_south, tile_east, tile_north


def _render_html(style: dict[str, Any], width: int, height: int, bbox_wgs84: tuple[float, float, float, float]) -> str:
    west, south, east, north = bbox_wgs84
    style_json = json.dumps(style)
    return f"""<!DOCTYPE html>
<html><head>
<link href="https://unpkg.com/maplibre-gl@4/dist/maplibre-gl.css" rel="stylesheet" />
<script src="https://unpkg.com/maplibre-gl@4/dist/maplibre-gl.js"></script>
<style>html,body,#map{{margin:0;padding:0;width:{width}px;height:{height}px;overflow:hidden;}}</style>
</head><body><div id="map"></div>
<script>
const style = {style_json};
const map = new maplibregl.Map({{
  container: "map",
  style: style,
  bounds: [[{west}, {south}], [{east}, {north}]],
  fitBoundsOptions: {{ padding: 0, animate: false }},
  interactive: false,
  preserveDrawingBuffer: true,
  pixelRatio: 2,
  attributionControl: false,
}});
map.on("idle", () => {{ window.__mapReady = true; }});
map.on("error", (e) => {{ window.__mapError = String(e && e.error || e); }});
</script></body></html>"""


def _render_tile(page, style: dict[str, Any], width: int, height: int, bbox_wgs84: tuple[float, float, float, float]) -> Image.Image:
    page.goto("about:blank", wait_until="domcontentloaded")
    page.set_viewport_size({"width": width, "height": height})
    page.set_content(_render_html(style, width, height, bbox_wgs84), wait_until="load")
    page.wait_for_function("window.__mapReady === true || window.__mapError", timeout=120_000)
    err = page.evaluate("window.__mapError || null")
    if err:
        raise RuntimeError(f"OpenFreeMap render failed: {err}")

    data_url = page.evaluate(
        """() => {
            const canvas = document.querySelector("#map canvas");
            return canvas ? canvas.toDataURL("image/png") : null;
        }"""
    )
    if not data_url:
        raise RuntimeError("OpenFreeMap render returned no canvas image.")

    _header, encoded = data_url.split(",", 1)
    return Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")


def render_openfreemap_basemap(
    bbox_wgs84: tuple[float, float, float, float],
    target_width: int,
    target_height: int,
    style_name: str,
    *,
    tile_max_px: int = 4096,
) -> Image.Image:
    """Stitch MapLibre renders from OpenFreeMap (roads/water/landuse, no labels)."""
    from playwright.sync_api import sync_playwright

    style_id = OPENFREEMAP_STYLE_IDS[style_name]
    style = fetch_nolabel_style(style_id)

    cols = max(1, math.ceil(target_width / tile_max_px))
    rows = max(1, math.ceil(target_height / tile_max_px))
    tile_count = cols * rows

    print(f"  OpenFreeMap style: {style_id} (symbol layers removed)")
    print(f"  Render grid: {cols} x {rows} = {tile_count} tiles (max {tile_max_px}px per tile)")

    mosaic = Image.new("RGB", (target_width, target_height), (242, 243, 240))

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page()

        for row in range(rows):
            for col in range(cols):
                left = round(col * target_width / cols)
                right = round((col + 1) * target_width / cols)
                top = round(row * target_height / rows)
                bottom = round((row + 1) * target_height / rows)
                tile_w = max(1, right - left)
                tile_h = max(1, bottom - top)
                sub_bbox = _sub_bounds(bbox_wgs84, col, row, cols, rows)
                index = row * cols + col + 1
                print(f"  Tile {index}/{tile_count}: {tile_w} x {tile_h} px")
                tile_img = _render_tile(page, style, tile_w, tile_h, sub_bbox)
                if tile_img.size != (tile_w, tile_h):
                    tile_img = tile_img.resize((tile_w, tile_h), Image.Resampling.LANCZOS)
                mosaic.paste(tile_img, (left, top))

        browser.close()

    return mosaic
