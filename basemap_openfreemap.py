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


def _rgb_css(rgb: tuple[int, int, int]) -> str:
    r, g, b = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
    return f"rgb({r}, {g}, {b})"


def _set_paint_color(paint: dict[str, Any], key: str, rgb: tuple[int, int, int]) -> None:
    """Replace a paint color even when the style uses a zoom expression."""
    css = _rgb_css(rgb)
    value = paint.get(key)
    if isinstance(value, list) and value and value[0] == "interpolate":
        # Flatten zoom ramps to a solid palette color.
        paint[key] = css
    else:
        paint[key] = css


def apply_basemap_palette(style: dict[str, Any], palette: dict[str, tuple[int, int, int]]) -> dict[str, Any]:
    """Recolor all MapLibre fill/line/background layers to a fixed palette."""
    for layer in style.get("layers") or []:
        layer_type = layer.get("type")
        lid = str(layer.get("id") or "").lower()
        src = str(layer.get("source-layer") or "").lower()
        paint = layer.setdefault("paint", {})
        blob = f"{lid} {src}"

        if layer_type == "background":
            _set_paint_color(paint, "background-color", palette["background"])
            continue

        if layer_type == "fill":
            if "water" in blob:
                _set_paint_color(paint, "fill-color", palette["water"])
            elif src == "park" or "park" in lid or "pitch" in blob or "cemetery" in blob:
                _set_paint_color(paint, "fill-color", palette["park"])
            elif "wood" in blob or "forest" in blob:
                _set_paint_color(paint, "fill-color", palette["wood"])
            elif "grass" in blob or "landcover" in blob:
                _set_paint_color(paint, "fill-color", palette["park"])
            elif "sand" in blob:
                _set_paint_color(paint, "fill-color", palette["sand"])
            elif "building" in blob:
                _set_paint_color(paint, "fill-color", palette["building"])
            elif "residential" in blob or "suburb" in blob:
                _set_paint_color(paint, "fill-color", palette["residential"])
            elif "industrial" in blob or "commercial" in blob or "railway" in blob:
                _set_paint_color(paint, "fill-color", palette["land"])
            elif "landuse" in blob or "landcover" in src:
                _set_paint_color(paint, "fill-color", palette["land"])
            else:
                _set_paint_color(paint, "fill-color", palette["land"])
            continue

        if layer_type == "line":
            if "water" in blob:
                _set_paint_color(paint, "line-color", palette["water"])
            elif "rail" in blob:
                _set_paint_color(paint, "line-color", palette["rail"])
            elif "boundary" in blob or "admin" in blob:
                _set_paint_color(paint, "line-color", palette["boundary"])
            elif "path" in blob or "pier" in blob or "aeroway" in blob:
                _set_paint_color(paint, "line-color", palette["road_path"])
            elif "casing" in blob:
                # Soft edge around roads
                _set_paint_color(paint, "line-color", palette["road_minor"])
            elif any(k in blob for k in ("motorway", "major", "primary", "trunk", "secondary")):
                _set_paint_color(paint, "line-color", palette["road_major"])
            elif "minor" in blob or "service" in blob or "transportation" in src:
                _set_paint_color(paint, "line-color", palette["road_minor"])
            elif "building" in blob:
                _set_paint_color(paint, "line-color", palette["building"])
            else:
                _set_paint_color(paint, "line-color", palette["road_minor"])
            continue
    return style


def reorder_woods_under_features(style: dict[str, Any]) -> dict[str, Any]:
    """Draw woods above ground; water under roads/buildings/parks."""
    background: list[dict[str, Any]] = []
    ground: list[dict[str, Any]] = []
    woods: list[dict[str, Any]] = []
    water: list[dict[str, Any]] = []
    rest: list[dict[str, Any]] = []

    for layer in style.get("layers") or []:
        layer_type = layer.get("type")
        lid = str(layer.get("id") or "").lower()
        src = str(layer.get("source-layer") or "").lower()
        blob = f"{lid} {src}"

        if layer_type == "background":
            background.append(layer)
        elif "water" in blob:
            water.append(layer)
        elif layer_type == "fill" and any(k in blob for k in ("wood", "forest")):
            woods.append(layer)
        elif layer_type == "fill" and (
            "landuse_residential" in lid
            or "landuse-residential" in lid
            or ("residential" in blob and "road" not in blob)
            or lid in {"land", "landcover_ice_shelf", "landcover_glacier"}
            or (
                src in {"landuse", "landcover"}
                and not any(k in blob for k in ("park", "wood", "forest", "grass", "sand", "pitch", "cemetery"))
            )
        ):
            ground.append(layer)
        else:
            rest.append(layer)

    # bottom → top: ground, woods, water, then parks/buildings/roads
    style["layers"] = background + ground + woods + water + rest
    return style


def fetch_nolabel_style(
    style_id: str,
    *,
    palette: dict[str, tuple[int, int, int]] | None = None,
    park_rgb: tuple[int, int, int] | None = None,
    wood_rgb: tuple[int, int, int] | None = None,
) -> dict[str, Any]:
    cache_key = f"{style_id}|{palette}|{park_rgb}|{wood_rgb}|woods-under|roads-over-water"
    if cache_key in _STYLE_CACHE:
        return _STYLE_CACHE[cache_key]

    url = f"https://tiles.openfreemap.org/styles/{style_id}"
    req = urllib.request.Request(url, headers={"User-Agent": "MappingSiteExport/1.0"})
    with urllib.request.urlopen(req, timeout=90) as resp:
        style = json.load(resp)
    style["layers"] = [layer for layer in style["layers"] if layer.get("type") != "symbol"]

    if palette:
        style = apply_basemap_palette(style, palette)
    else:
        # Legacy park/wood-only override path.
        legacy = {
            "background": (242, 243, 240),
            "land": (234, 234, 230),
            "residential": (234, 234, 230),
            "building": (230, 230, 225),
            "park": park_rgb or (186, 214, 150),
            "wood": wood_rgb or (150, 188, 138),
            "water": (194, 200, 202),
            "road_major": (255, 255, 255),
            "road_minor": (255, 255, 255),
            "road_path": (255, 255, 255),
            "rail": (200, 200, 200),
            "sand": (245, 238, 188),
            "boundary": (200, 200, 200),
        }
        style = apply_basemap_palette(style, legacy)

    style = reorder_woods_under_features(style)
    _STYLE_CACHE[cache_key] = style
    return style


def _sub_bounds(
    bbox_wgs84: tuple[float, float, float, float],
    col: int,
    row: int,
    cols: int,
    rows: int,
    *,
    overlap_frac: float = 0.0,
) -> tuple[float, float, float, float]:
    west, south, east, north = bbox_wgs84
    lon_step = (east - west) / cols
    lat_step = (north - south) / rows
    pad_lon = lon_step * overlap_frac
    pad_lat = lat_step * overlap_frac
    tile_west = west + col * lon_step - pad_lon
    tile_east = west + (col + 1) * lon_step + pad_lon
    tile_north = north - row * lat_step + pad_lat
    tile_south = north - (row + 1) * lat_step - pad_lat
    return (
        max(west, tile_west),
        max(south, tile_south),
        min(east, tile_east),
        min(north, tile_north),
    )


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
  pixelRatio: 1,
  attributionControl: false,
  fadeDuration: 0,
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
    img = Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
    if img.size != (width, height):
        img = img.resize((width, height), Image.Resampling.LANCZOS)
    return img


def render_openfreemap_basemap(
    bbox_wgs84: tuple[float, float, float, float],
    target_width: int,
    target_height: int,
    style_name: str,
    *,
    tile_max_px: int = 4096,
    palette: dict[str, tuple[int, int, int]] | None = None,
    park_rgb: tuple[int, int, int] | None = None,
    wood_rgb: tuple[int, int, int] | None = None,
) -> Image.Image:
    """Stitch MapLibre renders from OpenFreeMap (roads/water/landuse, no labels)."""
    from playwright.sync_api import sync_playwright

    style_id = OPENFREEMAP_STYLE_IDS[style_name]
    style = fetch_nolabel_style(
        style_id,
        palette=palette,
        park_rgb=park_rgb,
        wood_rgb=wood_rgb,
    )

    cols = max(1, math.ceil(target_width / tile_max_px))
    rows = max(1, math.ceil(target_height / tile_max_px))
    tile_count = cols * rows
    # Small geographic overlap reduces visible seams between Playwright tiles.
    overlap_frac = 0.04 if tile_count > 1 else 0.0

    print(f"  OpenFreeMap style: {style_id} (symbol layers removed)")
    print(f"  Render grid: {cols} x {rows} = {tile_count} tiles (max {tile_max_px}px per tile)")

    bg = (242, 243, 240)
    if palette and "background" in palette:
        bg = palette["background"]
    mosaic = Image.new("RGB", (target_width, target_height), bg)

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

                # Expand render size by overlap, then crop back to the mosaic slot.
                pad_x = int(round(tile_w * overlap_frac)) if overlap_frac else 0
                pad_y = int(round(tile_h * overlap_frac)) if overlap_frac else 0
                render_w = tile_w + (pad_x if col > 0 else 0) + (pad_x if col < cols - 1 else 0)
                render_h = tile_h + (pad_y if row > 0 else 0) + (pad_y if row < rows - 1 else 0)
                sub_bbox = _sub_bounds(
                    bbox_wgs84, col, row, cols, rows, overlap_frac=overlap_frac
                )
                index = row * cols + col + 1
                print(f"  Tile {index}/{tile_count}: {tile_w} x {tile_h} px")
                tile_img = _render_tile(page, style, render_w, render_h, sub_bbox)

                crop_left = pad_x if col > 0 else 0
                crop_top = pad_y if row > 0 else 0
                crop_right = crop_left + tile_w
                crop_bottom = crop_top + tile_h
                tile_img = tile_img.crop((crop_left, crop_top, crop_right, crop_bottom))
                if tile_img.size != (tile_w, tile_h):
                    tile_img = tile_img.resize((tile_w, tile_h), Image.Resampling.LANCZOS)
                mosaic.paste(tile_img, (left, top))

        browser.close()

    return mosaic
