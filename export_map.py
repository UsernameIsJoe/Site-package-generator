#!/usr/bin/env python3
"""
GIS → Illustrator export (layers + basemap in one run).

How to use in VS Code:
  1. Edit INPUT_FOLDER below (path to your GIS mother folder).
  2. Optionally tune basemap style / image settings.
  3. Open this file and press Run (▶) — no terminal commands needed.

Expected input layout:
  INPUT_FOLDER/
    Layer A/
      *.shp
    Layer B/
      *.shp

Output (named after the input folder):
  output/<INPUT_FOLDER_NAME>/
    layers.svg
    layers.geo.json
    basemap.png
    basemap.tif
    basemap_bbox.txt
    site_model.3dm          (optional Rhino 3D site model)
    site_model_readme.txt
"""

from __future__ import annotations

import html
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import contextily as cx
import geopandas as gpd
import numpy as np
import xyzservices
import xyzservices.providers as xyz
from PIL import Image, ImageEnhance
from pyproj import CRS, Geod, Transformer
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry

try:
    import rasterio
    from rasterio.transform import from_bounds
except ImportError:
    rasterio = None
    from_bounds = None


# ============================================================
# SETTINGS — EDIT THIS SECTION, THEN PRESS RUN
# ============================================================

# Path to the GIS mother folder (contains subfolders with shapefiles).
# Examples:
#   Path(r"C:\Users\tu\Desktop\Mapping\att")
#   Path(r"D:\GIS\my_site")
INPUT_FOLDER = Path(r"C:\Users\tu\Desktop\Mapping\Ward")

# Root folder for all exports. Each run creates a subfolder named after INPUT_FOLDER.
OUTPUT_ROOT = Path(__file__).resolve().parent / "output"

# Longest SVG/basemap side in points/pixels.
MAX_DIMENSION = 8000.0

# Layer draw order: "size-desc" (largest bottom), "size-asc", or "alpha"
LAYER_ORDER = "size-desc"

# Create basemap PNG aligned to the SVG canvas.
CREATE_BASEMAP = True

# Create Rhino 3D site model (.3dm) from OpenStreetMap for the same area.
# Independent of Illustrator 2D — only shares the geographic bounds.
CREATE_SITE_MODEL = True

# -------------------------
# BASEMAP STYLE
# -------------------------
# Recommended (free, no API key): Positron / Voyager / DarkMatter via OpenFreeMap + MapLibre.
#   Full OSM cartography (roads, water, buildings) with labels stripped.
# Also: "Bright", "Liberty", "Fiord" | "Imagery" (Esri satellite) | "Blank" | "Hillshade" (terrain only).
# Optional: set BASEMAP_RASTER_SOURCE = "carto" and a free CARTO key for faster raster tiles.
BASE_STYLE = "Positron"
BASEMAP_RASTER_SOURCE = "openfreemap"  # openfreemap | carto

# OpenFreeMap tile render size (px). Larger = fewer Playwright tiles, more RAM per tile.
BASEMAP_OPENFREEMAP_TILE_MAX_PX = 8192

# SRTM sample grid for Hillshade (higher = smoother relief, slower download).
BASEMAP_TERRAIN_GRID = 80

# Hillshade look (Positron-like neutral gray).
HILLSHADE_BASE_RGB = (242, 242, 240)
HILLSHADE_AZ = 315
HILLSHADE_ALT = 45
HILLSHADE_VERT_EXAG = 2.0
HILLSHADE_STRENGTH = 0.20

# Optional CARTO key: https://carto.com/basemaps/apikey (also env CARTO_BASEMAP_API_KEY)
CARTO_BASEMAP_API_KEY = os.environ.get("CARTO_BASEMAP_API_KEY", "")

# Auto-pick tile zoom from canvas size, or set manually.
AUTO_ZOOM = True
ZOOM = 18
MAX_ZOOM = 20

# Image adjustments (same defaults as BaseMap_Creator.ipynb)
SATURATION = 1.00
CONTRAST = 2.00
BRIGHTNESS = 1.00

# Basemap print resolution. Output pixels = SVG canvas points * (OUTPUT_DPI / 72).
OUTPUT_DPI = 300

# Use CARTO @2x (retina) tiles — required for sharp 300 DPI exports.
BASEMAP_RETINA = True

# -------------------------
# 3D SITE MODEL (Rhino)
# -------------------------
# Expand GIS export bounds by this many meters on every side.
SITE_MODEL_PADDING_M = 10.0

# Warn when the model area is large (Overpass is slower above ~1 km²).
SITE_MODEL_MAX_AREA_KM2 = 1.0

# If area exceeds the limit:
#   True  -> shrink to a centered clip of SITE_MODEL_MAX_AREA_KM2
#   False -> keep the full folder site bounds (parcels/buildings/contours)
SITE_MODEL_AUTO_DOWNSIZE = False

# Simplify dense contour polylines for SVG (meters in the source CRS). 0 = none.
CONTOUR_SIMPLIFY_M = 2.0

# Include a terrain mesh from Open-Meteo elevation samples.
SITE_MODEL_INCLUDE_TERRAIN = True

# Contour line interval in meters (Cadmapper-style topography).
SITE_MODEL_CONTOUR_INTERVAL_M = 1.0

# SRTM sample grid resolution (higher = smoother terrain, slower download).
# For large sites (~20 km²) use 50–80; small sites can use 30–40.
SITE_MODEL_TERRAIN_GRID_SIZE = 60

# Building extrusion when OSM has no height / building:levels:
# randomly 7.0 or 8.5 m (see site_model_3d.UNKNOWN_BUILDING_HEIGHTS_M).
SITE_MODEL_DEFAULT_BUILDING_HEIGHT_M = 10.0  # unused for unknowns; kept for API compat
SITE_MODEL_METERS_PER_LEVEL = 3.0

# Roads / parks / water / parking export as draped centerline / outline curves.
SITE_MODEL_ROAD_SURFACES = False


# ============================================================
# INTERNAL — normally leave alone
# ============================================================

LABEL_FIELD_CANDIDATES = (
    "name",
    "label",
    "elev_ft",
    "elev_m",
    "site_addr",
    "address",
    "historic_n",
    "common_nam",
    "fld_zone",
    "zone_subty",
    "owner1",
    "loc_id",
    "map_par_id",
    "prop_id",
    "town_name",
    "city",
)

CARTO_NOLABELS_VARIANTS = {
    "Positron": "light_nolabels",
    "Voyager": "rastertiles/voyager_nolabels",
    "DarkMatter": "dark_nolabels",
}

BASEMAP_ATTRIBUTION = {
    "Hillshade": "Elevation: SRTM (OpenTopoData / Open-Meteo)",
    "Blank": "",
    "Imagery": "© Esri — Source: Esri, Maxar, Earthstar Geographics, USDA FSA, USGS, AeroGRID, IGN, IGP, and the GIS User Community",
    "Positron": "© OpenStreetMap contributors © OpenMapTiles © OpenFreeMap",
    "Voyager": "© OpenStreetMap contributors © OpenMapTiles © OpenFreeMap",
    "DarkMatter": "© OpenStreetMap contributors © OpenMapTiles © OpenFreeMap",
    "Bright": "© OpenStreetMap contributors © OpenMapTiles © OpenFreeMap",
    "Liberty": "© OpenStreetMap contributors © OpenMapTiles © OpenFreeMap",
    "Fiord": "© OpenStreetMap contributors © OpenMapTiles © OpenFreeMap",
}

OPENFREEMAP_BASEMAP_STYLES = frozenset({"Positron", "Voyager", "DarkMatter", "Bright", "Liberty", "Fiord"})
FREE_BASEMAP_STYLES = frozenset({"Hillshade", "Blank", "Imagery", *OPENFREEMAP_BASEMAP_STYLES})
TILE_BASEMAP_STYLES = frozenset({"Imagery", *CARTO_NOLABELS_VARIANTS.keys()})


def basemap_output_pixels(canvas_width_pt: float, canvas_height_pt: float, dpi: float) -> tuple[int, int]:
    """Match Illustrator: 1 pt = 1/72 inch -> dpi pixels per point."""
    scale = dpi / 72.0
    return int(round(canvas_width_pt * scale)), int(round(canvas_height_pt * scale))


def carto_api_key() -> str:
    return (CARTO_BASEMAP_API_KEY or os.environ.get("CARTO_BASEMAP_API_KEY", "")).strip()


def upscale_elevation_grid(elev: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    """Bicubic upscale of elevation rows/cols to exact output pixel size."""
    img = Image.fromarray(elev.astype(np.float32), mode="F")
    return np.array(img.resize((target_width, target_height), Image.Resampling.BICUBIC))


def render_blank_basemap(target_width: int, target_height: int) -> Image.Image:
    return Image.new("RGB", (target_width, target_height), HILLSHADE_BASE_RGB)


def render_hillshade_basemap(metadata: dict, target_width: int, target_height: int) -> Image.Image:
    """Positron-like gray hillshade from free SRTM — no tile API, no labels."""
    from matplotlib.colors import LightSource

    from site_model_3d import fetch_elevation_grid

    export_bounds = metadata["bounds"]["export"]
    epsg = metadata["projection"]["epsg"]
    if epsg is None:
        raise ValueError("Geo metadata is missing a source EPSG code.")

    print(f"  Fetching SRTM elevation ({BASEMAP_TERRAIN_GRID}x{BASEMAP_TERRAIN_GRID}, free)...")
    terrain = fetch_elevation_grid(export_bounds, epsg, grid_size=BASEMAP_TERRAIN_GRID)
    if terrain is None:
        print("  Elevation unavailable — using flat gray.")
        return render_blank_basemap(target_width, target_height)

    elev_hr = upscale_elevation_grid(terrain.elev, target_width, target_height)
    width_m = export_bounds["maxx"] - export_bounds["minx"]
    height_m = export_bounds["maxy"] - export_bounds["miny"]
    cellsize = max(width_m / target_width, height_m / target_height, 0.1)

    shade = LightSource(azdeg=HILLSHADE_AZ, altdeg=HILLSHADE_ALT).hillshade(
        elev_hr,
        vert_exag=HILLSHADE_VERT_EXAG,
        dx=cellsize,
        dy=cellsize,
    )
    base = float(HILLSHADE_BASE_RGB[0])
    channel = np.clip(base * (1.0 - HILLSHADE_STRENGTH + HILLSHADE_STRENGTH * shade), 0, 255).astype(
        np.uint8
    )
    rgb = np.stack([channel, channel, channel], axis=-1)
    return Image.fromarray(rgb, mode="RGB")


def tile_basemap_provider(style: str):
    if style == "Imagery":
        return xyz.Esri.WorldImagery
    if style in CARTO_NOLABELS_VARIANTS:
        api_key = carto_api_key()
        if not api_key:
            raise ValueError(f"Style {style} requires CARTO_BASEMAP_API_KEY.")
        return build_carto_basemap_provider(style, api_key, retina=BASEMAP_RETINA)
    raise ValueError(f"Unknown tile basemap style: {style}")


def download_tile_basemap(
    bbox_wgs84: tuple[float, float, float, float],
    target_width: int,
    target_height: int,
    provider,
    *,
    retina: bool = False,
) -> Image.Image:
    west, south, east, north = bbox_wgs84
    effective_max_zoom = min(MAX_ZOOM, int(getattr(provider, "max_zoom", 20) or 20))
    if AUTO_ZOOM:
        zoom = estimate_zoom(bbox_wgs84, target_width, target_height, effective_max_zoom, retina=retina)
    else:
        zoom = min(ZOOM, effective_max_zoom)

    print(f"  Retina: {retina}")
    print(f"  Zoom:  {zoom}" + (" (auto)" if AUTO_ZOOM else "") + f" (max {effective_max_zoom})")

    img, tile_extent = cx.bounds2img(
        west,
        south,
        east,
        north,
        zoom=zoom,
        source=provider,
        ll=True,
        n_connections=4,
    )
    cropped = crop_to_bbox(img, tile_extent, bbox_wgs84)
    native_h, native_w = cropped.shape[:2]
    print(f"  Native crop: {native_w} x {native_h} px")

    map_image = array_to_rgb_image(cropped)
    if native_w != target_width or native_h != target_height:
        if native_w < target_width * 0.9 or native_h < target_height * 0.9:
            print(
                "  WARNING: native resolution below target; output may look soft. "
                f"Try MAX_ZOOM={effective_max_zoom} or smaller OUTPUT_DPI."
            )
        map_image = map_image.resize((target_width, target_height), Image.Resampling.LANCZOS)
    return map_image


def build_carto_basemap_provider(style: str, api_key: str, *, retina: bool = True):
    """CARTO raster NoLabels tiles (no text/icons). Retina @2x for print resolution."""
    if style not in CARTO_NOLABELS_VARIANTS:
        raise ValueError(f"Style {style} is not a CARTO NoLabels style.")
    variant = CARTO_NOLABELS_VARIANTS[style]
    scale_suffix = "@2x" if retina else ""
    url = f"https://{{s}}.basemaps.cartocdn.com/{variant}/{{z}}/{{x}}/{{y}}{scale_suffix}.png?key={api_key}"
    return xyzservices.TileProvider(
        {
            "url": url,
            "subdomains": "abcd",
            "max_zoom": 20,
            "attribution": BASEMAP_ATTRIBUTION[style],
            "name": f"CartoDB.{style}NoLabels",
        }
    )

PADDING_RATIO = 0.02


@dataclass
class LayerSource:
    name: str
    path: Path
    gdf: gpd.GeoDataFrame


@dataclass
class ExportContext:
    layers: list[LayerSource]
    raw_minx: float
    raw_miny: float
    raw_maxx: float
    raw_maxy: float
    minx: float
    miny: float
    maxx: float
    maxy: float
    scale: float
    width: float
    height: float
    padding_ratio: float
    # Folder-derived site box (parcels/buildings/contours) — used by 3D + SVG reference.
    site_minx: float | None = None
    site_miny: float | None = None
    site_maxx: float | None = None
    site_maxy: float | None = None


def sanitize_id(value: str) -> str:
    cleaned = re.sub(r"[^\w\- ]+", "", value).strip()
    cleaned = re.sub(r"\s+", "_", cleaned)
    return cleaned or "layer"


def bounds_dict(minx: float, miny: float, maxx: float, maxy: float) -> dict[str, float]:
    return {
        "minx": round(minx, 6),
        "miny": round(miny, 6),
        "maxx": round(maxx, 6),
        "maxy": round(maxy, 6),
        "width": round(maxx - minx, 6),
        "height": round(maxy - miny, 6),
    }


def describe_crs(crs) -> dict[str, Any]:
    if crs is None:
        return {"epsg": None, "name": None, "units": None, "wkt": None}
    try:
        crs_obj = CRS.from_user_input(crs)
    except Exception:
        return {"epsg": None, "name": str(crs), "units": None, "wkt": str(crs)}

    units = None
    try:
        if crs_obj.axis_info:
            units = crs_obj.axis_info[0].unit_name
    except Exception:
        units = None

    try:
        epsg = crs_obj.to_epsg()
    except Exception:
        epsg = None

    return {"epsg": epsg, "name": crs_obj.name, "units": units, "wkt": crs_obj.to_wkt()}


def discover_layer_folders(input_dir: Path) -> list[Path]:
    folders = [
        sub
        for sub in sorted(input_dir.iterdir())
        if sub.is_dir() and list(sub.glob("*.shp"))
    ]
    if not folders:
        raise SystemExit(
            f"No shapefile subfolders found in {input_dir}\n"
            "Expected: INPUT_FOLDER/<layer_name>/*.shp"
        )
    return folders


def load_folder_layers(folder: Path) -> gpd.GeoDataFrame:
    shapefiles = sorted(folder.glob("*.shp"))
    frames = [gpd.read_file(shp) for shp in shapefiles]
    merged = gpd.GeoDataFrame(gpd.pd.concat(frames, ignore_index=True), crs=frames[0].crs)
    if merged.crs is None:
        prj_files = sorted(folder.glob("*.prj"))
        if prj_files:
            merged = merged.set_crs(prj_files[0].read_text())
    # Dense contour lines need simplification or the SVG becomes huge.
    if CONTOUR_SIMPLIFY_M > 0 and "contour" in folder.name.lower():
        print(
            f"  Simplifying contours in '{folder.name}' "
            f"(tolerance {CONTOUR_SIMPLIFY_M:g} map units)..."
        )
        merged = merged.copy()
        merged["geometry"] = merged.geometry.simplify(CONTOUR_SIMPLIFY_M, preserve_topology=True)
    return merged


def discover_layers(input_dir: Path) -> list[LayerSource]:
    layers: list[LayerSource] = []
    for folder in discover_layer_folders(input_dir):
        gdf = load_folder_layers(folder)
        if gdf.empty or gdf.geometry.isna().all():
            print(f"  Skipping empty layer: {folder.name}")
            continue
        layers.append(LayerSource(name=folder.name, path=folder, gdf=gdf))
    if not layers:
        raise SystemExit(f"No non-empty layers found in {input_dir}")
    return layers


def harmonize_crs(layers: list[LayerSource]) -> list[LayerSource]:
    """Reproject every layer into one shared CRS (required for a correct shared bbox)."""
    reference = next((layer.gdf.crs for layer in layers if layer.gdf.crs is not None), None)
    if reference is None:
        raise SystemExit("No CRS found on any layer; cannot build a map.")

    ref_epsg = None
    try:
        ref_epsg = reference.to_epsg()
    except Exception:
        ref_epsg = None

    out: list[LayerSource] = []
    for layer in layers:
        gdf = layer.gdf
        if gdf.crs is None:
            print(f"  Assuming CRS EPSG:{ref_epsg or reference} for '{layer.name}'")
            gdf = gdf.set_crs(reference)
        elif gdf.crs != reference:
            print(f"  Reprojecting '{layer.name}' -> {reference}")
            gdf = gdf.to_crs(reference)
        out.append(LayerSource(name=layer.name, path=layer.path, gdf=gdf))
    return out


def _layer_is_extent_anchor(name: str) -> bool:
    """Layers that define the real site footprint (not long roads / district overlays)."""
    hint = name.lower()
    anchors = ("parcel", "tax", "building", "structure", "contour")
    return any(token in hint for token in anchors)


def site_extent_bounds(layers: list[LayerSource]) -> tuple[float, float, float, float]:
    """
    Site bbox from parcel/building/contour layers when present.
    Sprawl layers (historic districts, long road centerlines) must not set the canvas.
    """
    anchors = [layer for layer in layers if _layer_is_extent_anchor(layer.name)]
    use = anchors if anchors else layers
    names = ", ".join(layer.name for layer in use)
    print(f"  Site extent from: {names}")

    minx = min(float(layer.gdf.total_bounds[0]) for layer in use)
    miny = min(float(layer.gdf.total_bounds[1]) for layer in use)
    maxx = max(float(layer.gdf.total_bounds[2]) for layer in use)
    maxy = max(float(layer.gdf.total_bounds[3]) for layer in use)
    return minx, miny, maxx, maxy


def clip_layers_to_bounds(
    layers: list[LayerSource],
    bounds: tuple[float, float, float, float],
) -> list[LayerSource]:
    """Keep only geometry that intersects the site extent (same box for SVG + 3D)."""
    from shapely.geometry import box

    clip = box(*bounds)
    out: list[LayerSource] = []
    for layer in layers:
        gdf = layer.gdf
        hit = gdf[gdf.geometry.intersects(clip)].copy()
        if hit.empty:
            print(f"  Skipping '{layer.name}' (outside site extent)")
            continue
        hit["geometry"] = hit.geometry.intersection(clip)
        hit = hit[hit.geometry.notna() & ~hit.geometry.is_empty]
        if hit.empty:
            print(f"  Skipping '{layer.name}' (empty after clip)")
            continue
        out.append(LayerSource(name=layer.name, path=layer.path, gdf=hit))
    if not out:
        raise SystemExit("All layers were empty after clipping to site extent.")
    return out


def sort_layers(layers: list[LayerSource], order: str) -> list[LayerSource]:
    if order == "alpha":
        return sorted(layers, key=lambda layer: layer.name.lower())
    if order == "size-asc":
        return sorted(layers, key=lambda layer: len(layer.gdf))
    return sorted(layers, key=lambda layer: len(layer.gdf), reverse=True)


def guess_label_fields(gdf: gpd.GeoDataFrame, max_fields: int = 3) -> list[str]:
    columns = [col for col in gdf.columns if col != "geometry"]
    chosen = [col for col in LABEL_FIELD_CANDIDATES if col in columns][:max_fields]
    if chosen:
        return chosen
    string_cols = [
        col
        for col in columns
        if gdf[col].dtype == object or str(gdf[col].dtype).startswith("string")
    ]
    return string_cols[:max_fields]


def geometry_parts(geom: BaseGeometry | None) -> Iterable[BaseGeometry]:
    if geom is None or geom.is_empty:
        return
    if isinstance(geom, Polygon):
        yield geom
        return
    if isinstance(geom, MultiPolygon):
        for part in geom.geoms:
            if not part.is_empty:
                yield part
        return
    if isinstance(geom, LineString):
        yield geom
        return
    if isinstance(geom, MultiLineString):
        for part in geom.geoms:
            if not part.is_empty:
                yield part
        return
    yield geom


def build_label(row, fields: list[str]) -> str:
    parts: list[str] = []
    for field in fields:
        if field not in row.index:
            continue
        value = row[field]
        if value is None or (isinstance(value, float) and value != value):
            continue
        text = str(value).strip()
        if text:
            parts.append(text)
    return " | ".join(parts)


def transform_point(x: float, y: float, scale: float, minx: float, maxy: float) -> tuple[float, float]:
    return (x - minx) * scale, (maxy - y) * scale


def path_from_linestring(
    line: LineString,
    scale: float,
    minx: float,
    maxy: float,
    precision: int,
) -> str:
    coords = list(line.coords)
    if len(coords) < 2:
        return ""

    def point(x: float, y: float) -> str:
        sx, sy = transform_point(x, y, scale, minx, maxy)
        return f"{sx:.{precision}f},{sy:.{precision}f}"

    commands = [f"M {point(coords[0][0], coords[0][1])}"]
    for x, y in coords[1:]:
        commands.append(f"L {point(x, y)}")
    return " ".join(commands)


def path_from_polygon(
    polygon: Polygon,
    scale: float,
    minx: float,
    maxy: float,
    precision: int,
) -> str:
    def point(x: float, y: float) -> str:
        sx, sy = transform_point(x, y, scale, minx, maxy)
        return f"{sx:.{precision}f},{sy:.{precision}f}"

    commands: list[str] = []
    exterior = list(polygon.exterior.coords)
    if len(exterior) < 4:
        return ""

    commands.append(f"M {point(exterior[0][0], exterior[0][1])}")
    for x, y in exterior[1:]:
        commands.append(f"L {point(x, y)}")
    commands.append("Z")

    for interior in polygon.interiors:
        ring = list(interior.coords)
        if len(ring) < 4:
            continue
        commands.append(f"M {point(ring[0][0], ring[0][1])}")
        for x, y in ring[1:]:
            commands.append(f"L {point(x, y)}")
        commands.append("Z")

    return " ".join(commands)


def build_export_context(
    layers: list[LayerSource],
    max_dimension: float,
    padding_ratio: float,
    site_bounds: tuple[float, float, float, float] | None = None,
) -> ExportContext:
    if site_bounds is not None:
        minx, miny, maxx, maxy = site_bounds
        raw_minx, raw_miny, raw_maxx, raw_maxy = minx, miny, maxx, maxy
    else:
        combined = gpd.GeoSeries(
            [
                geom
                for layer in layers
                for geom in layer.gdf.geometry
                if geom is not None and not geom.is_empty
            ]
        )
        minx, miny, maxx, maxy = combined.total_bounds
        raw_minx, raw_miny, raw_maxx, raw_maxy = minx, miny, maxx, maxy

    width_map = maxx - minx
    height_map = maxy - miny
    if width_map == 0 or height_map == 0:
        raise SystemExit("Could not compute map bounds from input geometries.")

    pad_x = width_map * padding_ratio
    pad_y = height_map * padding_ratio
    minx -= pad_x
    miny -= pad_y
    maxx += pad_x
    maxy += pad_y
    width_map = maxx - minx
    height_map = maxy - miny
    scale = max_dimension / max(width_map, height_map)

    site_minx = site_miny = site_maxx = site_maxy = None
    if site_bounds is not None:
        site_minx, site_miny, site_maxx, site_maxy = site_bounds

    return ExportContext(
        layers=layers,
        raw_minx=raw_minx,
        raw_miny=raw_miny,
        raw_maxx=raw_maxx,
        raw_maxy=raw_maxy,
        minx=minx,
        miny=miny,
        maxx=maxx,
        maxy=maxy,
        scale=scale,
        width=width_map * scale,
        height=height_map * scale,
        padding_ratio=padding_ratio,
        site_minx=site_minx,
        site_miny=site_miny,
        site_maxx=site_maxx,
        site_maxy=site_maxy,
    )


def build_geo_metadata(
    input_dir: Path,
    output_path: Path,
    ctx: ExportContext,
    max_dimension: float,
) -> dict[str, Any]:
    crs_values = {layer.gdf.crs for layer in ctx.layers if layer.gdf.crs is not None}
    if len(crs_values) > 1:
        print("  Warning: layers use mixed CRS; metadata uses the first layer CRS.")

    reference_crs = next((layer.gdf.crs for layer in ctx.layers if layer.gdf.crs is not None), None)
    metadata_path = output_path.with_suffix(".geo.json")

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_folder": str(input_dir.resolve()),
        "export_file": str(output_path.resolve()),
        "metadata_file": str(metadata_path.resolve()),
        "format": "svg",
        "projection": describe_crs(reference_crs),
        "bounds": {
            "combined": bounds_dict(ctx.raw_minx, ctx.raw_miny, ctx.raw_maxx, ctx.raw_maxy),
            "export": bounds_dict(ctx.minx, ctx.miny, ctx.maxx, ctx.maxy),
            # Folder site box (parcels/buildings/contours). SVG/PNG/3D all use this extent.
            "site": (
                bounds_dict(ctx.site_minx, ctx.site_miny, ctx.site_maxx, ctx.site_maxy)
                if None not in (ctx.site_minx, ctx.site_miny, ctx.site_maxx, ctx.site_maxy)
                else bounds_dict(ctx.raw_minx, ctx.raw_miny, ctx.raw_maxx, ctx.raw_maxy)
            ),
            "padding_ratio": ctx.padding_ratio,
            "layers": [
                {
                    "name": layer.name,
                    "folder": str(layer.path.resolve()),
                    "feature_count": int(len(layer.gdf)),
                    **bounds_dict(*layer.gdf.total_bounds),
                }
                for layer in ctx.layers
            ],
        },
        "canvas": {
            "width_pt": round(ctx.width, 6),
            "height_pt": round(ctx.height, 6),
            "max_dimension_pt": max_dimension,
            "scale_pt_per_map_unit": round(ctx.scale, 10),
        },
        "coordinate_transform": {
            "description": "Convert source map coordinates to SVG/PDF canvas points.",
            "source_crs": describe_crs(reference_crs),
            "export_minx": round(ctx.minx, 6),
            "export_maxy": round(ctx.maxy, 6),
            "scale": round(ctx.scale, 10),
            "formulas": {
                "canvas_x": "(source_x - export_minx) * scale",
                "canvas_y": "(export_maxy - source_y) * scale",
                "source_x": "export_minx + (canvas_x / scale)",
                "source_y": "export_maxy - (canvas_y / scale)",
            },
        },
    }


def export_svg(ctx: ExportContext, output_path: Path, input_dir: Path, precision: int = 2) -> dict:
    lines: list[str] = [
        '<?xml version="1.0" encoding="utf-8"?>',
        f"<!-- Source: {html.escape(str(input_dir.resolve()))} -->",
        '<svg version="1.1"',
        '     xmlns="http://www.w3.org/2000/svg"',
        '     xmlns:xlink="http://www.w3.org/1999/xlink"',
        '     x="0px" y="0px"',
        f'     width="{ctx.width:.{precision}f}px" height="{ctx.height:.{precision}f}px"',
        f'     viewBox="0 0 {ctx.width:.{precision}f} {ctx.height:.{precision}f}"',
        f'     style="enable-background:new 0 0 {ctx.width:.{precision}f} {ctx.height:.{precision}f};"',
        '     xml:space="preserve">',
        "  <desc>One group per GIS subfolder. Open in Adobe Illustrator.</desc>",
    ]

    feature_count = 0
    for layer in ctx.layers:
        group_id = sanitize_id(layer.name)
        label_fields = guess_label_fields(layer.gdf)
        lines.append(f'  <g id="{group_id}" data-name="{html.escape(layer.name, quote=True)}">')

        for feature_index, row in layer.gdf.iterrows():
            for part_index, part in enumerate(geometry_parts(row.geometry)):
                if isinstance(part, Polygon):
                    path_data = path_from_polygon(part, ctx.scale, ctx.minx, ctx.maxy, precision)
                elif isinstance(part, LineString):
                    path_data = path_from_linestring(part, ctx.scale, ctx.minx, ctx.maxy, precision)
                else:
                    continue
                if not path_data:
                    continue

                label = build_label(row, label_fields)
                feature_id = f"{group_id}_{feature_index}_{part_index}"
                if label:
                    lines.append(f'    <g id="{feature_id}">')
                    lines.append(f"      <title>{html.escape(label)}</title>")
                    lines.append(
                        '      <path fill="none" stroke="#000000" stroke-width="0.25" '
                        f'd="{path_data}" />'
                    )
                    lines.append("    </g>")
                else:
                    lines.append(
                        '    <path fill="none" stroke="#000000" stroke-width="0.25" '
                        f'id="{feature_id}" d="{path_data}" />'
                    )
                feature_count += 1

        lines.append("  </g>")

    lines.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "output": output_path,
        "groups": len(ctx.layers),
        "features": feature_count,
        "width": ctx.width,
        "height": ctx.height,
    }


def source_bounds_to_wgs84(metadata: dict) -> tuple[float, float, float, float]:
    export_bounds = metadata["bounds"]["export"]
    source_epsg = metadata["projection"]["epsg"]
    if source_epsg is None:
        raise ValueError("Geo metadata is missing a source EPSG code.")

    transformer = Transformer.from_crs(f"EPSG:{source_epsg}", "EPSG:4326", always_xy=True)
    minx, miny, maxx, maxy = (
        export_bounds["minx"],
        export_bounds["miny"],
        export_bounds["maxx"],
        export_bounds["maxy"],
    )
    lons: list[float] = []
    lats: list[float] = []
    for x, y in ((minx, miny), (minx, maxy), (maxx, miny), (maxx, maxy)):
        lon, lat = transformer.transform(x, y)
        lons.append(lon)
        lats.append(lat)
    return min(lons), min(lats), max(lons), max(lats)


def estimate_zoom(
    bbox_wgs84: tuple[float, float, float, float],
    target_width: int,
    target_height: int,
    max_zoom: int,
    *,
    retina: bool = False,
) -> int:
    west, south, east, north = bbox_wgs84
    center_lat = (south + north) / 2
    geod = Geod(ellps="WGS84")
    _, _, width_m = geod.inv(west, center_lat, east, center_lat)
    _, _, height_m = geod.inv((west + east) / 2, south, (west + east) / 2, north)
    meters_per_pixel_needed = max(abs(width_m) / target_width, abs(height_m) / target_height)
    if retina:
        meters_per_pixel_needed /= 2.0
    meters_per_pixel_equator = 156543.03392804097 * math.cos(math.radians(center_lat))

    for candidate in range(max_zoom, 0, -1):
        if meters_per_pixel_equator / (2**candidate) <= meters_per_pixel_needed:
            return candidate
    return 1


def crop_to_bbox(img: np.ndarray, tile_extent, bbox_wgs84: tuple[float, float, float, float]) -> np.ndarray:
    west, south, east, north = bbox_wgs84
    tile_min_x, tile_max_x, tile_min_y, tile_max_y = tile_extent
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    bbox_min_x, bbox_min_y = transformer.transform(west, south)
    bbox_max_x, bbox_max_y = transformer.transform(east, north)

    height, width = img.shape[:2]
    x_resolution = (tile_max_x - tile_min_x) / width
    y_resolution = (tile_max_y - tile_min_y) / height

    left = max(0, int(round((bbox_min_x - tile_min_x) / x_resolution)))
    right = min(width, int(round((bbox_max_x - tile_min_x) / x_resolution)))
    top = max(0, int(round((tile_max_y - bbox_max_y) / y_resolution)))
    bottom = min(height, int(round((tile_max_y - bbox_min_y) / y_resolution)))

    cropped = img[top:bottom, left:right]
    if cropped.size == 0:
        raise RuntimeError("Basemap crop returned an empty image.")
    return cropped


def apply_image_adjustments(image: Image.Image) -> Image.Image:
    image = ImageEnhance.Color(image).enhance(SATURATION)
    image = ImageEnhance.Contrast(image).enhance(CONTRAST)
    image = ImageEnhance.Brightness(image).enhance(BRIGHTNESS)
    return image


def array_to_rgb_image(array: np.ndarray) -> Image.Image:
    cropped_uint8 = np.asarray(array).astype(np.uint8)
    if cropped_uint8.shape[2] == 4:
        return Image.fromarray(cropped_uint8, mode="RGBA").convert("RGB")
    return Image.fromarray(cropped_uint8).convert("RGB")


def save_geotiff(path: Path, image_array: np.ndarray, bbox_wgs84: tuple[float, float, float, float]) -> None:
    if rasterio is None or from_bounds is None:
        print("  Skipping GeoTIFF (rasterio not installed).")
        return

    west, south, east, north = bbox_wgs84
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    bbox_min_x, bbox_min_y = transformer.transform(west, south)
    bbox_max_x, bbox_max_y = transformer.transform(east, north)
    crop_height, crop_width = image_array.shape[:2]
    geo_transform = from_bounds(bbox_min_x, bbox_min_y, bbox_max_x, bbox_max_y, crop_width, crop_height)

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=crop_height,
        width=crop_width,
        count=3,
        dtype=np.uint8,
        crs="EPSG:3857",
        transform=geo_transform,
        compress="deflate",
    ) as dst:
        dst.write(image_array[:, :, 0], 1)
        dst.write(image_array[:, :, 1], 2)
        dst.write(image_array[:, :, 2], 3)


def export_basemap(metadata: dict, output_dir: Path) -> dict:
    valid_styles = sorted(FREE_BASEMAP_STYLES | set(CARTO_NOLABELS_VARIANTS))
    if BASE_STYLE not in valid_styles:
        raise ValueError(f"Invalid BASE_STYLE: {BASE_STYLE}. Choose from: {valid_styles}")

    bbox_wgs84 = source_bounds_to_wgs84(metadata)
    west, south, east, north = bbox_wgs84
    canvas_w = metadata["canvas"]["width_pt"]
    canvas_h = metadata["canvas"]["height_pt"]
    target_width, target_height = basemap_output_pixels(canvas_w, canvas_h, OUTPUT_DPI)

    print()
    print("Building basemap...")
    print(f"  Style: {BASE_STYLE}")
    print(f"  Target: {target_width} x {target_height} px @ {OUTPUT_DPI} DPI")

    use_carto = (
        BASEMAP_RASTER_SOURCE == "carto"
        and BASE_STYLE in CARTO_NOLABELS_VARIANTS
        and bool(carto_api_key())
    )

    if BASE_STYLE == "Hillshade":
        map_image = render_hillshade_basemap(metadata, target_width, target_height)
    elif BASE_STYLE == "Blank":
        print("  Flat fill (no network).")
        map_image = render_blank_basemap(target_width, target_height)
    elif use_carto:
        retina = BASEMAP_RETINA
        provider = tile_basemap_provider(BASE_STYLE)
        print("  Downloading CARTO raster tiles...")
        map_image = download_tile_basemap(
            bbox_wgs84,
            target_width,
            target_height,
            provider,
            retina=retina,
        )
        map_image = apply_image_adjustments(map_image)
    elif BASE_STYLE in OPENFREEMAP_BASEMAP_STYLES:
        from basemap_openfreemap import render_openfreemap_basemap

        print("  Rendering OpenFreeMap (MapLibre, no labels)...")
        map_image = render_openfreemap_basemap(
            bbox_wgs84,
            target_width,
            target_height,
            BASE_STYLE,
            tile_max_px=BASEMAP_OPENFREEMAP_TILE_MAX_PX,
        )
    elif BASE_STYLE == "Imagery":
        provider = tile_basemap_provider("Imagery")
        print("  Downloading Esri imagery tiles...")
        map_image = download_tile_basemap(
            bbox_wgs84,
            target_width,
            target_height,
            provider,
            retina=False,
        )
        map_image = apply_image_adjustments(map_image)
    elif BASE_STYLE in CARTO_NOLABELS_VARIANTS:
        raise SystemExit(
            f"BASE_STYLE={BASE_STYLE} with CARTO raster requires a free API key.\n"
            "  Either set BASEMAP_RASTER_SOURCE = 'openfreemap' (default, no key),\n"
            "  or get a free key: https://carto.com/basemaps/apikey"
        )
    else:
        raise ValueError(f"Unsupported BASE_STYLE: {BASE_STYLE}")

    final_img = np.array(map_image).astype(np.uint8)

    output_png = output_dir / "basemap.png"
    output_tif = output_dir / "basemap.tif"
    output_bbox = output_dir / "basemap_bbox.txt"

    # Allow large 300 DPI exports (PIL default limit blocks ~591 MP images).
    Image.MAX_IMAGE_PIXELS = max(Image.MAX_IMAGE_PIXELS, target_width * target_height * 2)
    map_image.save(output_png, dpi=(OUTPUT_DPI, OUTPUT_DPI))
    save_geotiff(output_tif, final_img, bbox_wgs84)

    export_bounds = metadata["bounds"]["export"]
    output_bbox.write_text(
        "\n".join(
            [
                f"source_crs=EPSG:{metadata['projection']['epsg']}",
                f"source_export_minx={export_bounds['minx']}",
                f"source_export_miny={export_bounds['miny']}",
                f"source_export_maxx={export_bounds['maxx']}",
                f"source_export_maxy={export_bounds['maxy']}",
                "download_crs=EPSG:4326",
                f"west={west}",
                f"south={south}",
                f"east={east}",
                f"north={north}",
                f"canvas_width_pt={canvas_w}",
                f"canvas_height_pt={canvas_h}",
                f"output_width_px={target_width}",
                f"output_height_px={target_height}",
                f"output_dpi={OUTPUT_DPI}",
                f"scale_pt_per_map_unit={metadata['canvas']['scale_pt_per_map_unit']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"  Wrote {output_png}")
    print(f"  Wrote {output_tif}")
    print(f"  Wrote {output_bbox}")
    return {"png": output_png, "tif": output_tif, "bbox": output_bbox}


def run() -> None:
    input_dir = Path(INPUT_FOLDER).expanduser().resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"INPUT_FOLDER does not exist or is not a folder:\n  {input_dir}")

    output_dir = (Path(OUTPUT_ROOT).expanduser().resolve() / input_dir.name)
    output_dir.mkdir(parents=True, exist_ok=True)

    svg_path = output_dir / "layers.svg"
    geo_path = output_dir / "layers.geo.json"

    print("=" * 60)
    print("GIS -> Illustrator export")
    print("=" * 60)
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print()

    total_steps = 1 + int(CREATE_BASEMAP) + int(CREATE_SITE_MODEL)
    step = 1

    print(f"{step}/{total_steps}  Building vector layers...")
    layers = discover_layers(input_dir)
    layers = harmonize_crs(layers)
    # Site box from parcels/buildings/contours — shared by SVG, PNG, and 3D.
    site_bounds = site_extent_bounds(layers)
    layers = clip_layers_to_bounds(layers, site_bounds)
    layers = sort_layers(layers, LAYER_ORDER)
    ctx = build_export_context(layers, MAX_DIMENSION, PADDING_RATIO, site_bounds=site_bounds)
    svg_result = export_svg(ctx, svg_path, input_dir)
    metadata = build_geo_metadata(input_dir, svg_path, ctx, MAX_DIMENSION)
    geo_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    metadata["metadata_file"] = str(geo_path.resolve())

    print(f"  Wrote {svg_path}")
    print(f"  Wrote {geo_path}")
    print(f"  Groups: {svg_result['groups']}")
    print(f"  Features: {svg_result['features']:,}")
    print(f"  Canvas: {svg_result['width']:.1f} x {svg_result['height']:.1f} pt")
    projection = metadata["projection"]
    if projection.get("epsg"):
        print(f"  CRS: EPSG:{projection['epsg']} ({projection.get('name')})")
    site = metadata["bounds"]["site"]
    export = metadata["bounds"]["export"]
    print(
        "  Site bounds (folders): "
        f"minx={site['minx']}, miny={site['miny']}, "
        f"maxx={site['maxx']}, maxy={site['maxy']} "
        f"({site['width']:.0f} x {site['height']:.0f} m)"
    )
    print(
        "  Export bounds (SVG/PNG +2% pad): "
        f"{export['width']:.0f} x {export['height']:.0f} m"
    )

    step = 2
    if CREATE_BASEMAP:
        print()
        print(f"{step}/{total_steps}  Building basemap...")
        export_basemap(metadata, output_dir)
        step += 1
    else:
        print()
        print("Basemap skipped (CREATE_BASEMAP = False)")

    if CREATE_SITE_MODEL:
        print()
        print(f"{step}/{total_steps}  Building Rhino 3D site model...")
        from site_model_3d import create_site_model

        create_site_model(
            metadata,
            output_dir,
            padding_m=SITE_MODEL_PADDING_M,
            max_area_km2=SITE_MODEL_MAX_AREA_KM2,
            auto_downsize=SITE_MODEL_AUTO_DOWNSIZE,
            include_terrain=SITE_MODEL_INCLUDE_TERRAIN,
            contour_interval_m=SITE_MODEL_CONTOUR_INTERVAL_M,
            terrain_grid_size=SITE_MODEL_TERRAIN_GRID_SIZE,
            default_building_height_m=SITE_MODEL_DEFAULT_BUILDING_HEIGHT_M,
            meters_per_level=SITE_MODEL_METERS_PER_LEVEL,
            road_surfaces=SITE_MODEL_ROAD_SURFACES,
            gis_folder=INPUT_FOLDER,
        )
    else:
        print()
        print("Site model skipped (CREATE_SITE_MODEL = False)")

    print()
    print("Done.")
    print(f"Open in Illustrator: {svg_path}")
    if CREATE_BASEMAP:
        print(f"Place underneath:    {output_dir / 'basemap.png'}")
        print(f"Attribution (basemap): {BASEMAP_ATTRIBUTION.get(BASE_STYLE, '')}")
    if CREATE_SITE_MODEL:
        print(f"Open in Rhino:       {output_dir / 'site_model.3dm'}")
        print("Attribution (model):  © OpenStreetMap contributors (ODbL)")


if __name__ == "__main__":
    try:
        run()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
