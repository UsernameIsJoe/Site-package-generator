#!/usr/bin/env python3
"""
Basemap export aligned to the GIS SVG output.

References: output/BaseMap_Creator.ipynb (Map Setting / basemap export cell)
Uses the same Carto tile styles and image adjustments as that notebook, but
reads projection + bounding box from the att/ geo metadata instead of manual
rectangle selection or scale-mode fitting.

Typical workflow:
  python export_to_illustrator.py --input ./att
  python create_basemap.py --geo-metadata ./output/att.geo.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import contextily as cx
import numpy as np
import xyzservices.providers as xyz
from PIL import Image, ImageEnhance
from pyproj import Geod, Transformer
from rasterio.transform import from_bounds

try:
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    import rasterio
except ImportError:
    rasterio = None


# ============================================================
# SETTINGS — EDIT ONLY THIS SECTION
# ============================================================

# -------------------------
# INPUT / OUTPUT
# -------------------------

# Geo metadata written by export_to_illustrator.py.
# If None, derived from ATT_FOLDER name in OUTPUT_DIR.
GEO_METADATA = None

# Source GIS folder (used only when geo metadata must be built).
ATT_FOLDER = Path(__file__).resolve().parent / "att"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

OUTPUT_NAME = "basemap"


# -------------------------
# BASE STYLE
# -------------------------
# Same options as BaseMap_Creator.ipynb:
# "Voyager", "Positron", "DarkMatter"

BASE_STYLE = "Positron"


# -------------------------
# MAP RESOLUTION
# -------------------------
# Set AUTO_ZOOM = True to pick a zoom level from the SVG canvas size + bbox.
# Or set AUTO_ZOOM = False and choose ZOOM manually (same as notebook).

AUTO_ZOOM = True
ZOOM = 16
MAX_ZOOM = 18


# -------------------------
# IMAGE APPEARANCE
# -------------------------
# Same defaults as BaseMap_Creator.ipynb

SATURATION = 1.00
CONTRAST = 2.00
BRIGHTNESS = 1.00


# -------------------------
# EXPORT SETTINGS
# -------------------------

OUTPUT_DPI = 300


# -------------------------
# PREVIEW
# -------------------------

SHOW_PREVIEW = False
PREVIEW_SIZE = 12


# ============================================================
# STYLE SOURCES (from BaseMap_Creator.ipynb)
# ============================================================

STYLE_SOURCES = {
    "Voyager": xyz.CartoDB.VoyagerNoLabels,
    "Positron": xyz.CartoDB.PositronNoLabels,
    "DarkMatter": xyz.CartoDB.DarkMatterNoLabels,
}


def estimate_zoom(
    bbox_wgs84: tuple[float, float, float, float],
    target_width: int,
    target_height: int,
    max_zoom: int,
) -> int:
    west, south, east, north = bbox_wgs84
    center_lat = (south + north) / 2
    geod = Geod(ellps="WGS84")

    _, _, width_m = geod.inv(west, center_lat, east, center_lat)
    _, _, height_m = geod.inv((west + east) / 2, south, (west + east) / 2, north)
    width_m = abs(width_m)
    height_m = abs(height_m)

    meters_per_pixel_needed = max(width_m / target_width, height_m / target_height)
    meters_per_pixel_equator = 156543.03392804097 * math.cos(math.radians(center_lat))

    for candidate in range(1, max_zoom + 1):
        meters_per_pixel = meters_per_pixel_equator / (2**candidate)
        if meters_per_pixel <= meters_per_pixel_needed:
            return candidate
    return max_zoom


def choose_zoom(
    bbox_wgs84: tuple[float, float, float, float],
    target_width: int,
    target_height: int,
) -> int:
    if AUTO_ZOOM:
        return estimate_zoom(bbox_wgs84, target_width, target_height, MAX_ZOOM)
    return ZOOM


def resolve_geo_metadata_path(explicit: Path | None, att_folder: Path, output_dir: Path) -> Path:
    if explicit is not None:
        return explicit.resolve()
    return (output_dir / f"{att_folder.name}.geo.json").resolve()


def load_or_build_geo_metadata(path: Path, att_folder: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    print(f"Geo metadata not found at {path}; building from {att_folder}...")
    from export_to_illustrator import build_geo_metadata, prepare_export

    ctx = prepare_export(att_folder.resolve(), max_dimension=8000.0, layer_order="size-desc")
    metadata = build_geo_metadata(
        att_folder.resolve(),
        path.with_suffix(".svg"),
        ctx,
        fmt="svg",
        max_dimension=8000.0,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def source_bounds_to_wgs84(metadata: dict) -> tuple[float, float, float, float]:
    export_bounds = metadata["bounds"]["export"]
    source_epsg = metadata["projection"]["epsg"]
    if source_epsg is None:
        raise ValueError("Geo metadata is missing a source EPSG code.")

    transformer = Transformer.from_crs(f"EPSG:{source_epsg}", "EPSG:4326", always_xy=True)
    minx = export_bounds["minx"]
    miny = export_bounds["miny"]
    maxx = export_bounds["maxx"]
    maxy = export_bounds["maxy"]

    corner_pairs = ((minx, miny), (minx, maxy), (maxx, miny), (maxx, maxy))
    lons: list[float] = []
    lats: list[float] = []
    for x, y in corner_pairs:
        lon, lat = transformer.transform(x, y)
        lons.append(lon)
        lats.append(lat)

    return min(lons), min(lats), max(lons), max(lats)


def crop_to_bbox(img: np.ndarray, tile_extent, bbox_wgs84: tuple[float, float, float, float]) -> np.ndarray:
    west, south, east, north = bbox_wgs84
    tile_min_x, tile_max_x, tile_min_y, tile_max_y = tile_extent

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    bbox_min_x, bbox_min_y = transformer.transform(west, south)
    bbox_max_x, bbox_max_y = transformer.transform(east, north)

    height, width = img.shape[:2]
    x_resolution = (tile_max_x - tile_min_x) / width
    y_resolution = (tile_max_y - tile_min_y) / height

    left = int(round((bbox_min_x - tile_min_x) / x_resolution))
    right = int(round((bbox_max_x - tile_min_x) / x_resolution))
    top = int(round((tile_max_y - bbox_max_y) / y_resolution))
    bottom = int(round((tile_max_y - bbox_min_y) / y_resolution))

    left = max(0, left)
    right = min(width, right)
    top = max(0, top)
    bottom = min(height, bottom)

    cropped = img[top:bottom, left:right]
    if cropped.size == 0:
        raise RuntimeError("Crop returned an empty image. Try increasing ZOOM or checking geo metadata bounds.")
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


def save_bbox_sidecar(path: Path, metadata: dict, bbox_wgs84: tuple[float, float, float, float]) -> None:
    west, south, east, north = bbox_wgs84
    export_bounds = metadata["bounds"]["export"]
    source_epsg = metadata["projection"]["epsg"]

    lines = [
        f"source_crs=EPSG:{source_epsg}",
        f"source_export_minx={export_bounds['minx']}",
        f"source_export_miny={export_bounds['miny']}",
        f"source_export_maxx={export_bounds['maxx']}",
        f"source_export_maxy={export_bounds['maxy']}",
        "download_crs=EPSG:4326",
        f"west={west}",
        f"south={south}",
        f"east={east}",
        f"north={north}",
        f"canvas_width_pt={metadata['canvas']['width_pt']}",
        f"canvas_height_pt={metadata['canvas']['height_pt']}",
        f"scale_pt_per_map_unit={metadata['canvas']['scale_pt_per_map_unit']}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_geotiff(path: Path, image_array: np.ndarray, bbox_wgs84: tuple[float, float, float, float]) -> None:
    if rasterio is None:
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


def create_basemap(
    geo_metadata_path: Path,
    att_folder: Path,
    output_dir: Path,
    output_name: str = OUTPUT_NAME,
) -> dict:
    if BASE_STYLE not in STYLE_SOURCES:
        raise ValueError(f"Invalid BASE_STYLE: {BASE_STYLE}. Choose from: {list(STYLE_SOURCES)}")

    metadata = load_or_build_geo_metadata(geo_metadata_path, att_folder)
    bbox_wgs84 = source_bounds_to_wgs84(metadata)
    west, south, east, north = bbox_wgs84

    target_width = int(round(metadata["canvas"]["width_pt"]))
    target_height = int(round(metadata["canvas"]["height_pt"]))
    zoom = choose_zoom(bbox_wgs84, target_width, target_height)

    print("Downloading basemap...")
    print()
    print("Settings:")
    print(f"  Base style:   {BASE_STYLE}")
    print(f"  Zoom:         {zoom}" + (" (auto)" if AUTO_ZOOM else ""))
    print(f"  Saturation:   {SATURATION:.2f}")
    print(f"  Contrast:     {CONTRAST:.2f}")
    print(f"  Brightness:   {BRIGHTNESS:.2f}")
    print(f"  Output DPI:   {OUTPUT_DPI}")
    print()
    print("Source projection:")
    print(f"  EPSG:{metadata['projection']['epsg']} ({metadata['projection']['name']})")
    print()
    print("Export bounds (source CRS):")
    export_bounds = metadata["bounds"]["export"]
    print(f"  minx={export_bounds['minx']}")
    print(f"  miny={export_bounds['miny']}")
    print(f"  maxx={export_bounds['maxx']}")
    print(f"  maxy={export_bounds['maxy']}")
    print()
    print("Download bounds (WGS84):")
    print(f"  West:  {west:.6f}")
    print(f"  South: {south:.6f}")
    print(f"  East:  {east:.6f}")
    print(f"  North: {north:.6f}")
    print()
    print(f"Target canvas (matches SVG): {target_width} x {target_height} px")
    print()

    img, tile_extent = cx.bounds2img(
        west,
        south,
        east,
        north,
        zoom=zoom,
        source=STYLE_SOURCES[BASE_STYLE],
        ll=True,
        n_connections=4,
    )

    cropped = crop_to_bbox(img, tile_extent, bbox_wgs84)
    map_image = apply_image_adjustments(array_to_rgb_image(cropped))
    map_image = map_image.resize((target_width, target_height), Image.Resampling.LANCZOS)

    final_img = np.array(map_image).astype(np.uint8)
    crop_height, crop_width = final_img.shape[:2]

    output_dir.mkdir(parents=True, exist_ok=True)
    output_png = output_dir / f"{output_name}.png"
    output_tif = output_dir / f"{output_name}.tif"
    output_bbox = output_dir / f"{output_name}_bbox.txt"

    map_image.save(output_png, dpi=(OUTPUT_DPI, OUTPUT_DPI))
    save_geotiff(output_tif, final_img, bbox_wgs84)
    save_bbox_sidecar(output_bbox, metadata, bbox_wgs84)

    if SHOW_PREVIEW and HAS_MPL:
        plt.figure(figsize=(PREVIEW_SIZE, PREVIEW_SIZE))
        plt.imshow(final_img)
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.show()
    elif SHOW_PREVIEW:
        print("  Preview skipped (matplotlib not installed).")

    print()
    print("Export complete.")
    print()
    print(f"  Pixel dimensions: {crop_width} x {crop_height}")
    print(f"  Print size at {OUTPUT_DPI} DPI: {crop_width / OUTPUT_DPI:.2f} x {crop_height / OUTPUT_DPI:.2f} inches")
    print()
    print(f"  PNG:  {output_png}")
    print(f"  TIF:  {output_tif}")
    print(f"  BBox: {output_bbox}")
    print()
    print("  Place this PNG under the SVG in Illustrator; dimensions match the vector export.")
    print("  Attribution: © OpenStreetMap contributors © CARTO")

    return {
        "png": output_png,
        "tif": output_tif,
        "bbox": output_bbox,
        "width": crop_width,
        "height": crop_height,
        "metadata": metadata,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a basemap aligned to att/ geo metadata and SVG canvas size."
    )
    parser.add_argument(
        "--geo-metadata",
        type=Path,
        default=GEO_METADATA,
        help="Path to .geo.json from export_to_illustrator.py",
    )
    parser.add_argument(
        "--att-folder",
        type=Path,
        default=ATT_FOLDER,
        help="GIS source folder (used to build geo metadata if missing)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for basemap outputs",
    )
    parser.add_argument(
        "--output-name",
        default=OUTPUT_NAME,
        help="Base filename for exported basemap files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    geo_metadata_path = resolve_geo_metadata_path(args.geo_metadata, args.att_folder, args.output_dir)

    try:
        create_basemap(
            geo_metadata_path=geo_metadata_path,
            att_folder=args.att_folder,
            output_dir=args.output_dir,
            output_name=args.output_name,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
