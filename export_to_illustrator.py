#!/usr/bin/env python3
"""
GIS → Illustrator pipeline.

Scans a folder of subfolders (each containing shapefiles) and writes one SVG or PDF
with a named group per subfolder. Illustrator opens this as one layer with grouped
objects — one group per GIS dataset folder.

Usage:
  python export_to_illustrator.py --input ./att
  python export_to_illustrator.py --input ./projects --batch
  python export_to_illustrator.py --input ./att --output ./output/site_map.svg
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import geopandas as gpd
import pymupdf as fitz
from pyproj import CRS
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry

# Preferred attribute columns for <title> labels (first matches win).
LABEL_FIELD_CANDIDATES = (
    "name",
    "label",
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
        axis = crs_obj.axis_info
        if axis:
            units = axis[0].unit_name
    except Exception:
        units = None

    epsg = None
    try:
        epsg = crs_obj.to_epsg()
    except Exception:
        epsg = None

    return {
        "epsg": epsg,
        "name": crs_obj.name,
        "units": units,
        "wkt": crs_obj.to_wkt(),
    }


def layer_bounds(layer: LayerSource) -> dict[str, float]:
    minx, miny, maxx, maxy = layer.gdf.total_bounds
    return bounds_dict(minx, miny, maxx, maxy)


def build_geo_metadata(
    input_dir: Path,
    output_path: Path,
    ctx: ExportContext,
    fmt: str,
    max_dimension: float,
) -> dict[str, Any]:
    crs_values = {layer.gdf.crs for layer in ctx.layers if layer.gdf.crs is not None}
    if len(crs_values) > 1:
        print("  Warning: layers use mixed coordinate systems; metadata uses the first layer CRS.")

    reference_crs = next((layer.gdf.crs for layer in ctx.layers if layer.gdf.crs is not None), None)
    metadata_path = output_path.with_suffix(".geo.json")

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_folder": str(input_dir.resolve()),
        "export_file": str(output_path.resolve()),
        "metadata_file": str(metadata_path.resolve()),
        "format": fmt,
        "projection": describe_crs(reference_crs),
        "bounds": {
            "combined": bounds_dict(ctx.raw_minx, ctx.raw_miny, ctx.raw_maxx, ctx.raw_maxy),
            "export": bounds_dict(ctx.minx, ctx.miny, ctx.maxx, ctx.maxy),
            "padding_ratio": ctx.padding_ratio,
            "layers": [
                {
                    "name": layer.name,
                    "folder": str(layer.path.resolve()),
                    "feature_count": int(len(layer.gdf)),
                    **layer_bounds(layer),
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


def write_geo_metadata(metadata: dict[str, Any], output_path: Path) -> Path:
    metadata_path = output_path.with_suffix(".geo.json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata_path


def print_geo_summary(metadata: dict[str, Any]) -> None:
    projection = metadata["projection"]
    combined = metadata["bounds"]["combined"]
    units = projection.get("units") or "map units"
    epsg = projection.get("epsg")
    name = projection.get("name") or "Unknown CRS"

    print("  Geospatial metadata:")
    if epsg:
        print(f"    CRS: EPSG:{epsg} ({name})")
    else:
        print(f"    CRS: {name}")
    print(f"    Units: {units}")
    print(
        "    Combined bounds: "
        f"minx={combined['minx']}, miny={combined['miny']}, "
        f"maxx={combined['maxx']}, maxy={combined['maxy']}"
    )
    print(f"    Metadata file: {metadata['metadata_file']}")


def sanitize_id(value: str) -> str:
    cleaned = re.sub(r"[^\w\- ]+", "", value).strip()
    cleaned = re.sub(r"\s+", "_", cleaned)
    return cleaned or "layer"


def folder_has_layers(folder: Path) -> bool:
    if not folder.is_dir():
        return False
    return any(sub.is_dir() and list(sub.glob("*.shp")) for sub in folder.iterdir())


def discover_layer_folders(input_dir: Path) -> list[Path]:
    """Return subfolders that contain at least one shapefile."""
    folders = [
        sub
        for sub in sorted(input_dir.iterdir())
        if sub.is_dir() and list(sub.glob("*.shp"))
    ]
    if not folders:
        raise SystemExit(
            f"No shapefile subfolders found in {input_dir}\n"
            "Expected: input_dir/<layer_name>/*.shp"
        )
    return folders


def load_folder_layers(folder: Path) -> gpd.GeoDataFrame:
    shapefiles = sorted(folder.glob("*.shp"))
    if not shapefiles:
        raise ValueError(f"No shapefiles in {folder}")
    frames = [gpd.read_file(shp) for shp in shapefiles]
    merged = gpd.GeoDataFrame(gpd.pd.concat(frames, ignore_index=True), crs=frames[0].crs)
    if merged.crs is None:
        prj_files = sorted(folder.glob("*.prj"))
        if prj_files:
            merged = merged.set_crs(prj_files[0].read_text())
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


def sort_layers(layers: list[LayerSource], order: str) -> list[LayerSource]:
    if order == "alpha":
        return sorted(layers, key=lambda layer: layer.name.lower())
    if order == "size-asc":
        return sorted(layers, key=lambda layer: len(layer.gdf))
    # Default: largest first so big base layers sit below smaller overlays.
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


def ring_to_points(ring, scale: float, minx: float, maxy: float, close: bool = True) -> list[fitz.Point]:
    points = [fitz.Point(*transform_point(x, y, scale, minx, maxy)) for x, y in ring.coords]
    if close and points and points[0] != points[-1]:
        points.append(points[0])
    return points


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
) -> ExportContext:
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
    )


def add_polygon_to_shape(shape: fitz.Shape, polygon: Polygon, ctx: ExportContext) -> None:
    if len(list(polygon.exterior.coords)) < 4:
        return
    shape.draw_polyline(ring_to_points(polygon.exterior, ctx.scale, ctx.minx, ctx.maxy))
    for interior in polygon.interiors:
        if len(list(interior.coords)) < 4:
            continue
        shape.draw_polyline(ring_to_points(interior, ctx.scale, ctx.minx, ctx.maxy))


def prepare_export(input_dir: Path, max_dimension: float, layer_order: str) -> ExportContext:
    layers = sort_layers(discover_layers(input_dir), layer_order)
    return build_export_context(layers, max_dimension, padding_ratio=0.02)


def export_svg(
    ctx: ExportContext,
    output_path: Path,
    input_dir: Path,
    precision: int = 2,
) -> dict:
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
        "  <metadata id=\"geo-metadata\">",
        f"    <![CDATA[{json.dumps({'metadata_file': str(output_path.with_suffix('.geo.json').resolve())})}]]>",
        "  </metadata>",
    ]

    feature_count = 0
    for layer in ctx.layers:
        group_id = sanitize_id(layer.name)
        label_fields = guess_label_fields(layer.gdf)
        lines.append(f'  <g id="{group_id}" data-name="{html.escape(layer.name, quote=True)}">')

        for feature_index, row in layer.gdf.iterrows():
            for part_index, part in enumerate(geometry_parts(row.geometry)):
                if not isinstance(part, Polygon):
                    continue
                path_data = path_from_polygon(part, ctx.scale, ctx.minx, ctx.maxy, precision)
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


def export_pdf(
    ctx: ExportContext,
    output_path: Path,
    chunk_size: int = 1000,
) -> dict:
    doc = fitz.open()
    page = doc.new_page(width=ctx.width, height=ctx.height)

    feature_count = 0
    for index, layer in enumerate(ctx.layers, start=1):
        oc = doc.add_ocg(layer.name, on=True)
        shape = page.new_shape()
        layer_features = 0

        for _, row in layer.gdf.iterrows():
            for part in geometry_parts(row.geometry):
                if not isinstance(part, Polygon):
                    continue
                add_polygon_to_shape(shape, part, ctx)
                feature_count += 1
                layer_features += 1

                if layer_features % chunk_size == 0:
                    shape.finish(color=(0, 0, 0), width=0.25, closePath=False, oc=oc)
                    shape.commit()
                    shape = page.new_shape()

        if layer_features % chunk_size != 0:
            shape.finish(color=(0, 0, 0), width=0.25, closePath=False, oc=oc)
            shape.commit()

        print(f"  Group {index}/{len(ctx.layers)}: {layer.name} ({layer_features:,} features)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(output_path)
    doc.close()

    return {
        "output": output_path,
        "groups": len(ctx.layers),
        "features": feature_count,
        "width": ctx.width,
        "height": ctx.height,
    }


def default_output_path(input_dir: Path, output_dir: Path, fmt: str) -> Path:
    return output_dir / f"{input_dir.name}.{fmt}"


def run_pipeline(
    input_dir: Path,
    output_path: Path,
    fmt: str,
    max_dimension: float,
    layer_order: str,
) -> dict:
    input_dir = input_dir.resolve()
    print(f"Processing {input_dir}")

    ctx = prepare_export(input_dir, max_dimension, layer_order)
    if fmt == "svg":
        result = export_svg(ctx, output_path, input_dir)
    else:
        result = export_pdf(ctx, output_path)

    metadata = build_geo_metadata(input_dir, output_path, ctx, fmt, max_dimension)
    metadata_path = write_geo_metadata(metadata, output_path)
    result["metadata"] = metadata
    result["metadata_path"] = metadata_path

    print(f"  Wrote {result['output']}")
    print(f"  Groups: {result['groups']}")
    print(f"  Features: {result['features']:,}")
    print(f"  Canvas: {result['width']:.1f} x {result['height']:.1f} pt")
    print_geo_summary(metadata)
    return result


def run_batch(
    input_root: Path,
    output_dir: Path,
    fmt: str,
    max_dimension: float,
    layer_order: str,
) -> list[dict]:
    projects = [
        child
        for child in sorted(input_root.iterdir())
        if child.is_dir() and folder_has_layers(child)
    ]
    if not projects:
        raise SystemExit(
            f"No project folders found in {input_root}\n"
            "Expected: input_root/<project>/<layer_name>/*.shp"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    for project in projects:
        output_path = default_output_path(project, output_dir, fmt)
        results.append(
            run_pipeline(project, output_path, fmt, max_dimension, layer_order)
        )
        print()
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert GIS subfolder shapefiles into one grouped Illustrator file."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parent / "att",
        help="Folder containing layer subfolders with shapefiles (default: ./att)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file path (default: ./output/<input_folder_name>.svg)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "output",
        help="Output directory for batch mode (default: ./output)",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process every project subfolder under --input",
    )
    parser.add_argument(
        "--format",
        choices=("svg", "pdf"),
        default="svg",
        help="Output format (default: svg — best Illustrator group support)",
    )
    parser.add_argument(
        "--max-dimension",
        type=float,
        default=8000.0,
        help="Longest canvas side in points (default: 8000)",
    )
    parser.add_argument(
        "--layer-order",
        choices=("size-desc", "size-asc", "alpha"),
        default="size-desc",
        help="Group draw order (default: size-desc, largest/bottom first)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        if args.batch:
            run_batch(
                args.input,
                args.output_dir,
                args.format,
                args.max_dimension,
                args.layer_order,
            )
        else:
            output = args.output
            if output is None:
                output = default_output_path(args.input, args.output_dir, args.format)
            run_pipeline(
                args.input,
                output,
                args.format,
                args.max_dimension,
                args.layer_order,
            )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
