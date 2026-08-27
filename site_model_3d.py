#!/usr/bin/env python3
"""
3D site model for Rhino from OpenStreetMap (+ optional terrain).

Uses the same geographic export bounds as export_map.py, expands them by a
padding distance, fetches OSM features via Overpass, and writes a layered .3dm.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import requests
import rhino3dm
from pyproj import Transformer
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon, mapping, shape

try:
    import osm2geojson
except ImportError:
    osm2geojson = None


# Default road carriageway widths (meters) when OSM width=* is missing.
# Similar spirit to Cadmapper class widths.
DEFAULT_ROAD_WIDTHS_M = {
    "motorway": 15.0,
    "motorway_link": 10.0,
    "trunk": 12.0,
    "trunk_link": 8.0,
    "primary": 10.0,
    "primary_link": 7.0,
    "secondary": 8.0,
    "secondary_link": 6.0,
    "tertiary": 7.0,
    "tertiary_link": 5.5,
    "residential": 6.0,
    "living_street": 5.0,
    "unclassified": 6.0,
    "service": 4.0,
    "road": 6.0,
    "pedestrian": 4.0,
    "footway": 2.0,
    "path": 2.0,
    "cycleway": 2.5,
    "steps": 2.0,
    "track": 3.0,
}

LAYER_COLORS = {
    "Buildings": (200, 200, 200, 255),
    "Roads_Major": (60, 60, 60, 255),
    "Roads_Minor": (90, 90, 90, 255),
    "Roads_Paths": (140, 140, 140, 255),
    "Railways": (120, 80, 40, 255),
    "Parks_OpenSpace": (120, 180, 120, 255),
    "Water": (100, 160, 210, 255),
    "Parking": (170, 170, 150, 255),
    "Landuse": (190, 210, 160, 255),
    "Terrain": (160, 140, 110, 255),
    "Contours": (130, 100, 70, 255),
}

MAJOR_HIGHWAYS = {
    "motorway",
    "motorway_link",
    "trunk",
    "trunk_link",
    "primary",
    "primary_link",
    "secondary",
    "secondary_link",
}
PATH_HIGHWAYS = {"footway", "path", "cycleway", "steps", "pedestrian", "track"}

OVERPASS_ENDPOINTS = (
    "https://lz4.overpass-api.de/api/interpreter",
    "https://overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass.osm.ch/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
)

LAYER_QUERIES = {
    "buildings": 'way["building"]({bbox}); relation["building"]({bbox});',
    "highways": 'way["highway"]({bbox});',
    "railways": 'way["railway"]({bbox});',
    "waterways": 'way["waterway"]({bbox});',
    "parks": (
        'way["leisure"~"park|garden|playground|nature_reserve|pitch|golf_course"]({bbox});'
        'relation["leisure"~"park|garden|playground|nature_reserve|pitch|golf_course"]({bbox});'
    ),
    "landuse": (
        'way["landuse"~"grass|meadow|forest|recreation_ground|village_green|cemetery|allotments|orchard|vineyard|greenfield"]({bbox});'
        'relation["landuse"~"grass|meadow|forest|recreation_ground|village_green|cemetery|allotments|orchard|vineyard|greenfield"]({bbox});'
    ),
    "natural": (
        'way["natural"~"water|wood|wetland|scrub|grassland"]({bbox});'
        'relation["natural"~"water|wood|wetland|scrub|grassland"]({bbox});'
    ),
    "parking": 'way["amenity"="parking"]({bbox}); relation["amenity"="parking"]({bbox});',
}

# One Overpass body per tile (much fewer requests than per-layer queries).
COMBINED_OSM_FRAGMENT = "\n  ".join(LAYER_QUERIES.values())


def parse_metric_tag(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    text = str(value).strip().lower().replace(",", ".")
    if not text:
        return default
    try:
        if text.endswith("m"):
            return float(text[:-1].strip())
        if "'" in text or "ft" in text or "feet" in text:
            # Rough feet support: 12' or 12 ft
            cleaned = text.replace("ft", "").replace("feet", "").replace("'", "").strip()
            return float(cleaned) * 0.3048
        return float(text.split()[0])
    except ValueError:
        return default


# When OSM/GIS has no height or building:levels, pick one of these at random.
UNKNOWN_BUILDING_HEIGHTS_M = (7.0, 8.5)

# Sink building solids this many meters below the terrain sample under the footprint.
BUILDING_SINK_M = 1.0

# Densify park/water/etc. outline curves so they follow terrain between vertices.
SURFACE_OUTLINE_MAX_SEGMENT_M = 5.0

# These layers used to be thin extrusion pads; export as draped outline curves instead.
SURFACE_OUTLINE_LAYERS = frozenset({"Water", "Parks_OpenSpace", "Parking", "Landuse"})


def building_height_m(props: dict, default_height: float, meters_per_level: float) -> float:
    height = parse_metric_tag(props.get("height"))
    if height is not None and height > 0:
        return height
    levels = parse_metric_tag(props.get("building:levels"))
    if levels is not None and levels > 0:
        return levels * meters_per_level
    # No height / levels: randomly 7.0 or 8.5 m (default_height unused).
    _ = default_height
    return float(random.choice(UNKNOWN_BUILDING_HEIGHTS_M))


def densify_linestring(line: LineString, max_segment_m: float = SURFACE_OUTLINE_MAX_SEGMENT_M) -> LineString:
    """Insert vertices so draped curves follow terrain between sparse corners."""
    if line.is_empty or line.length <= max_segment_m:
        return line
    count = max(2, int(math.ceil(line.length / max_segment_m)) + 1)
    coords = [line.interpolate(float(d)).coords[0] for d in np.linspace(0.0, line.length, count)]
    return LineString(coords)


def polygon_rings_as_lines(polygon: Polygon) -> list[LineString]:
    rings: list[LineString] = []
    if len(polygon.exterior.coords) >= 2:
        rings.append(LineString(polygon.exterior.coords))
    for interior in polygon.interiors:
        if len(interior.coords) >= 2:
            rings.append(LineString(interior.coords))
    return rings


def add_polygon_outline_on_terrain(
    model: rhino3dm.File3dm,
    polygon: Polygon,
    layer_index: int,
    terrain: TerrainGrid | None,
    origin_x: float,
    origin_y: float,
    *,
    max_segment_m: float = SURFACE_OUTLINE_MAX_SEGMENT_M,
    z_boost: float = 0.05,
) -> int:
    """Add densified outline rings as curves draped on terrain (no surface pads)."""
    added = 0
    for ring in polygon_rings_as_lines(polygon):
        dense = densify_linestring(ring, max_segment_m=max_segment_m)
        if add_line_on_terrain(model, dense, layer_index, terrain, origin_x, origin_y, z_boost=z_boost):
            added += 1
    return added


def road_width_m(props: dict) -> float:
    width = parse_metric_tag(props.get("width"))
    if width is not None and width > 0:
        return width
    highway = str(props.get("highway", "")).lower()
    lanes = parse_metric_tag(props.get("lanes"))
    if lanes is not None and lanes > 0:
        return max(3.0, lanes * 3.25)
    return DEFAULT_ROAD_WIDTHS_M.get(highway, 5.0)


def expand_bounds(bounds: dict[str, float], padding_m: float) -> dict[str, float]:
    return {
        "minx": bounds["minx"] - padding_m,
        "miny": bounds["miny"] - padding_m,
        "maxx": bounds["maxx"] + padding_m,
        "maxy": bounds["maxy"] + padding_m,
        "width": bounds["width"] + 2 * padding_m,
        "height": bounds["height"] + 2 * padding_m,
    }


def area_km2(bounds: dict[str, float]) -> float:
    return (bounds["maxx"] - bounds["minx"]) * (bounds["maxy"] - bounds["miny"]) / 1_000_000.0


def shrink_bounds_to_max_area(bounds: dict[str, float], max_area_km2: float) -> dict[str, float]:
    """Keep center and aspect ratio; shrink until area <= max_area_km2."""
    width = bounds["maxx"] - bounds["minx"]
    height = bounds["maxy"] - bounds["miny"]
    current = width * height / 1_000_000.0
    if current <= max_area_km2 or current <= 0:
        return bounds

    scale = math.sqrt(max_area_km2 / current)
    cx = (bounds["minx"] + bounds["maxx"]) / 2
    cy = (bounds["miny"] + bounds["maxy"]) / 2
    half_w = (width * scale) / 2
    half_h = (height * scale) / 2
    return {
        "minx": cx - half_w,
        "miny": cy - half_h,
        "maxx": cx + half_w,
        "maxy": cy + half_h,
        "width": half_w * 2,
        "height": half_h * 2,
    }


def source_bounds_to_wgs84(
    bounds: dict[str, float],
    source_epsg: int,
) -> tuple[float, float, float, float]:
    transformer = Transformer.from_crs(f"EPSG:{source_epsg}", "EPSG:4326", always_xy=True)
    corners = (
        (bounds["minx"], bounds["miny"]),
        (bounds["minx"], bounds["maxy"]),
        (bounds["maxx"], bounds["miny"]),
        (bounds["maxx"], bounds["maxy"]),
    )
    lons, lats = [], []
    for x, y in corners:
        lon, lat = transformer.transform(x, y)
        lons.append(lon)
        lats.append(lat)
    return min(lons), min(lats), max(lons), max(lats)


def build_overpass_query(west: float, south: float, east: float, north: float, fragment: str) -> str:
    # Overpass bbox order: south,west,north,east
    bbox = f"{south:.8f},{west:.8f},{north:.8f},{east:.8f}"
    return f"""
[out:json][timeout:90];
(
  {fragment.format(bbox=bbox)}
);
out geom;
""".strip()


def wgs84_tiles(
    west: float,
    south: float,
    east: float,
    north: float,
    max_span_deg: float = 0.03,
) -> list[tuple[float, float, float, float]]:
    """Split a WGS84 bbox into tiles so Overpass stays reliable on large sites."""
    tiles: list[tuple[float, float, float, float]] = []
    lat = south
    while lat < north - 1e-12:
        lat2 = min(lat + max_span_deg, north)
        lon = west
        while lon < east - 1e-12:
            lon2 = min(lon + max_span_deg, east)
            tiles.append((lon, lat, lon2, lat2))
            lon = lon2
        lat = lat2
    return tiles or [(west, south, east, north)]


def _post_overpass(query: str) -> dict:
    last_error: Exception | None = None
    for endpoint in OVERPASS_ENDPOINTS:
        try:
            print(f"    Overpass: {endpoint}")
            response = requests.post(
                endpoint,
                data={"data": query},
                timeout=180,
                headers={"User-Agent": "MappingSiteModel/1.0"},
            )
            if response.status_code == 429:
                print("    rate-limited (429); backing off...")
                time.sleep(8)
                last_error = RuntimeError("429 Too Many Requests")
                continue
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_error = exc
            print(f"    failed ({exc})")
            time.sleep(2)
    raise RuntimeError(f"Overpass query failed: {last_error}")


def fetch_osm_geojson(west: float, south: float, east: float, north: float) -> dict:
    if osm2geojson is None:
        raise RuntimeError("osm2geojson is required. Install with: pip install osm2geojson")

    tiles = wgs84_tiles(west, south, east, north)
    if len(tiles) > 1:
        print(f"  OSM fetch tiled into {len(tiles)} cells (1 query each).")

    all_features: list[dict] = []
    seen_ids: set[tuple] = set()
    consecutive_failures = 0

    for tile_i, (tw, ts, te, tn) in enumerate(tiles, start=1):
        if consecutive_failures >= 2:
            print("  Skipping remaining OSM tiles (Overpass unavailable / rate-limited).")
            break
        if len(tiles) > 1:
            print(f"  Tile {tile_i}/{len(tiles)}: W={tw:.5f} S={ts:.5f} E={te:.5f} N={tn:.5f}")
        print("  Fetching OSM (combined layers)...")
        query = build_overpass_query(tw, ts, te, tn, COMBINED_OSM_FRAGMENT)
        try:
            data = _post_overpass(query)
            geojson = osm2geojson.json2geojson(data)
            count = 0
            for feat in geojson.get("features") or []:
                props = feat.get("properties") or {}
                key = (
                    props.get("type") or props.get("id") or props.get("@id"),
                    props.get("id"),
                    str(feat.get("geometry")),
                )
                if key in seen_ids:
                    continue
                seen_ids.add(key)
                all_features.append(feat)
                count += 1
            print(f"    -> {count} new features")
            if count > 0:
                consecutive_failures = 0
            else:
                consecutive_failures += 1
        except Exception as exc:
            print(f"    WARNING: tile failed ({exc})")
            consecutive_failures += 1
        time.sleep(2.0)

    return {"type": "FeatureCollection", "features": all_features}


def _folder_hints(name: str) -> str:
    return name.lower().replace("_", " ").replace("-", " ")


def load_local_gis_features(
    gis_folder: Path | None,
    model_bounds: dict[str, float],
    source_epsg: int,
) -> list[dict]:
    """Load buildings/roads from local shapefiles when OSM is unavailable or sparse."""
    if gis_folder is None or not gis_folder.is_dir():
        return []

    try:
        import geopandas as gpd
    except ImportError:
        print("  WARNING: geopandas missing; cannot load local GIS fallback.")
        return []

    features: list[dict] = []
    clip_poly = Polygon(
        [
            (model_bounds["minx"], model_bounds["miny"]),
            (model_bounds["maxx"], model_bounds["miny"]),
            (model_bounds["maxx"], model_bounds["maxy"]),
            (model_bounds["minx"], model_bounds["maxy"]),
        ]
    )

    for sub in sorted(p for p in gis_folder.iterdir() if p.is_dir()):
        hint = _folder_hints(sub.name)
        role = None
        if "building" in hint or "structure" in hint:
            role = "building"
        elif "road" in hint or "street" in hint or "highway" in hint or "tiger" in hint:
            role = "highway"
        elif "openspace" in hint or "open space" in hint or "park" in hint:
            role = "park"
        if role is None:
            continue

        shps = list(sub.glob("*.shp"))
        if not shps:
            continue
        shp = shps[0]
        try:
            gdf = gpd.read_file(shp)
        except Exception as exc:
            print(f"  WARNING: could not read {shp.name}: {exc}")
            continue
        if gdf.crs is None:
            gdf = gdf.set_crs(epsg=source_epsg)
        else:
            gdf = gdf.to_crs(epsg=source_epsg)

        gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
        if gdf.empty:
            continue
        gdf = gdf[gdf.intersects(clip_poly)].copy()
        if gdf.empty:
            continue
        gdf["geometry"] = gdf.geometry.intersection(clip_poly)
        gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
        added = 0
        for _, row in gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            props = {"_source": "local_gis"}
            if role == "building":
                props["building"] = "yes"
            elif role == "highway":
                props["highway"] = "secondary"
            else:
                props["leisure"] = "park"
            features.append({"geometry": geom, "properties": props})
            added += 1
        print(f"  Local GIS fallback ({role}): {added} from '{sub.name}'")

    return features


def features_to_geojson(features: list[dict]) -> dict:
    out = []
    for feature in features:
        geom = feature.get("geometry")
        if geom is None:
            continue
        if hasattr(geom, "__geo_interface__"):
            geom = mapping(geom)
        out.append(
            {
                "type": "Feature",
                "geometry": geom,
                "properties": feature.get("properties") or {},
            }
        )
    return {"type": "FeatureCollection", "features": out}


def features_from_geojson_file(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"  WARNING: could not read OSM cache {path.name}: {exc}")
        return []
    features: list[dict] = []
    for feature in payload.get("features") or []:
        geom = feature.get("geometry")
        if not geom:
            continue
        try:
            shapely_geom = shape(geom)
        except Exception:
            continue
        if shapely_geom.is_empty:
            continue
        features.append({"geometry": shapely_geom, "properties": feature.get("properties") or {}})
    return features


def load_or_refresh_osm_features(
    features: list[dict],
    output_dir: Path,
    *,
    area_km2: float = 1.0,
    min_keep_ratio: float = 0.75,
) -> list[dict]:
    """Keep the richer of a fresh OSM pull vs last good cache (Overpass is flaky)."""
    cache_path = output_dir / "site_model_osm.geo.json"
    cached = features_from_geojson_file(cache_path)
    min_to_cache = max(300, int(area_km2 * 400))

    if cached and len(features) < max(1, int(len(cached) * min_keep_ratio)):
        print(
            f"  OSM fetch sparse ({len(features)} features); "
            f"reusing cache ({len(cached)} features from {cache_path.name})."
        )
        return cached
    if not features and cached:
        print(f"  OSM empty; reusing cache ({len(cached)} features).")
        return cached
    if len(features) >= min_to_cache and len(features) > len(cached):
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps(features_to_geojson(features), ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"  Cached OSM features ({len(features)}) -> {cache_path.name}")
        except Exception as exc:
            print(f"  WARNING: could not write OSM cache: {exc}")
    elif not cached and 0 < len(features) < min_to_cache:
        print(
            f"  WARNING: OSM returned only {len(features)} features "
            f"(expected ~{min_to_cache}+ for this area). Parks/water may be incomplete; "
            "re-run later when Overpass is healthier to build a cache."
        )
    return features


def project_geojson_to_crs(geojson: dict, source_epsg: int) -> list[dict]:
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{source_epsg}", always_xy=True)
    features: list[dict] = []
    for feature in geojson.get("features", []):
        geom = feature.get("geometry")
        if not geom:
            continue
        try:
            shapely_geom = shape(geom)
        except Exception:
            continue
        if shapely_geom.is_empty:
            continue

        def _project_coords(coords):
            if isinstance(coords[0], (float, int)):
                x, y = transformer.transform(coords[0], coords[1])
                return (x, y)
            return [_project_coords(c) for c in coords]

        projected = {
            "type": geom["type"],
            "coordinates": _project_coords(geom["coordinates"]),
        }
        features.append(
            {
                "geometry": shape(projected),
                "properties": feature.get("properties") or {},
            }
        )
    return features


def clip_features(features: list[dict], bounds: dict[str, float]) -> list[dict]:
    clip_poly = Polygon(
        [
            (bounds["minx"], bounds["miny"]),
            (bounds["maxx"], bounds["miny"]),
            (bounds["maxx"], bounds["maxy"]),
            (bounds["minx"], bounds["maxy"]),
        ]
    )
    clipped: list[dict] = []
    for feature in features:
        geom = feature["geometry"]
        try:
            part = geom.intersection(clip_poly)
        except Exception:
            continue
        if part.is_empty:
            continue
        clipped.append({"geometry": part, "properties": feature["properties"]})
    return clipped


def iter_polygons(geom) -> list[Polygon]:
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return [g for g in geom.geoms if isinstance(g, Polygon) and not g.is_empty]
    return []


def iter_lines(geom) -> list[LineString]:
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, LineString):
        return [geom]
    if isinstance(geom, MultiLineString):
        return [g for g in geom.geoms if isinstance(g, LineString) and not g.is_empty]
    return []


def feature_props(raw: dict | None) -> dict:
    """Flatten osm2geojson properties (tags may be nested)."""
    raw = raw or {}
    tags = raw.get("tags")
    if isinstance(tags, dict):
        merged = dict(tags)
        for key, value in raw.items():
            if key != "tags":
                merged.setdefault(key, value)
        return merged
    return dict(raw)


def classify_feature(props: dict) -> str | None:
    if props.get("building"):
        return "Buildings"
    highway = props.get("highway")
    if highway:
        if highway in MAJOR_HIGHWAYS:
            return "Roads_Major"
        if highway in PATH_HIGHWAYS:
            return "Roads_Paths"
        return "Roads_Minor"
    if props.get("railway"):
        return "Railways"
    leisure = props.get("leisure")
    if leisure in {"park", "garden", "playground", "nature_reserve", "pitch", "golf_course"}:
        return "Parks_OpenSpace"
    landuse = props.get("landuse")
    if landuse in {"grass", "meadow", "forest", "recreation_ground", "village_green", "cemetery", "allotments", "orchard", "vineyard", "greenfield"}:
        return "Parks_OpenSpace" if landuse != "forest" else "Landuse"
    natural = props.get("natural")
    if natural in {"water", "wetland"}:
        return "Water"
    if natural in {"wood", "scrub", "grassland"}:
        return "Landuse"
    if props.get("waterway"):
        return "Water"
    if props.get("amenity") == "parking":
        return "Parking"
    return None


def polyline_from_coords(coords, z: float = 0.0) -> rhino3dm.PolylineCurve | None:
    points = []
    for xy in coords:
        if len(xy) < 2:
            continue
        points.append(rhino3dm.Point3d(float(xy[0]), float(xy[1]), z))
    if len(points) < 2:
        return None
    return rhino3dm.PolylineCurve(points)


def ensure_layers(model: rhino3dm.File3dm) -> dict[str, int]:
    indices: dict[str, int] = {}
    for name, color in LAYER_COLORS.items():
        indices[name] = model.Layers.AddLayer(name, color)
    return indices


@dataclass
class TerrainGrid:
    """SRTM elevation grid in source CRS, with local XY origin at SW corner."""

    xs: np.ndarray  # world X (source CRS)
    ys: np.ndarray  # world Y (source CRS)
    elev: np.ndarray  # absolute elevation (m)
    origin_x: float
    origin_y: float
    z_base: float  # elev min — local Z is elev - z_base

    @property
    def local_elev(self) -> np.ndarray:
        return self.elev - self.z_base

    def to_local(self, x: float, y: float) -> tuple[float, float]:
        return x - self.origin_x, y - self.origin_y

    def sample_z(self, x: float, y: float) -> float:
        """Bilinear sample of local elevation at world XY."""
        xs, ys, elev = self.xs, self.ys, self.local_elev
        if x <= xs[0]:
            ix = 0.0
        elif x >= xs[-1]:
            ix = float(len(xs) - 1)
        else:
            ix = float(np.interp(x, xs, np.arange(len(xs))))

        if y <= ys[0]:
            iy = 0.0
        elif y >= ys[-1]:
            iy = float(len(ys) - 1)
        else:
            iy = float(np.interp(y, ys, np.arange(len(ys))))

        i0 = int(math.floor(iy))
        j0 = int(math.floor(ix))
        i1 = min(i0 + 1, elev.shape[0] - 1)
        j1 = min(j0 + 1, elev.shape[1] - 1)
        ty = iy - i0
        tx = ix - j0
        z00 = elev[i0, j0]
        z01 = elev[i0, j1]
        z10 = elev[i1, j0]
        z11 = elev[i1, j1]
        return float((1 - ty) * ((1 - tx) * z00 + tx * z01) + ty * ((1 - tx) * z10 + tx * z11))

    def sample_z_geom(self, geom) -> float:
        if geom is None or geom.is_empty:
            return 0.0
        try:
            pt = geom.representative_point()
            return self.sample_z(pt.x, pt.y)
        except Exception:
            return 0.0

    def sample_z_max(self, geom) -> float:
        """Highest local Z under a polygon footprint (keeps buildings above terrain)."""
        if geom is None or geom.is_empty:
            return 0.0
        zs: list[float] = []
        try:
            for poly in getattr(geom, "geoms", [geom]):
                if not hasattr(poly, "exterior"):
                    continue
                for x, y in list(poly.exterior.coords)[:: max(1, len(poly.exterior.coords) // 24)]:
                    zs.append(self.sample_z(float(x), float(y)))
                pt = poly.representative_point()
                zs.append(self.sample_z(pt.x, pt.y))
        except Exception:
            pass
        return max(zs) if zs else 0.0

    def build_mesh(self) -> rhino3dm.Mesh:
        mesh = rhino3dm.Mesh()
        rows, cols = self.local_elev.shape
        for i in range(rows):
            for j in range(cols):
                lx, ly = self.to_local(float(self.xs[j]), float(self.ys[i]))
                mesh.Vertices.Add(lx, ly, float(self.local_elev[i, j]))
        for i in range(rows - 1):
            for j in range(cols - 1):
                i0 = i * cols + j
                i1 = i0 + 1
                i2 = i0 + cols + 1
                i3 = i0 + cols
                mesh.Faces.AddFace(i0, i1, i2, i3)
        mesh.Normals.ComputeNormals()
        return mesh

    def build_contours(self, interval_m: float) -> list[tuple[float, list[tuple[float, float, float]]]]:
        """Return list of (elevation_label, polyline_xyz_local)."""
        import matplotlib.pyplot as plt

        z = self.local_elev
        z_min = float(np.nanmin(z))
        z_max = float(np.nanmax(z))
        if not math.isfinite(z_min) or not math.isfinite(z_max) or z_max - z_min < 0.1:
            return []

        start = math.ceil(z_min / interval_m) * interval_m
        levels = np.arange(start, z_max + interval_m * 0.5, interval_m)
        if len(levels) == 0:
            levels = np.array([(z_min + z_max) / 2])

        # contour expects X,Y meshgrid matching Z shape (rows=y, cols=x)
        xx, yy = np.meshgrid(self.xs, self.ys)
        fig = plt.figure()
        try:
            cs = plt.contour(xx, yy, z, levels=levels)
            contours: list[tuple[float, list[tuple[float, float, float]]]] = []
            # Matplotlib >=3.8 uses cs.allsegs; older uses collections
            if hasattr(cs, "allsegs"):
                for level, segs in zip(cs.levels, cs.allsegs):
                    for seg in segs:
                        if len(seg) < 2:
                            continue
                        pts = []
                        for x, y in seg:
                            lx, ly = self.to_local(float(x), float(y))
                            pts.append((lx, ly, float(level)))
                        contours.append((float(level), pts))
            return contours
        finally:
            plt.close(fig)


def fetch_elevation_grid(
    bounds: dict[str, float],
    source_epsg: int,
    grid_size: int = 40,
) -> TerrainGrid | None:
    """Fetch SRTM elevations (OpenTopoData, then Open-Meteo fallback)."""
    to_wgs = Transformer.from_crs(f"EPSG:{source_epsg}", "EPSG:4326", always_xy=True)

    # Align grid exactly to model bounds so SW corner is local (0,0).
    xs = np.linspace(bounds["minx"], bounds["maxx"], grid_size)
    ys = np.linspace(bounds["miny"], bounds["maxy"], grid_size)

    locations: list[str] = []
    lat_list: list[float] = []
    lon_list: list[float] = []
    for y in ys:
        for x in xs:
            lon, lat = to_wgs.transform(float(x), float(y))
            locations.append(f"{lat:.5f},{lon:.5f}")
            lat_list.append(lat)
            lon_list.append(lon)

    elevations = _fetch_elevations_opentopo(locations)
    if elevations is None:
        print("  OpenTopoData unavailable; trying Open-Meteo...")
        elevations = _fetch_elevations_openmeteo_points(lat_list, lon_list)

    if elevations is None or len(elevations) != grid_size * grid_size:
        print("  Terrain elevation fetch failed.")
        return None

    elev = np.array(elevations, dtype=float).reshape((grid_size, grid_size))
    if np.all(np.isnan(elev)):
        print("  Terrain elevations are empty.")
        return None

    elev = np.nan_to_num(elev, nan=float(np.nanmean(elev)))
    z_base = float(np.min(elev))
    return TerrainGrid(
        xs=xs,
        ys=ys,
        elev=elev,
        origin_x=bounds["minx"],
        origin_y=bounds["miny"],
        z_base=z_base,
    )


def _fetch_elevations_opentopo(locations: list[str]) -> list[float] | None:
    elevations: list[float] = []
    batch_size = 100
    try:
        for i in range(0, len(locations), batch_size):
            batch = locations[i : i + batch_size]
            response = requests.get(
                "https://api.opentopodata.org/v1/srtm30m",
                params={"locations": "|".join(batch)},
                timeout=90,
                headers={"User-Agent": "MappingSiteModel/1.0"},
            )
            response.raise_for_status()
            payload = response.json()
            if payload.get("status") != "OK":
                raise RuntimeError(payload.get("error") or "OpenTopoData error")
            for result in payload.get("results") or []:
                elev = result.get("elevation")
                elevations.append(float(elev) if elev is not None else float("nan"))
            time.sleep(1.1)
        return elevations
    except Exception as exc:
        print(f"  OpenTopoData failed ({exc})")
        return None


def _fetch_elevations_openmeteo_points(lats: list[float], lons: list[float]) -> list[float] | None:
    elevations: list[float] = []
    batch_size = 80
    try:
        for i in range(0, len(lats), batch_size):
            response = requests.get(
                "https://api.open-meteo.com/v1/elevation",
                params={
                    "latitude": ",".join(f"{v:.5f}" for v in lats[i : i + batch_size]),
                    "longitude": ",".join(f"{v:.5f}" for v in lons[i : i + batch_size]),
                },
                timeout=90,
                headers={"User-Agent": "MappingSiteModel/1.0"},
            )
            response.raise_for_status()
            elevations.extend(float(v) for v in response.json().get("elevation") or [])
            time.sleep(0.5)
        return elevations
    except Exception as exc:
        print(f"  Open-Meteo failed ({exc})")
        return None


def _fetch_elevations_openmeteo(lats: np.ndarray, lons: np.ndarray) -> list[float] | None:
    # Kept for compatibility; prefer _fetch_elevations_openmeteo_points.
    lat_list = [float(v) for v in np.repeat(lats, len(lons))]
    lon_list = [float(v) for v in np.tile(lons, len(lats))]
    return _fetch_elevations_openmeteo_points(lat_list, lon_list)


def add_polygon_extrusion(
    model: rhino3dm.File3dm,
    polygon: Polygon,
    layer_index: int,
    terrain: TerrainGrid | None,
    origin_x: float,
    origin_y: float,
    height: float,
    base_boost: float = 0.0,
    *,
    base_z: float | None = None,
) -> bool:
    exterior = list(polygon.exterior.coords)
    if exterior[0] != exterior[-1]:
        exterior = list(exterior) + [exterior[0]]
    if len(exterior) < 4:
        return False

    if base_z is None:
        # Default: flat base at the HIGHEST terrain under the footprint.
        if terrain is not None:
            base_z = terrain.sample_z_max(polygon) + base_boost
        else:
            base_z = base_boost
    height = abs(float(height))
    if height <= 0:
        height = 0.05

    flat = [
        rhino3dm.Point3d(float(x) - origin_x, float(y) - origin_y, base_z)
        for x, y in exterior
    ]
    curve = rhino3dm.PolylineCurve(flat)
    if curve is None:
        return False

    attrs = rhino3dm.ObjectAttributes()
    attrs.LayerIndex = layer_index
    extrusion = rhino3dm.Extrusion.Create(curve, height, True)
    if extrusion is None:
        model.Objects.AddCurve(curve, attrs)
        return True

    # Extrusion.Create can point "up" or "down" depending on curve winding.
    # Force Min.Z == base_z, Max.Z == base_z + height.
    bb = extrusion.GetBoundingBox()
    if abs(bb.Min.Z - base_z) > 1e-3:
        extrusion.Transform(rhino3dm.Transform.Translation(0.0, 0.0, base_z - bb.Min.Z))
        bb = extrusion.GetBoundingBox()
    if bb.Max.Z < base_z + height * 0.5:
        extrusion.Transform(rhino3dm.Transform.Translation(0.0, 0.0, height))

    model.Objects.AddExtrusion(extrusion, attrs)
    return True


def building_base_z(polygon: Polygon, terrain: TerrainGrid | None, sink_m: float = BUILDING_SINK_M) -> float:
    """Bottom face Z so the footprint centroid sits sink_m below terrain at that point."""
    c = polygon.centroid
    ground = terrain.sample_z(float(c.x), float(c.y)) if terrain is not None else 0.0
    return ground - sink_m


def add_line_on_terrain(
    model: rhino3dm.File3dm,
    line: LineString,
    layer_index: int,
    terrain: TerrainGrid | None,
    origin_x: float,
    origin_y: float,
    z_boost: float = 0.05,
) -> bool:
    """Add a draped centerline curve on the terrain (no road width surfaces)."""
    if line.length == 0:
        return False

    points = []
    for x, y in line.coords:
        lx = float(x) - origin_x
        ly = float(y) - origin_y
        z = (terrain.sample_z(float(x), float(y)) if terrain is not None else 0.0) + z_boost
        points.append(rhino3dm.Point3d(lx, ly, z))
    if len(points) < 2:
        return False
    curve = rhino3dm.PolylineCurve(points)
    attrs = rhino3dm.ObjectAttributes()
    attrs.LayerIndex = layer_index
    model.Objects.AddCurve(curve, attrs)
    return True


def write_site_model_readme(path: Path, info: dict[str, Any]) -> None:
    lines = [
        "3D site model (Rhino) — Cadmapper-style",
        "=======================================",
        "",
        f"Source CRS: EPSG:{info['source_epsg']} ({info.get('source_crs_name')})",
        "Model coordinates: LOCAL meters with SW corner at origin (0,0,0).",
        f"World origin offset (source CRS): X={info['origin_x']:.3f}, Y={info['origin_y']:.3f}",
        f"Vertical datum: local Z = elevation - {info.get('z_base', 0):.2f} m",
        "",
        f"Model bounds (source CRS, meters):",
        f"  minx={info['bounds']['minx']}",
        f"  miny={info['bounds']['miny']}",
        f"  maxx={info['bounds']['maxx']}",
        f"  maxy={info['bounds']['maxy']}",
        f"Area: {info['area_km2']:.3f} km²",
        f"Padding applied: {info['padding_m']} m on all sides",
        f"Downsized: {info['downsized']}",
        f"Contour interval: {info.get('contour_interval_m')} m",
        "",
        "Layers:",
    ]
    for name, count in info["layer_counts"].items():
        lines.append(f"  {name}: {count}")
    lines.extend(
        [
            "",
            "Data sources (same family as Cadmapper):",
            "  Buildings / roads / parks / water: OpenStreetMap (ODbL)",
            "  Terrain mesh + contours: NASA SRTM via OpenTopoData / Open-Meteo",
            "",
            "Open site_model.3dm in Rhino.",
            "  Buildings: extruded solids; bottom centroid sits 1 m below terrain.",
            "  Roads / parks / water / parking: curves draped on the terrain mesh.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def create_site_model(
    metadata: dict,
    output_dir: Path,
    *,
    padding_m: float = 10.0,
    max_area_km2: float = 1.0,
    auto_downsize: bool = True,
    include_terrain: bool = True,
    contour_interval_m: float = 1.0,
    terrain_grid_size: int = 40,
    default_building_height_m: float = 10.0,
    meters_per_level: float = 3.0,
    road_surfaces: bool = False,
    gis_folder: Path | None = None,
) -> dict[str, Any]:
    # road_surfaces kept for API compatibility; roads are always centerline curves.
    _ = road_surfaces
    source_epsg = metadata["projection"]["epsg"]
    if source_epsg is None:
        raise ValueError("Geo metadata is missing EPSG code; cannot build site model.")

    # 3D uses folder site bounds (parcels/buildings/contours), not the enlarged SVG canvas.
    site_bounds = metadata["bounds"].get("site") or metadata["bounds"].get("combined")
    if not site_bounds:
        site_bounds = metadata["bounds"]["export"]
    model_bounds = expand_bounds(site_bounds, padding_m)
    original_area = area_km2(model_bounds)
    downsized = False

    print(f"  Model area after +{padding_m:.0f} m padding: {original_area:.2f} km²")
    print(
        f"  Using folder site bounds (not SVG canvas): "
        f"minx={site_bounds['minx']:.2f}, miny={site_bounds['miny']:.2f}, "
        f"maxx={site_bounds['maxx']:.2f}, maxy={site_bounds['maxy']:.2f}"
    )
    if original_area > max_area_km2:
        print(
            f"  WARNING: Area {original_area:.2f} km² exceeds {max_area_km2:.2f} km² "
            "(Overpass is slower; Cadmapper free tier is ~1 km²)."
        )
        if auto_downsize:
            model_bounds = shrink_bounds_to_max_area(model_bounds, max_area_km2)
            downsized = True
            print(
                f"  Auto-downsized to centered {area_km2(model_bounds):.2f} km² "
                f"(set SITE_MODEL_AUTO_DOWNSIZE = False to keep full site bounds)."
            )
        else:
            print("  Keeping full folder site bounds (SITE_MODEL_AUTO_DOWNSIZE = False).")

    origin_x = model_bounds["minx"]
    origin_y = model_bounds["miny"]
    print(f"  Local origin (SW corner): world X={origin_x:.2f}, Y={origin_y:.2f} -> Rhino (0,0)")

    west, south, east, north = source_bounds_to_wgs84(model_bounds, source_epsg)
    print(
        f"  OSM bbox WGS84: west={west:.6f}, south={south:.6f}, "
        f"east={east:.6f}, north={north:.6f}"
    )

    terrain: TerrainGrid | None = None
    if include_terrain:
        print("  Fetching SRTM terrain (Cadmapper-style)...")
        terrain = fetch_elevation_grid(model_bounds, source_epsg, grid_size=terrain_grid_size)
        if terrain is None:
            print("  WARNING: No terrain - buildings will sit on flat Z=0.")
        else:
            print(
                f"  Terrain OK: elev range {terrain.z_base:.1f}–"
                f"{float(np.max(terrain.elev)):.1f} m "
                f"(local Z 0–{float(np.max(terrain.local_elev)):.1f} m)"
            )

    geojson = fetch_osm_geojson(west, south, east, north)
    features = project_geojson_to_crs(geojson, source_epsg)
    features = clip_features(features, model_bounds)
    print(f"  OSM features after clip: {len(features)}")
    features = load_or_refresh_osm_features(features, output_dir, area_km2=area_km2(model_bounds))

    has_buildings = any(classify_feature(feature_props(f["properties"])) == "Buildings" for f in features)
    has_roads = any(
        (classify_feature(feature_props(f["properties"])) or "").startswith("Roads_") for f in features
    )
    local = load_local_gis_features(gis_folder, model_bounds, source_epsg)
    if local:
        local_buildings = [f for f in local if feature_props(f["properties"]).get("building")]
        local_roads = [f for f in local if feature_props(f["properties"]).get("highway")]
        local_parks = [f for f in local if feature_props(f["properties"]).get("leisure")]
        # Prefer MassGIS building footprints when present (matches SVG building layer).
        if local_buildings:
            features = [
                f
                for f in features
                if classify_feature(feature_props(f["properties"])) != "Buildings"
            ] + local_buildings
            print(f"  Using {len(local_buildings)} local GIS buildings (matches SVG structures).")
        if not has_roads and local_roads:
            features.extend(local_roads)
            print(f"  Added {len(local_roads)} local GIS road centerlines.")
        # Always merge local openspace — do not drop it when OSM returns a partial park set.
        if local_parks:
            features.extend(local_parks)
            print(f"  Added {len(local_parks)} local GIS openspace/park polygons.")
        print(f"  Features after local GIS merge: {len(features)}")

    if not features:
        raise RuntimeError(
            "No site features returned (OSM failed and no local building/road shapefiles). "
            "Retry later or check INPUT_FOLDER GIS layers."
        )

    model = rhino3dm.File3dm()
    model.Settings.ModelUnitSystem = rhino3dm.UnitSystem.Meters
    layer_index = ensure_layers(model)
    layer_counts = {name: 0 for name in LAYER_COLORS}

    # Terrain mesh + contours first (under site features)
    if terrain is not None:
        attrs = rhino3dm.ObjectAttributes()
        attrs.LayerIndex = layer_index["Terrain"]
        model.Objects.AddMesh(terrain.build_mesh(), attrs)
        layer_counts["Terrain"] = 1

        print(f"  Building contours every {contour_interval_m:g} m...")
        contour_count = 0
        for _level, pts in terrain.build_contours(contour_interval_m):
            if len(pts) < 2:
                continue
            curve = rhino3dm.PolylineCurve([rhino3dm.Point3d(*p) for p in pts])
            attrs = rhino3dm.ObjectAttributes()
            attrs.LayerIndex = layer_index["Contours"]
            model.Objects.AddCurve(curve, attrs)
            contour_count += 1
        layer_counts["Contours"] = contour_count
        print(f"  Contour polylines: {contour_count}")

    for feature in features:
        props = feature_props(feature["properties"])
        layer_name = classify_feature(props)
        if layer_name is None:
            continue
        geom = feature["geometry"]
        idx = layer_index[layer_name]

        if layer_name == "Buildings":
            height = building_height_m(props, default_building_height_m, meters_per_level)
            for poly in iter_polygons(geom):
                if add_polygon_extrusion(
                    model,
                    poly,
                    idx,
                    terrain,
                    origin_x,
                    origin_y,
                    height,
                    base_z=building_base_z(poly, terrain, BUILDING_SINK_M),
                ):
                    layer_counts[layer_name] += 1
            continue

        if layer_name.startswith("Roads_"):
            for line in iter_lines(geom):
                if add_line_on_terrain(model, line, idx, terrain, origin_x, origin_y):
                    layer_counts[layer_name] += 1
            # Plaza / pedestrian areas tagged as highway polygons -> outline curves
            for poly in iter_polygons(geom):
                ring = LineString(poly.exterior.coords)
                if add_line_on_terrain(model, ring, idx, terrain, origin_x, origin_y):
                    layer_counts[layer_name] += 1
            continue

        if layer_name == "Railways":
            for line in iter_lines(geom):
                if add_line_on_terrain(model, line, idx, terrain, origin_x, origin_y):
                    layer_counts[layer_name] += 1
            continue

        # Parks, water, parking, landuse: outline curves draped on terrain (not pads).
        if layer_name in SURFACE_OUTLINE_LAYERS:
            for poly in iter_polygons(geom):
                layer_counts[layer_name] += add_polygon_outline_on_terrain(
                    model, poly, idx, terrain, origin_x, origin_y
                )
            for line in iter_lines(geom):
                dense = densify_linestring(line)
                if add_line_on_terrain(model, dense, idx, terrain, origin_x, origin_y):
                    layer_counts[layer_name] += 1
            continue

        for poly in iter_polygons(geom):
            if add_polygon_extrusion(model, poly, idx, terrain, origin_x, origin_y, height=0.05):
                layer_counts[layer_name] += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "site_model.3dm"
    readme_path = output_dir / "site_model_readme.txt"
    model.Write(str(model_path), 7)

    info = {
        "source_epsg": source_epsg,
        "source_crs_name": metadata["projection"].get("name"),
        "bounds": model_bounds,
        "origin_x": origin_x,
        "origin_y": origin_y,
        "z_base": terrain.z_base if terrain is not None else 0.0,
        "area_km2": area_km2(model_bounds),
        "padding_m": padding_m,
        "downsized": downsized,
        "contour_interval_m": contour_interval_m,
        "layer_counts": layer_counts,
        "model_path": model_path,
        "readme_path": readme_path,
    }
    write_site_model_readme(readme_path, info)

    print(f"  Wrote {model_path}")
    print(f"  Wrote {readme_path}")
    for name, count in layer_counts.items():
        if count:
            print(f"    {name}: {count}")
    return info

