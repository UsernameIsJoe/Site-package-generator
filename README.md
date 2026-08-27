# Site Package Generator

Generate an Illustrator-ready site package and optional Rhino 3D model from a folder of GIS shapefiles.

## What it produces

For an input folder such as `Underwood/` or `Ward/`:

| Output | Description |
|--------|-------------|
| `layers.svg` | Vector GIS layers for Illustrator |
| `layers.geo.json` | Bounds / CRS metadata |
| `basemap.png` / `.tif` | Print-resolution basemap (Sherwin-Williams palette, no labels) |
| `site_model.3dm` | Rhino site model (buildings, roads, parks, water, terrain, contours) |

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m playwright install chromium
```

## Use

1. Put shapefile layer folders inside a mother folder (e.g. `Underwood/Building Structures (2-D)/*.shp`).
2. Set `INPUT_FOLDER` at the top of `export_map.py`.
3. Run `export_map.py` (VS Code ▶ or `python export_map.py`).

Outputs land in `output/<folder_name>/`.

## Notes

- Basemap uses **OpenFreeMap** (no API key) via MapLibre / Playwright.
- Optional CARTO raster tiles need a free key if you switch `BASEMAP_RASTER_SOURCE = "carto"`.
- Tune colors in the `SW_*` / `BASEMAP_PALETTE` settings in `export_map.py`.
- Attribution: © OpenStreetMap contributors; OpenMapTiles; OpenFreeMap.
