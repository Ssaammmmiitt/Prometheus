# Days 4–5 manual — GEE exports + local static layers

Hands-on guide. GEE work runs in Google's cloud; you only start tasks and later download. Do local OSM / distance layers in parallel on your Mac.

---

## What “done” looks like

| Check | Target |
|---|---|
| Folder size | `data/raw/` ≈ **1.3 GB** (order of magnitude) |
| Layout | GEE files under `data/raw/gee/{era5,lst,ndvi,static}/` |
| Local static | `data/static/dist_road.tif`, `dist_settlement.tif`, `physio_regions.tif` (+ terrain if needed) |
| Test | `pytest tests/test_static_alignment.py` — every listed static GeoTIFF matches grid **465×912**, EPSG:4326, same transform as the Nepal mask |

You already have `nepal_mask`, elevation, slope under `data/static/` from earlier. GEE still exports elev/slope/aspect/TWI/WorldCover so the whole static pack lives in one place.

---

## Prerequisites (once)

1. **Google account** with Earth Engine enabled  
   - Go to [https://earthengine.google.com/](https://earthengine.google.com/) → Register / join Cloud Project if asked.  
   - Free academic / personal EE access is enough for this project.

2. **Google Drive** with free space (several GB headroom).

3. **This repo** on your Mac, venv activated:

```bash
cd /Users/sammit/Desktop/Projects/Prometheus
source .prometheus-venv/bin/activate
```

4. Scripts live in:

```text
gee/00_roi.js          # ROI constants (reference only)
gee/era5_daily.js
gee/lst_8day.js
gee/ndvi_16day.js
gee/static.js
```

ROI matches `configs/base.yaml` / Nepal mask bbox:

```text
lon 80.0129–88.2056 · lat 26.3386–30.5158 · EPSG:4326
Years 2016–**2026** · months Jan–May only  
(If 2016–2025 GEE tasks already ran, set `START_YEAR = END_YEAR = 2026` in the JS and export only the new season.)
```

---

## Part A — GEE exports (start first; let them run)

### A1. Open the Code Editor

1. Open [https://code.earthengine.google.com/](https://code.earthengine.google.com/).
2. Sign in with the account that has EE access.
3. (Optional) Create a Cloud Project if the editor asks; accept defaults.

### A2. Run one script at a time

For **each** file in order below:

1. Open the `.js` file locally in Cursor / VS Code.
2. **Select all → copy**.
3. In the EE Code Editor, **clear the script pane** and **paste**.
4. Click **Run** (top center).  
   - You should see a `print(...)` message in the Console.  
   - For LST/NDVI, the first run calls `getInfo()` and may take **1–3 minutes** while it lists images — wait.
5. Open the **Tasks** tab (right panel).
6. Start exports:
   - Either click **RUN** next to each task, or  
   - Select many → use bulk start if available.
7. Leave the browser tab open or just check back later. Tasks run on Google servers; your Mac can do other work.

| Order | Script | Drive folder | Scale | What you get |
|---|---|---|---|---|
| 1 | `static.js` | `Prometheus_GEE/static` | **1000** m | elev+slope+aspect, twi, worldcover fractions (~few files) |
| 2 | `era5_daily.js` | `Prometheus_GEE/era5` | **11132** m (~0.1°) | **50** monthly stacks (`era5_YYYY_MM.tif`) |
| 3 | `lst_8day.js` | `Prometheus_GEE/lst` | **1000** m | ~one GeoTIFF per 8-day composite in season |
| 4 | `ndvi_16day.js` | `Prometheus_GEE/ndvi` | **1000** m | ~one GeoTIFF per 16-day composite in season |

**Critical for ERA5:** scale must stay **11132**. Exporting ERA5 at 1000 m only invents fake 1 km weather and wastes Drive space. Weather stays ~9 km until feature code upsamples to the 1 km grid.

### A3. How many Tasks / how long

Rough counts (order of magnitude):

| Family | Tasks | Wall time after Start |
|---|---|---|
| Static | 3 | minutes–hour |
| ERA5 | 50 (10 y × 5 mo) | hours; often overnight batch |
| LST | ~150–200 | hours |
| NDVI | ~80–100 | hours |

EE free tier has daily export quotas. If tasks fail with “user memory limit” or “export too large”, re-run only failed months, or split years (edit `START_YEAR` / `END_YEAR` in the script and export in two batches).

### A4. Download from Drive → project

1. Open [https://drive.google.com](https://drive.google.com) → folder **`Prometheus_GEE`**.
2. When a subfolder looks complete (file counts match Tasks successes), download **as a zip** or use [rclone](https://rclone.org/) / Drive desktop sync.
3. Unpack into the repo:

```bash
mkdir -p data/raw/gee/{era5,lst,ndvi,static}

# After downloading Drive folders, move files, e.g.:
# mv ~/Downloads/era5/*.tif data/raw/gee/era5/
# mv ~/Downloads/lst/*.tif  data/raw/gee/lst/
# ...
```

Suggested final layout:

```text
data/raw/gee/
  era5/era5_2016_01.tif ...
  lst/lst_20160101.tif ...
  ndvi/ndvi_20160101.tif ...
  static/elev_slope_aspect.tif  twi.tif  worldcover_frac.tif
```

4. Check size:

```bash
du -sh data/raw data/raw/gee/*
```

Expect total raw (FIRMS + GEE) on the order of **~1–1.5 GB** with the plan’s budgeting; ERA5 native is small vs 1 km weather.

### A5. Optional: pull terrain WorldCover into `data/static/`

After static GEE files land, you can regrid them to the **exact** mask grid (Day 6 will systematize this). For a quick copy check, keep the Drive GeoTIFFs under `data/raw/gee/static/` for now.

---

## Part B — Local non-GEE static (do while Tasks run)

### B1. Download OpenStreetMap Nepal extract

1. Open Geofabrik: [https://download.geofabrik.de/asia/nepal.html](https://download.geofabrik.de/asia/nepal.html)
2. Download **`nepal-latest-free.shp.zip`** (shapefile pack; easy with geopandas) **or** `nepal-latest.osm.pbf` if you prefer osmium later.
3. Unpack:

```bash
mkdir -p data/raw/osm
unzip ~/Downloads/nepal-latest-free.shp.zip -d data/raw/osm/nepal-free
# Expect folders like gis_osm_roads_free_1.* , gis_osm_places_free_1.* , etc.
```

### B2. Build distance-to-road and distance-to-settlement

Script: `scripts/build_local_static.py`

It:

1. Reads the Nepal mask / canonical grid from config.
2. Rasterizes OSM roads (and places/settlements) onto that grid as binary presence.
3. Runs `scipy.ndimage.distance_transform_edt` to get distance in **pixels**, converts to **km** using ~1 km pixel size from config.
4. Writes aligned GeoTIFFs under `data/static/`.
5. Builds **physiographic region** map (see B3).

```bash
source .prometheus-venv/bin/activate
# install if missing: pip install geopandas shapely scipy
python scripts/build_local_static.py \
  --osm-dir data/raw/osm/nepal-free
```

Outputs:

```text
data/static/dist_road.tif
data/static/dist_settlement.tif
data/static/physio_regions.tif   # codes 1–4
data/static/physio_regions.json  # code legend
```

Road layers used (if present): `gis_osm_roads_free_1.*`  
Settlement layers: `gis_osm_places_free_1.*` and/or `gis_osm_landuse_free_1` filtered to residential if places are thin.

### B3. Physiographic regions (Terai / Chure / Middle / High)

**Preferred:** drop a polygon file if you have one:

```bash
# e.g. ICIMOD or national physio shapefile
python scripts/build_local_static.py \
  --osm-dir data/raw/osm/nepal-free \
  --physio-vector /path/to/nepal_physio.gpkg
```

Attribute with region names (script looks for columns containing `name`, `region`, `class`, `physio` case-insensitively).

**Default fallback (no vector):** elevation classes from SRTM already on the grid (`data/static/elevation_*.tif` or GEE elev). Standard working thresholds:

| Code | Region | Elev rule (approx.) |
|---|---|---|
| 1 | Terai | &lt; 300 m |
| 2 | Chure | 300–1000 m |
| 3 | MiddleMountains | 1000–3000 m |
| 4 | HighMountains | ≥ 3000 m |

Document in the report that elevation bands are a **proxy** for physiographic regions if you use the default. Fine for LOYO features and subgroup tables; swap polygons later if you get official boundaries.

### B4. Alignment test

```bash
pytest tests/test_static_alignment.py -q
```

This asserts every file listed in the test (mask companions + dist_road / dist_settlement / physio) match `grid.assert_aligned`.

---

## Part C — Parallel workflow (recommended day schedule)

```text
Day 4 morning
  ✓ Register EE if needed
  ✓ Run static.js → Start 3 tasks
  ✓ Run era5_daily.js → Start 50 tasks
  ✓ Start Geofabrik download

Day 4 afternoon
  ✓ Run lst_8day.js + ndvi_16day.js → Start tasks
  ✓ Unzip OSM; run build_local_static.py
  ✓ pytest static alignment
  ✓ Work on docs / Day 6 design while exports chew

Day 5
  ✓ Check Tasks for failures → re-RUN failed only
  ✓ Download completed Drive folders into data/raw/gee/
  ✓ du -sh data/raw
  ✓ Spot-check: open one .tif in QGIS next to nepal mask
```

Do **not** wait for GEE to finish before doing OSM. That is wasted wall-clock time.

---

## Part D — Troubleshooting

| Symptom | Fix |
|---|---|
| “Earth Engine not enabled” | Register at earthengine.google.com; wait for approval email if required |
| Tasks empty after Run | Script error in Console (red). Fix paste / band names and re-Run |
| LST/NDVI Run hangs | First `getInfo()` is slow. Wait 2–5 min. Do not spam Run. |
| `Image.clip: Can't transform (0.0,0.0)` | MODIS is sinusoidal — **do not** `clip` a WGS84 box. Scripts now crop via `Export.region` only. Re-paste current `ndvi_16day.js` / `lst_8day.js`, cancel failed tasks, Start again. |
| Export fails “too many pixels” | Already set `maxPixels: 1e13`. Check `region: roi` is set |
| ERA5 huge files | You probably set scale 1000 by mistake. Re-export with **11132** |
| Quota exceeded | Pause new Starts; resume next day with remaining months |
| OSM script finds no roads | Wrong `--osm-dir`; point at the folder containing `gis_osm_roads_free_1.shp` |
| Test fails shape | GEE native grids are **not** yet warp-aligned; alignment test is for **local** 1 km static products only until Day 6 regrid |

---

## Part E — What you are *not* doing yet (Day 6+)

| Later step | Why |
|---|---|
| Merge ERA5 months into Zarr / daily weather on 1 km grid | Feature cube |
| Temporal interpolate NDVI/LST to daily | Features |
| LightGBM training | Needs features first |

Days 4–5 are only: **kick off cloud exports + download**, and **finish human/physio static on the canonical grid**.

---

## Checklist (copy into notes)

- [ ] EE account works; Code Editor opens  
- [ ] `static.js` tasks succeeded → files in Drive  
- [ ] `era5_daily.js` all 50 months Start/complete (scale 11132)  
- [ ] `lst_8day.js` + `ndvi_16day.js` complete  
- [ ] Files under `data/raw/gee/{era5,lst,ndvi,static}/`  
- [ ] OSM Nepal extract in `data/raw/osm/`  
- [ ] `dist_road.tif`, `dist_settlement.tif`, `physio_regions.tif` in `data/static/`  
- [ ] `pytest tests/test_static_alignment.py` green  
- [ ] `du -sh data/raw` in ~1–1.5 GB ballpark  

When that list is complete, Days 4–5 are done.
