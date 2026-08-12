# Prometheus

**Daily wildfire risk forecasting for Nepal (pre-monsoon, January–May).**

Most Nepal fire maps answer “where can fires occur over a decade?” Prometheus
answers “what is the chance each 1 km forest cell burns **tomorrow** — and over
the next week?”

Plain-language tour: **[explanation.md](explanation.md)**  
Measured results: **[PROGRESS_REPORT.md](PROGRESS_REPORT.md)**

---

## What you get when it is running

| URL | What it is |
|---|---|
| http://localhost:5173 | Map, fires, district pages, accuracy |
| http://localhost:8000/docs | Live API |
| http://localhost:8000/api/health | Forecast count + default date |

The map paints **calibrated** fire probability (yellow → purple). 🔥 markers
are satellite detections that already happened — not the forecast.

---

## Why the map shows 2024–2025, not 2026

The **model was trained and tested on 2016–2026**. 2026 is a real evaluation
year (PR-AUC 0.144). The **website only lists years that were written as daily
GeoTIFFs**. The first backfill was 2024 and 2025 (~303 days), as specified in
the build plan.

2026 is not missing from science; it was not exported to `runs/forecasts/`
yet. To put it on the map (needs the feature cube for 2026, which this repo
already has if you built through Day 8):

```bash
source .prometheus-venv/bin/activate
python scripts/forecast.py --backfill 2026
python scripts/forecast.py --verify 2026-01-01 2026-05-30
```

Then refresh the app. The year tabs come from `/api/forecasts`, not a hardcoded
list.

---

## Requirements

| | Minimum |
|---|---|
| OS | macOS or Linux (this project was built on an M-series Mac) |
| Python | **3.12** (3.11 also works; 3.13 is not pinned) |
| Node.js | 18+ (for the website) |
| RAM | 8 GB comfortable; season warm-up uses ~3–4 GB |
| Disk | ~12 GB if `data/` + `runs/` already exist; much more if you rebuild GEE |

You do **not** need a GPU. LightGBM is the production model. The optional U-Net
comparison used Apple Metal and is not required to run the app.

---

## Path A — this machine already has data (fastest)

If `data/cube/`, `data/models/bundles/v1/`, and `runs/forecasts/` exist, skip
the multi-day data build.

```bash
cd /path/to/Prometheus

# 1. Python environment
python3.12 -m venv .prometheus-venv
source .prometheus-venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
python -m pip install fastapi "uvicorn[standard]" titiler.core

# 2. Website packages
cd frontend && npm install && cd ..

# 3. Sanity
python -c "from prometheus.config import cfg; print(cfg.years, cfg.season_months)"
python -m pytest tests/ -q

# 4. Run (two terminals)
make api          # http://127.0.0.1:8000
make ui           # http://localhost:5173
```

Open **http://localhost:5173**. Default date is **12 Apr 2025**.

If tiles 404, the API is not running or `runs/forecasts/` is empty — use Path B
from “Inference” downward.

---

## Path B — full rebuild from scratch

Do this only if `data/` is empty. Order matters. Budget several days for NASA
FIRMS + Google Earth Engine exports; local compute after that is hours, not
days.

### B1. Environment (same as Path A)

```bash
python3.12 -m venv .prometheus-venv
source .prometheus-venv/bin/activate
python -m pip install -U pip hatchling
python -m pip install -e ".[dev]"
python -m pip install fastapi "uvicorn[standard]" titiler.core
cd frontend && npm install && cd ..
```

All years, paths, and the 1 km grid live in `configs/base.yaml`. Do not
hardcode them elsewhere.

### B2. Fire labels (NASA FIRMS)

1. Get a free FIRMS MAP key: https://firms.modaps.eosdis.nasa.gov/api/area/
2. Save it (never commit it):

```bash
mkdir -p data/raw/firms
echo 'YOUR_KEY' > data/raw/firms/.map_key
```

3. Build the daily fire cube (uses the **area** API only; `day_range` must be
   1–5):

```bash
python scripts/build_fire_labels.py
```

Expect `data/cube/fire_daily.zarr` with shape like `(1664, 465, 912)` for
2016–2026 Jan–May.

### B3. Earth Engine predictors + local static

Follow **[docs/DAY4_5_MANUAL.md](docs/DAY4_5_MANUAL.md)** in full.

Short version:

- Run `gee/era5_daily.js`, `gee/lst_8day.js`, `gee/ndvi_16day.js`,
  `gee/static.js` in the Earth Engine Code Editor (years already set to 2026).
- Download GeoTIFFs into `data/raw/gee/`.
- Deduplicate with `python scripts/audit_gee_raw.py`.
- Rasterise OSM roads/settlements:

```bash
python scripts/build_local_static.py
```

You need a Nepal mask GeoTIFF at the path in `configs/base.yaml`
(`nepal_mask_1km_roiAligned.tif`) plus SRTM elevation/slope.

### B4. Feature cube and training table

```bash
python scripts/build_feature_cube.py          # ~4 min once rasters are local
python scripts/plot_cube_check.py --year 2021
python scripts/build_train_table.py           # ~40 s
python scripts/plot_feature_diagnostics.py
```

### B5. Baselines, LightGBM, full CV, freeze the bundle

```bash
python scripts/run_baselines.py
python scripts/train_lightgbm.py --year 2021
python scripts/run_cv.py                      # ~2 h
python scripts/plot_shap.py --year 2021
python scripts/build_model_bundle.py          # ~8 min; writes data/models/bundles/v1
python scripts/plot_calibration.py
```

Optional CNN (needs Metal outside a sandbox; skip if you are in a hurry):

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
python -c "from prometheus.cnn import stacks; stacks.build_all()"
python -u scripts/train_unet.py --batch-size 16
```

### B6. Daily maps the website can serve

```bash
make forecast DATE=2025-04-12
make backfill-forecasts                       # 2024 + 2025
# optional:
# python scripts/forecast.py --backfill 2026
make verify-forecasts
```

Then `make api` and `make ui` as in Path A.

---

## Day-to-day commands

```bash
source .prometheus-venv/bin/activate

make forecast DATE=2025-04-12     # one day → COGs + districts GeoJSON
make backfill-forecasts           # 2024–2025 seasons
make verify-forecasts             # score h1 vs next-day FIRMS
make api                          # FastAPI :8000
make ui                           # Vite :5173  (proxies /api → :8000)

python -m pytest tests/ -q
ruff check src scripts tests
cd frontend && npm run lint && npm run build
```

Useful API checks:

```bash
curl -s http://127.0.0.1:8000/api/health
curl -s http://127.0.0.1:8000/api/forecasts | head
curl -s -o /tmp/t.png -w "%{http_code} %{size_download}\n" \
  "http://127.0.0.1:8000/api/risk/tiles/7/93/53.png?date=2025-04-12&horizon=1"
```

---

## Repository map

```
configs/base.yaml          years, grid, features, CV — single source of truth
Makefile                   forecast / backfill / verify / api / ui
src/prometheus/
  grid.py                  465×912, EPSG:4326
  data/                    FIRMS download → cube
  features/                weather, veg, forest mask, training table
  eval/                    metrics, climatology, leave-one-year-out
  models/                  LightGBM, calibration, frozen bundle, predict
  cnn/                     optional U-Net (lost to LightGBM)
  infer/                   COG write, districts, backfill, verification
  api/                     FastAPI routes
gee/                       Earth Engine export scripts
scripts/                   CLIs only
frontend/                  React + Vite + Leaflet
tests/
docs/                      manuals + report draft
data/  runs/               gitignored runtime artefacts
explanation.md             beginner explanation + literature comparison
PROGRESS_REPORT.md         every measured number
BUILD_PLAN.md              original 3-week plan
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Website loads, map is blank / “API down” | FastAPI not running | `make api` in another terminal |
| `/api/risk/tiles/...` 404 | No COG for that date | `make forecast DATE=YYYY-MM-DD` or backfill |
| `date outside the modelled Jan–May season` | June–December | Only Jan–May exists by design |
| FIRMS HTTP 400 | `day_range` > 5 or wrong DATE | See `src/prometheus/data/firms.py` |
| `osgeo` / GDAL missing | Optional; rasterio is enough | Ignore unless rasterio itself fails |
| U-Net / MPS not found | Cursor sandbox hides Metal | Run `train_unet.py` in a normal Terminal |
| Light mode looks wrong after an update | Cached CSS | Hard refresh the browser |
| `districts` timeseries is slow the first time | Builds `_district_ts.json` | Wait once; later calls are instant |
| Port 8000 or 5173 busy | Old server still running | Kill that process or change `--port` |

---

## Rules we do not break

1. Every raster is 465 × 912, EPSG:4326, aligned to `grid.py`.
2. No years or paths hardcoded outside `configs/base.yaml`.
3. A model that does not beat **climatology** does not ship.
4. Always print the **base rate** next to a metric.
5. Never commit `data/`, `runs/`, `*.tif`, `*.zarr`, or `.map_key`.

---

## License / data

FIRMS, ERA5-Land, MODIS, SRTM, and WorldCover are used under their respective
open terms. Cite NASA FIRMS and ECMWF/ERA5 if you publish maps. This repository
is a student research system, not an official Government of Nepal product.
