# Prometheus — Progress Report

Living log of what was built, cleaned, and verified. Every number here comes
from a committed artefact under `runs/` or `data/`, not from an estimate.

| Field | Value |
|---|---|
| Project | Prometheus — daily wildfire risk forecasting for Nepal |
| Branch | `v2` |
| Last updated | 2026-08-11 |
| Status | **Days 1–8 complete.** Next: Day 9 (LightGBM) |
| Tests | **49 passing**, ruff clean |

---

## The thesis in one sentence

Published Nepal fire ML produces **static susceptibility maps** ("where can
fires occur?"); Prometheus produces a **dynamic daily forecast** ("what is the
probability each 1 km cell burns tomorrow?"). Every design decision below serves
that distinction.

---

## Current state at a glance

| Artefact | Path | Size | Shape |
|---|---|---|---|
| Fire labels | `data/cube/fire_daily.zarr` | 1.8 MB | 1664 × 465 × 912 |
| Feature cube | `data/cube/features_daily.zarr` | 5.69 GB | 1664 × 465 × 912 × 17 vars |
| Training table | `data/cube/train_table.parquet` | 149 MB | 2,065,833 × 53 |
| Per-fold norm stats | `data/models/norm_stats_v1.json` | 76 KB | 11 folds |
| Raw downloads | `data/raw/` | 4.7 GB | 377 GEE tiles + FIRMS |
| Static layers | `data/static/` | 2.4 MB | 6 aligned rasters |

**Canonical grid (never varies):** 465 × 912, EPSG:4326, pixel 0.008983152841195215°,
origin (80.01294235652578, 30.515770201540146), **168,064** valid Nepal cells.
Every raster in the project is asserted against it.

**Scope:** Jan 1 – May 31, **2016–2026** (11 seasons, 1664 days). Nepal's fire
season is ~91% of annual burned area in Mar–May; the monsoon half of the year
would add ~120 trivially-negative days per year and teach the model nothing.

---

## Repository layout

```
configs/base.yaml          single source of truth (years, grid, features, CV)
src/prometheus/
  config.py  grid.py       constants + canonical grid
  data/firms.py            FIRMS download → clean → rasterise
  features/
    warp.py                regrid anything onto the canonical grid
    weather.py             ERA5 9 km → 1 km, lapse-corrected
    vegetation.py          MODIS composites → daily
    forest.py              static layers + burnable mask
    cube.py                assemble features_daily.zarr
    derived.py             dryness, rolling windows, anomalies, fire history
    table.py               sampling + training table + norm stats
  eval/                    metrics, baselines, leave-one-year-out CV
  models/                  (Day 9)
gee/                       four Earth Engine export scripts
scripts/                   eight CLIs, one per build step
tests/                     49 tests
docs/                      DAY4_5_MANUAL.md, EXTEND_TO_2026.md, report draft
data/  runs/               gitignored
```

---

## Day 1 — Scaffold · Done

Package layout, `configs/base.yaml` as the single source of truth, and the
canonical 1 km Nepal grid/mask in `grid.py`. Python 3.12 venv at
`.prometheus-venv`.

**Verify:** `python -c "from prometheus.config import cfg; print(cfg.years, cfg.season_months)"`

---

## Day 2 — Fire labels · Done

FIRMS **Area API** only — the country endpoints are unavailable on the FIRMS
status board. `day_range` must be **1–5** (6+ returns HTTP 400), and `DATE` is
the window **start**, not its end. Both cost real debugging time; both are now
encoded in `firms.py` and covered by tests.

| Metric | Result |
|---|---|
| Download | 1271 chunks, ~78 min (cached; re-runs are incremental) |
| Raw → cleaned | 1,040,071 → **863,807** |
| Jan–May 2016–2026 | **821,478** detections (>120k **PASS**) |
| Sensors | VIIRS SNPP 433,320 · VIIRS NOAA-20 336,534 · MODIS 51,624 |
| Cube | `(1664, 465, 912)` · 1,876,970 fire pixel-days · 1604 days with fire |
| Outside Nepal mask | **0** |

Labels are dilated 1 pixel (3×3) to absorb satellite geolocation error.

```bash
python scripts/build_fire_labels.py     # MAP_KEY in data/raw/firms/.map_key
```

---

## Day 3 — Evaluation harness + baselines · Done

- `eval/metrics.py` — PR-AUC, ROC-AUC, Brier, skill vs climatology, top-k capture, reliability, ECE
- `eval/baselines.py` — MODIS 2003–2015 day-of-year climatology (±7 d temporal, σ=1 spatial); 7-day + 3×3 persistence
- `eval/cv.py` — leave-one-year-out, per-year table with mean ± std

### Results — mean over 11 seasons, all Nepal-mask pixels

```
model            PR-AUC  ROC-AUC    Brier top10%-capture
climatology      0.0416   0.8104   0.0067         0.5600
persistence      0.0493   0.7963   0.0569         0.6790
```

| Model | Reading |
|---|---|
| **Climatology** | Historical seasonality only. Strong ROC (0.81); the top 10% riskiest area catches **56%** of fires. Best Brier — the probabilities are well scaled. |
| **Persistence** | Higher PR-AUC and **68%** top-10% capture: recent fire plus neighbours is a genuinely strong short-term signal. Poor Brier, because it emits hard 0/1 scores. |

> **Shipping rule.** A model that does not beat **climatology PR-AUC 0.0416**
> (mean ± 0.0134 over 11 folds) does not ship. Always report mean ± std from
> `metrics_per_year.csv`, and always print the base rate next to the metric.

Interannual variance is the reason for 11 folds, not one test year: the base
rate swings **6.7×** across seasons (2020: 0.21% → 2021: 1.43%).

```bash
python scripts/run_baselines.py
```

---

## Days 4–5 — GEE exports + local static · Done

Manual step-by-step: **[docs/DAY4_5_MANUAL.md](docs/DAY4_5_MANUAL.md)**

| Script | Collection | Scale | Files landed |
|---|---|---|---|
| `gee/era5_daily.js` | ECMWF/ERA5_LAND/DAILY_AGGR | **11132 m** | 55 monthly stacks |
| `gee/lst_8day.js` | MODIS/061/MOD11A2 | 1000 m | 209 composites |
| `gee/ndvi_16day.js` | MODIS/061/MOD13Q1 | 1000 m | 110 composites |
| `gee/static.js` | SRTM + WorldCover v200 | 1000 m | 3 rasters |

**ERA5 stays at its native ~9 km.** Exporting it at 1 km would invent detail
that does not exist and inflate the download 121×; the lapse-rate downscaling on
Day 6 is the honest way to reach 1 km.

**Bug fixed:** MODIS lives in a sinusoidal CRS, so `img.clip(roi)` on a WGS84
rectangle fails with `Image.clip: Can't transform (0.0,0.0)`. The NDVI and LST
scripts now crop via `Export.region` alone.

**Deduplication:** Drive delivered 210 `name(1).tif` duplicates (20 NDVI, 190
LST) that were ~25% smaller than their twins. Removed, keeping the larger file
per date: 588 → 377 files, 1.6 GB → 1.2 GB. Re-check anytime with
`python scripts/audit_gee_raw.py`.

**Local non-GEE static** — `scripts/build_local_static.py` rasterises the
Geofabrik Nepal OSM extract and runs `distance_transform_edt`:

| Layer | Notes |
|---|---|
| `dist_road.tif` | OSM roads, km |
| `dist_settlement.tif` | OSM places + residential landuse, km |
| `physio_regions.tif` | Terai / Chure / Middle / High. **Elevation-band proxy** unless a polygon file is supplied — declare this as a limitation |

All six `data/static/` rasters pass `grid.assert_aligned`. The forest mask is
**not** stored here — it is derived from WorldCover on demand by
`features/forest.py` and written into the cube, so there is exactly one
definition of it.

---

## Train window extended to 2026

Originally frozen at 2016–2025. Since the Jan–May 2026 season is complete, it
was added: **11 LOYO folds** instead of 10. `configs/base.yaml` drives this, so
FIRMS windows and the GEE year loops followed automatically. Rationale and
step-by-step: **[docs/EXTEND_TO_2026.md](docs/EXTEND_TO_2026.md)**.

---

## Days 6–7 — Feature cube · Done

`data/cube/features_daily.zarr` — **1664 × 465 × 912**, float16, chunks
`(32, 256, 256)`, **5.69 GB**, rebuilds in ~4 min.

| Group | Variables |
|---|---|
| Weather, ERA5 9 km → 1 km (9) | `t2m_max` `t2m_min` `t2m` `d2m` `precip` `u10` `v10` `soil_water_l1` `surface_pressure` |
| Derived (3) | `rh` `vpd` `wind_speed` |
| Veg / thermal, interpolated to daily (5) | `ndvi` `evi` `lst_day` `lst_night` `lst_diff` |
| Static 2D (18) | terrain, WorldCover fractions, `dist_road`, `dist_settlement`, `physio_region`, `forest_mask`, `nepal_mask` |

### Lapse-rate downscaling — the methods contribution

```
T_1km = T_era5 + 0.0065 · (elev_era5 − elev_1km)
```

`elev_era5` is 1 km SRTM block-averaged onto the ERA5 cell and interpolated
back, so the correction carries **only** the terrain ERA5 cannot see. Measured
effect on 15 Apr 2021 versus plain bilinear: **−2.3 °C above 5000 m**, +0.3 °C in
the Terai.

Two supporting details: dewpoint uses **2 K/km**, not 6.5, because applying the
dry-air rate to both would leave relative humidity unchanged and discard the
signal being downscaled; and surface pressure is hypsometrically adjusted and
stored in **hPa**, since Pa exceeds the float16 maximum of 65504.

Cloud gaps in LST (up to 22% per composite) are closed along the time axis
before interpolation, so interpolation only ever runs between observed values.

### Forest mask

Burnable fraction (tree + shrub + grass) ≥ 0.25 **and** elevation ≤ 4500 m →
**126,622** of 168,064 Nepal cells, retaining **96% of fire pixel-days**.

The plan estimated ~80k. The thresholds that reach that number discard real
positives, so the looser mask is deliberate:

| Rule | Cells | Fire pixel-days kept |
|---|---|---|
| burnable ≥ 0.25, elev ≤ 4500 m | 126,622 | **96.1%** |
| burnable ≥ 0.5 | 118,563 | 93.1% |
| burnable ≥ 0.3, 50–3500 m | 110,748 | 94.6% |

The 4500 m treeline ceiling is free: it removes 3,839 high-alpine cells and
costs no positives at all.

### Verification

| Check | Result |
|---|---|
| Every layer shares shape / transform / CRS | **PASS** |
| Max NaN inside forest mask | **0.000%** (budget 5%) |
| Time axis matches `fire_daily.zarr` | **PASS** — both 1664 days, zero mismatched dates |
| Terai 2021 series physically sensible | **PASS** — `runs/cube/terai_cell_2021.png` |

Terai cell (28.553 N, 81.230 E, 179 m), Jan → Apr: `t2m_max` 20.0 → 34.1 °C,
`rh` 76.7 → 31.5%, `vpd` 0.38 → 2.29 kPa, `ndvi` 0.60 → 0.43, then a sharp
monsoon-onset reversal in mid-May. Textbook pre-monsoon curing.

```bash
python scripts/build_feature_cube.py                  # ~4 min
python scripts/build_feature_cube.py --years 2021     # single season
python scripts/plot_cube_check.py --year 2021
```

---

## Day 8 — Features and training table · Done

`data/cube/train_table.parquet` — **2,065,833 rows × 53 columns**
(44 features + 2 labels + 7 metadata), **149 MB**, builds in ~40 s.

| Group | Features |
|---|---|
| Weather (12) | `t2m_max` `t2m_min` `t2m` `d2m` `precip` `u10` `v10` `soil_water_l1` `surface_pressure` `rh` `vpd` `wind_speed` |
| Rolling / dryness (7) | `precip_7d` `precip_30d` `t2m_max_7d` `rh_min_7d` `wind_max_7d` `consecutive_dry_days` `days_since_rain` |
| Vegetation & thermal (6) | `ndvi` `evi` `ndvi_anomaly` `lst_day` `lst_night` `lst_diff` |
| Fire history (5) | `fire_clim` `days_since_fire` `fires_1yr` `fires_3yr` `fires_5yr` |
| Terrain (5) | `elevation` `slope` `aspect_sin` `aspect_cos` `twi` |
| Land cover (4) | `tree_frac` `shrub_frac` `grass_frac` `crop_frac` |
| Human (3) | `dist_road` `dist_settlement` `built_frac` |
| Temporal (2) | `doy_sin` `doy_cos` |

**Sampling.** All positive cell-days up to a 100k budget spread evenly over
year×month strata, plus negatives at **1:20** drawn within the same stratum, all
inside the forest mask. Capping positives thins only *training* — Day 10 scores
every forest cell straight off the cube, so the reported metrics are unaffected.
`--positive-cap 0` keeps all ~1.8M.

**Leakage controls** (each has a test):

| Control | Why |
|---|---|
| Labels look strictly forward; last H days of each season dropped | An incomplete lookahead window is not a zero |
| NDVI anomaly climatology is leave-one-year-out | Computed exactly by subtracting the year's own contribution from the running total |
| `norm_stats_v1.json` stores stats **per fold**, training years only | Whole-dataset statistics would leak the held-out season's distribution |
| `fire_clim` comes from MODIS **2003–2015** | Entirely outside the 2016–2026 label window |

### What the features say

| Feature | Cohen's d | Reading |
|---|---|---|
| `fire_clim` | **+0.84** | The out-of-sample historical prior is the single strongest signal |
| `vpd` | **+0.82** | Atmospheric dryness beats raw temperature |
| `rh` | **−0.76** | Matches the Nepal literature exactly |
| `fires_3yr` / `fires_5yr` / `fires_1yr` | +0.73 / +0.69 / +0.68 | Fire is strongly repeat-prone |
| `days_since_fire` | −0.58 | Recent burns re-burn |
| `soil_water_l1` | −0.48 | Named as a key driver by Hamal et al.; absent from v1 |

**The result that supports the thesis:** human-proximity features are nearly
flat (`dist_road` −0.06, `dist_settlement` +0.02), even though they dominate the
Nepal susceptibility literature. That is expected and worth stating plainly —
those studies predict *where* fires cluster over decades, and roads do not move,
so they carry almost no information about *which day* burns.

**Collinearity to disclose:** `t2m_max`–`t2m` r = 0.99, `surface_pressure`–`elevation`
r = 0.98, `fires_3yr`–`fires_5yr` r = 0.96. Harmless for LightGBM accuracy, but
SHAP will split credit between twins.

```bash
python scripts/build_train_table.py            # ~40 s
python scripts/plot_feature_diagnostics.py
```

---

## Standing rules

1. **Alignment.** 465 × 912, EPSG:4326, zero fire pixels outside the Nepal mask.
2. **No hardcoded years or paths** outside `configs/base.yaml`.
3. **Beat climatology PR-AUC 0.0416** or it does not ship.
4. **Always print the base rate** next to any metric, and lead with skill vs climatology.
5. **Evaluate on the full grid**, never on the sampled training table.
6. Never commit `data/`, `*.tif`, `*.zarr`, or `.map_key`.

---

## Known limitations (carry these into the report)

| Limitation | Impact |
|---|---|
| Wind is derived from daily-**mean** ERA5 u/v, so direction-averaging understates gusts | Treat as a relative signal, not observed wind speed |
| `physio_regions.tif` uses elevation bands as a proxy | Swap in official polygons if obtainable |
| `built_frac` substitutes for population density | WorldPop was never downloaded |
| Rolling windows truncate in early January | Only Jan–May was downloaded, so there is no December history |
| Nov–Dec fires (~8% of detections) are out of scope | Documented seasonal restriction |
| TWI is a slope-based approximation, not full flow routing | Adequate for tree models |

---

## Commands, end to end

```bash
source .prometheus-venv/bin/activate

python scripts/build_fire_labels.py        # labels     (~78 min first run)
python scripts/audit_gee_raw.py            # verify GEE downloads
python scripts/build_local_static.py --osm-dir data/raw/osm/nepal-free
python scripts/build_feature_cube.py       # feature cube (~4 min)
python scripts/build_train_table.py        # training table (~40 s)
python scripts/run_baselines.py            # baseline bar

python scripts/plot_cube_check.py --year 2021
python scripts/plot_feature_diagnostics.py

python -m pytest tests/ -q                 # 49 tests
ruff check src scripts tests
```

---

## Next — Day 9: LightGBM

Train with `scale_pos_weight` and early stopping; light search over
`num_leaves`, `min_data_in_leaf`, `learning_rate`, `feature_fraction`.

**Done when:** a single fold trains in under 2 minutes and beats climatology
PR-AUC on the full-grid evaluation.

| Then | |
|---|---|
| Day 10 | Full LOYO, SHAP, family ablations, per-region breakdown |
| Day 11 | Isotonic calibration, 7-day horizon, 5 risk classes, model card |
| Day 12 | Optional U-Net comparison on Kaggle |
