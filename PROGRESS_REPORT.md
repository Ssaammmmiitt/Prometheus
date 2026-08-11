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

## Day 9 — LightGBM · Done

One leave-one-year-out fold (holdout **2021**), trained with `scale_pos_weight`
and early stopping, then scored on **every forest cell of every day** of the
held-out season — 18,993,300 pixel-days, 353,637 of them positive.

| Model | PR-AUC | Top-10 % capture |
|---|---|---|
| **LightGBM** | **0.2195** | **0.648** |
| Persistence | 0.0691 | 0.410 |
| Climatology | 0.0566 | 0.362 |

Skill vs climatology **+287 %**. Fold trains in **48 s**, comfortably inside the
two-minute gate; full-grid scoring adds ~4 min.

**The climatology bar moved, and that is deliberate.** Day 3 reported climatology
PR-AUC 0.0416 over all Nepal cells. The model only speaks for the forest mask,
where fires are concentrated and the base rate is 1.86 % rather than the
all-Nepal rate. Comparing across two different pixel populations would be
meaningless, so `evaluate_grid` recomputes climatology *and* persistence on the
model's own cells and days inside the same run. 0.0566 is that recomputed bar.
The 0.0416 figure stays valid for the all-Nepal population reported on Day 3.

**Early stopping does not touch the held-out year.** The most recent remaining
season is set aside as an inner validation fold. A random row split would have
leaked badly — neighbouring cells on the same day are very nearly the same sample.

**Hyperparameter search: 24 random configs** over `num_leaves`,
`min_data_in_leaf`, `learning_rate`, `feature_fraction`. Worth stating plainly
that it barely mattered — the entire search spanned 0.3695 to 0.3840 inner
PR-AUC, and full-grid PR-AUC was 0.2197 with the untuned defaults versus 0.2195
with the winner. Tuning is not the lever on this problem; features and the
evaluation protocol are. Winner (now the config default): `num_leaves` 127,
`min_data_in_leaf` 200, `learning_rate` 0.02, `feature_fraction` 0.65.

**Importance, with the collinearity caveat applied.** Gain is dominated by
`days_since_fire` (25.7 %) and `fire_clim` (14.1 %), then day-of-year encoding
and same-day weather. Because credit is split arbitrarily between correlated
twins, `feature_importance` reports a pooled `pair_gain_pct` alongside the raw
share: `fires_5yr` shows 3.1 % alone but the `fires_3yr`/`fires_5yr` pair holds
4.1 %, and `elevation` shows 2.1 % while the `elevation`/`surface_pressure` pair
holds 3.6 %. Read these at the group level. The same caveat governs the Day 11
SHAP work — a twin's individual attribution is not interpretable on its own.

```bash
python scripts/train_lightgbm.py --year 2021              # fold + full-grid eval, ~5 min
python scripts/train_lightgbm.py --year 2021 --search 24  # + random search, ~17 min
```

---

## Day 10 — Full evaluation · Done

Ten leave-one-year-out folds, each scored on every forest cell of every day of
its held-out season. **2016 trains but is never held out** — it is the first year
with fire labels, so nothing exists before it to build fire history from, and
scoring it would measure a model whose history features are blank by
construction rather than a model that failed.

### Results — leave-one-year-out

| Holdout | PR-AUC | Climatology | Persistence | Top-10 % | Base rate |
|---|---|---|---|---|---|
| 2017 | 0.1236 | 0.0271 | 0.0203 | 0.753 | 0.32 % |
| 2018 | 0.1441 | 0.0351 | 0.0272 | 0.723 | 0.56 % |
| 2019 | 0.2358 | 0.0441 | 0.0567 | 0.804 | 0.81 % |
| 2020 | 0.0730 | 0.0153 | 0.0220 | 0.686 | 0.26 % |
| 2021 | 0.2195 | 0.0566 | 0.0691 | 0.648 | 1.86 % |
| 2022 | 0.1343 | 0.0460 | 0.0345 | 0.715 | 0.49 % |
| 2023 | 0.1330 | 0.0458 | 0.0395 | 0.617 | 1.19 % |
| 2024 | 0.1900 | 0.0687 | 0.0591 | 0.652 | 1.57 % |
| 2025 | 0.1500 | 0.0458 | 0.0323 | 0.699 | 0.72 % |
| 2026 | 0.1443 | 0.0408 | 0.0309 | 0.728 | 0.53 % |
| **mean ± std** | **0.1548 ± 0.0481** | 0.0425 ± 0.0148 | 0.0392 ± 0.0168 | 0.703 ± 0.055 | 0.83 % |

**LightGBM 0.1548 ± 0.0481 vs climatology 0.0425 — +281 % skill, and it beats
both baselines in all ten folds.** The fold-to-fold spread is large and tracks
the base rate: 2020 is the worst fold (0.0730) and also the quietest fire year
(0.26 %), while 2019 and 2021 are the best and among the busiest. PR-AUC is
base-rate dependent, so that spread is mostly the seasons differing, not the
model being unstable — which is exactly why the climatology column is reported
next to every fold rather than as a single global number.

### Ablations — mean ΔPR-AUC when a family is removed

| Variant | Features | PR-AUC | Δ | Δ % |
|---|---|---|---|---|
| drop_terrain | 35 | 0.1552 ± 0.0467 | +0.0004 | +0.6 % |
| **full** | **44** | **0.1548 ± 0.0481** | — | — |
| drop_human | 41 | 0.1543 ± 0.0486 | −0.0005 | −0.4 % |
| drop_vegetation | 38 | 0.1521 ± 0.0471 | −0.0027 | −1.6 % |
| drop_weather | 25 | 0.1404 ± 0.0462 | −0.0143 | −9.3 % |
| drop_fire_history | 39 | 0.0695 ± 0.0294 | −0.0853 | **−56.2 %** |

Dropping weather also drops the rolling and dryness aggregates (`precip_30d`,
`consecutive_dry_days`, and the rest) — keeping those would not be an honest
weather ablation. Day-of-year encoding belongs to no family and is never
dropped, since it is calendar position rather than an observed driver.

**Fire history is the model.** Remove it and PR-AUC more than halves, to 0.0695
— barely above the 0.0425 climatology bar. Weather is the only other family that
matters, at −9.3 %. Terrain and human proximity are worth nothing measurable:
dropping either moves PR-AUC by less than 0.001, well inside the ±0.048
fold-to-fold spread, and dropping terrain nominally *improves* the mean. This is
the Day 8 correlation finding confirmed under a much stronger test — static
susceptibility layers describe where fires cluster over decades, not which day
burns.

Because it carries the model, it is worth being explicit that fire history does
not leak. `fire_clim` is built from MODIS **2003–2015**, entirely before the
2016–2026 modelling period. Within a season, `days_since_fire` and the
`fires_Nyr` counters use detections through day *t* to predict day *t+1*, which
is information genuinely in hand at forecast time. Both properties are pinned by
tests, since a −56 % ablation result is only as trustworthy as its causality.

### Per-region breakdown

| Region | Positives | Base rate | PR-AUC | Climatology | Skill |
|---|---|---|---|---|---|
| Terai | 323,038 | 1.29 % | 0.1468 ± 0.0411 | 0.0655 | +136 % |
| Chure | 699,612 | 1.58 % | 0.2398 ± 0.0530 | 0.0604 | +341 % |
| Middle Mountains | 521,499 | 0.59 % | 0.0934 ± 0.0546 | 0.0172 | +469 % |
| High Mountains | 37,798 | 0.11 % | 0.0483 ± 0.0344 | 0.0022 | +2151 % |

Raw PR-AUC falls with elevation, but so does the base rate, so skill runs the
other way — the model adds *most* where fire is rarest, because climatology has
almost nothing to say there. Chure is the best-served belt in absolute terms and
also the busiest: 700k positive pixel-days, more than any other region.

### Do drivers differ by region? Only mildly.

SHAP share by family, per region (%):

| Region | Weather | Fire history | Vegetation | Calendar | Terrain | Human |
|---|---|---|---|---|---|---|
| Terai | 31.3 | 25.8 | 17.4 | 14.8 | 8.5 | 2.3 |
| Chure | 30.0 | 26.8 | 18.2 | 14.7 | 7.9 | 2.3 |
| Middle Mountains | 31.3 | 24.0 | 14.6 | 18.9 | 8.7 | 2.5 |
| High Mountains | 26.1 | 20.4 | 21.1 | 18.7 | 11.5 | 2.3 |

The literature argues drivers differ substantially by physiographic belt. **The
model only partly agrees.** The ranking is identical in all four belts — weather
first, fire history second, human proximity last and negligible everywhere. What
does shift is a monotone gradient with elevation: fire history falls from 25.8 %
in the Terai to 20.4 % in the High Mountains, while terrain rises 8.5 → 11.5 %
and vegetation and thermal rise 17.4 → 21.1 %. Feature by feature, the clearest
regional signal is `lst_day`, which reaches 10 % in the High Mountains but does
not enter the top four anywhere else. So: a real gradient, but a shift in
emphasis rather than the different-drivers-per-region story the literature tells.

### SHAP — global

| Feature | Share | | Feature | Share |
|---|---|---|---|---|
| `doy_cos` | 10.7 % | | `rh` | 5.9 % |
| `fire_clim` | 9.2 % | | `lst_day` | 5.8 % |
| `days_since_fire` | 6.8 % | | `precip` | 5.2 % |
| `doy_sin` | 6.6 % | | `lst_diff` | 3.7 % |
| `fires_5yr` | 6.0 % | | `elevation` | 3.0 % |

The dependence plots (`runs/shap/dependence_2021.png`) are all physically
readable, which is the real test of the narrative:

- **`rh`** is cleanly monotone — contribution crosses from positive to negative
  at roughly 60 % relative humidity.
- **`days_since_fire`** spikes hard in the first ~50 days after a burn, then
  decays to a mild negative by ~250 days. Recent fire predicts more fire, which
  is the repeat-ignition behaviour of Nepal's spring burning season.
- **`fires_5yr`** rises steeply to about 8 prior fires and then saturates.
- **`doy_cos` / `doy_sin`** trace the March–April peak.

Two plotting decisions worth knowing: the never-burned sentinel
(`days_since_fire` = 9999, about 30 % of rows) is hidden from the plots only —
the model still uses it — because otherwise that single value owns the axis and
colour scale. And **collinear twins split their SHAP credit**, so `fires_5yr` at
6.0 % and `elevation` at 3.0 % each understate their pair: read `fires_3yr`/
`fires_5yr` (r = 0.96) and `elevation`/`surface_pressure` (r = 0.98) together.
The plots label the twin in the panel title.

```bash
python scripts/run_cv.py                    # 10 folds × 6 variants, ~2 h 20 m
python scripts/plot_shap.py --year 2021     # beeswarm + dependence grid
```

---

## Day 11 — Calibration, the 7-day horizon, frozen bundle `v1` · Done

Year split, chosen so nothing reported is anything the model touched:
**fit 2016–2023 · early stopping 2024 · isotonic calibration 2025 · reported on 2026.**

### Calibration was load-bearing, not cosmetic

Training uses a 1:20 negative downsample, so the raw booster score averages
**0.18 against a true rate of 0.53 %** — overconfident by a factor of thirty. The
raw model is a good *ranker* and a useless *probability*.

| Horizon | Base rate | mean raw → calibrated | ECE raw → calibrated | Brier raw → calibrated | PR-AUC |
|---|---|---|---|---|---|
| h1 | 0.529 % | 0.181 → 0.0052 | 0.1757 → **0.00018** | 0.0729 → 0.0047 | 0.1535 → 0.1538 |
| h7 | 2.603 % | 0.179 → 0.0236 | 0.1531 → **0.00281** | 0.0709 → 0.0215 | 0.2880 → 0.2875 |

ECE improves by roughly **950×** for h1 and **55×** for h7, and calibrated mean
probability lands within 2 % of the observed base rate. PR-AUC is unchanged to
within ±0.0005 — isotonic is monotone, so it cannot reorder; the tiny movement is
distinct raw scores collapsing onto a shared calibrated value.

Two decisions worth defending. **Isotonic is fitted on full-grid predictions, not
on the training table** — a fit on the 1:20 sample would learn to map onto the
sampled prevalence and stay just as wrong, only in a subtler way. And **ECE is
reported with equal-count bins**: at a sub-1 % base rate, equal-width bins drop
almost every pixel into the first bin and report a flatteringly small number. Both
are printed; the equal-count figure is the honest one.

The calibrator is stored as ~500 interpolation breakpoints in the manifest rather
than a pickled sklearn object, so the bundle stays readable and portable across
library versions, and applying it is a single `np.interp`.

![Reliability h1](runs/calibration/reliability_h1.png)

The raw curve sits one to two orders of magnitude below the diagonal across the
entire range; the calibrated curve tracks it from 10⁻⁵ to 10⁻¹. Axes are log–log
because a linear reliability diagram at this base rate is a dot in the corner.

### Risk classes, held-out 2026

Quantiles `[0.5, 0.75, 0.9, 0.95]` of the predicted distribution over the
calibration season. Classes are **relative** — Extreme means the top 5 % of
place-days, per operational fire-danger convention — not fixed probabilities.

| Class | % of grid | Observed rate (h1) | % of fires captured |
|---|---|---|---|
| Low | 52.3 % | 0.039 % | 3.9 % |
| Moderate | 28.6 % | 0.215 % | 11.7 % |
| High | 12.6 % | 0.809 % | 19.3 % |
| Very High | 3.4 % | 2.046 % | 13.3 % |
| **Extreme** | **3.0 %** | **9.038 %** | **51.9 %** |

Observed fire rate rises monotonically across all five classes and spans a **232×
range** from Low to Extreme, which is what makes the labels operationally
meaningful. Extreme covers 3 % of the grid and contains over half of all fires.
Class shares do not exactly match the target quantiles (52.3 % vs 50 % for Low)
because thresholds are fitted on 2025 and applied to 2026, and because heavy ties
at very low probabilities cannot be split.

### `predict(date) -> (465, 912)` in well under a second

| Horizon | Trees | Median | Max |
|---|---|---|---|
| h1 | 202 | **0.340 s** | 0.377 s |
| h7 | 389 | **0.602 s** | 0.653 s |

The first call of a season pays a ~12 s warm-up. Rolling windows, dry-day
counters, and fire-history state for any single day all depend on the whole season
to date, so there is no cheaper honest way to get one day in isolation: the season
is built once and cached (~3.4 GB resident), after which each date is an array
slice plus one booster pass.

**Getting under a second required a deliberate trade.** Inference cost is linear
in tree count — about 1.6 ms per tree over the mask — and the Day 9 winner
(learning rate 0.02) needed 553 trees for h1 and 797 for h7, putting h7 at 1.33 s
and failing the gate. Since the Day 9 search showed learning rate to be worth less
than fold-to-fold noise, the frozen models use 0.05. That halves the trees and
costs about 2 % relative PR-AUC (h7 0.2935 → 0.2875), comfortably inside the
±0.048 fold spread, for a 2.2× latency reduction.

### Frozen and versioned

`data/models/bundles/v1/` holds both boosters, both calibrators, the class
thresholds, the exact year split, and `MODEL_CARD.md`. Loading a bundle by
version is the only supported way to predict, so any score traces back to the
artefacts that produced it. The model card records intended use and the limits
that matter: detections are not ground truth, no prediction is made off the forest
mask, skill degrades where there is no fire history, the static human and terrain
layers add nothing measurable, and the raw booster margin must never be read as a
probability.

```bash
python scripts/build_model_bundle.py    # train h1+h7, calibrate, freeze, benchmark (~8 min)
python scripts/plot_calibration.py      # reliability diagrams
```

```python
from prometheus.models.predict import RiskPredictor
p = RiskPredictor("latest")
risk    = p.predict("2026-04-01", horizon=1)   # (465, 912) calibrated probability
classes = p.risk_classes("2026-04-01")         # 0-4, -1 off-mask
```

---

## Day 12 — CNN comparison (local MPS) · Done

Optional U-Net comparison, run **locally on the M4 Pro over Metal (MPS)** rather
than Kaggle. Same leave-one-year-out years as Day 10 (2016 warms history only;
2017–2026 held out), same forest-masked pixel population, same metrics.

| Holdout | U-Net PR-AUC | LightGBM PR-AUC | Climatology | Winner |
|---|---|---|---|---|
| 2017 | 0.0595 | 0.1236 | 0.0271 | LightGBM |
| 2018 | 0.0874 | 0.1441 | 0.0351 | LightGBM |
| 2019 | 0.1144 | 0.2358 | 0.0441 | LightGBM |
| 2020 | 0.0571 | 0.0730 | 0.0153 | LightGBM |
| 2021 | 0.1318 | 0.2195 | 0.0566 | LightGBM |
| 2022 | 0.1148 | 0.1343 | 0.0460 | LightGBM |
| 2023 | 0.0816 | 0.1330 | 0.0458 | LightGBM |
| 2024 | 0.1287 | 0.1900 | 0.0687 | LightGBM |
| 2025 | 0.1129 | 0.1500 | 0.0458 | LightGBM |
| 2026 | 0.1241 | 0.1443 | 0.0408 | LightGBM |
| **mean ± std** | **0.1012 ± 0.0278** | **0.1548 ± 0.0481** | 0.0425 | **LightGBM 10/10** |

U-Net still beats climatology on every fold (mean skill about **+150 %**), and
its mean top-10 % capture (0.706) is close to LightGBM’s (0.703). What it does
not do is win on PR-AUC: LightGBM is ahead by about **35 % relative**, and by
at least some margin in all ten seasons.

**Verdict for the report: LightGBM is the primary model.** That is the more
common outcome on tabular geospatial problems, and reporting it honestly is the
point of this day — not a consolation prize for skipping a GPU.

Setup used: `smp.Unet(resnet18, imagenet, in_channels=44)`, Focal + Tversky
(β = 0.7), AdamW 3e-4, 20 epochs × 250 batches, batch size 16, ~**11.4 min per
fold** on MPS, ~**2 h total** for all ten folds plus full-grid scoring. Season
feature stacks were cached once (~5.6 GB per year). MPS only appears as
available outside the Cursor sandbox; training must run in a normal terminal (or
with full permissions).

```bash
# outside the sandbox so Metal is visible
source .prometheus-venv/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1
python -c "from prometheus.cnn import stacks; stacks.build_all()"   # once
python -u scripts/train_unet.py --batch-size 16                  # 10 folds
```

---

## Standing rules

1. **Alignment.** 465 × 912, EPSG:4326, zero fire pixels outside the Nepal mask.
2. **No hardcoded years or paths** outside `configs/base.yaml`.
3. **Beat climatology** or it does not ship — recomputed on the *same* pixel
   population as the model (0.0416 all-Nepal, 0.0566 forest mask), never across
   two different populations.
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
| 2016 is never a holdout fold — no prior year exists to build fire history from | 10 folds, not 11 |
| PR-AUC varies with each season's base rate | Always read a fold next to its own climatology column |
| Raw booster scores are inflated ~30× by the 1:20 downsample | Only the calibrated output is a probability |
| First `predict` of a season costs ~12 s and ~3.4 GB | Season features are cached; later dates are sub-second |

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
| Day 12 | U-Net LOYO on local MPS — LightGBM wins 10/10 folds |
