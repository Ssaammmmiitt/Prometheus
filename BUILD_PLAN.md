# Prometheus — 3-Week Build Plan (MacBook M4, free tier)

Revised for: local M4 development, free-tier only, 3-week timeline, Nepal pre-monsoon fire season.

---

# Part 1 — What we are building, in plain words

## The problem

Nepal burns every spring. Published work is unanimous on the shape of it: **~91% of Nepal's annual burned area occurs in March–May**, and **April alone accounts for roughly 62% of all fire detections** (Hamal et al. 2022, *Atmospheric Science Letters*; recent VIIRS clustering work covering 2021–2024). Your own FIRMS archive confirms this exactly — I measured 89.5% of detections in Jan–May with April at ~50%.

The drivers are also well established for Nepal specifically: **low pre-monsoon precipitation, low humidity, low soil moisture, and high temperature**, amplified in El Niño years by weakened westerly moisture transport. On top of that, **human factors dominate ignition** — proximity to roads and settlements shows up as a top predictor in essentially every Nepali study.

## The gap we are filling

Here is the important thing I found, and it defines your project's contribution.

Almost every published Nepal fire ML study is a **static susceptibility map**. The Rasuwa district Random Forest study (AUC 0.90), the Chure-Tarai-Madhesh fuzzy-AHP + RF study (AUC 0.95), the Terai Arc Landscape RF study, the Palpa district weighted-index map — all of them aggregate fire data over 10–20 years and produce a single map answering **"where in Nepal can fires occur?"**

That map does not change. It is the same in January as in April, the same in a drought year as a wet one. It cannot tell a district officer that **this week** is dangerous.

**We are building the other thing: a dynamic forecast.** Same country, same satellites, but the question is *"given today's weather and vegetation conditions, what is the probability that each 1 km cell in Nepal sees a fire tomorrow, and over the next week?"*

That is genuinely under-served for Nepal, it is what an operational system actually needs, and it is a defensible contribution for a Year-III project.

## How it works, end to end

```
NASA FIRMS satellite fire detections   ──┐
(VIIRS 375 m + MODIS 1 km, 2016–2025)    │
                                          ├──► daily 1 km training table
ERA5-Land daily weather                  │     (one row per forest cell per day)
MODIS NDVI + land surface temperature    │              │
SRTM terrain, land cover, roads,         │              ▼
settlements, fire history               ──┘     LightGBM  →  calibrated probability
                                                     │
                                                     ▼
                                          daily risk map (GeoTIFF)
                                                     │
                                          FastAPI  →  React map app
```

You train a model on ten past fire seasons. It learns which combinations of dryness, wind, fuel, terrain, and human proximity precede a fire detection. Then every day it scores all of Nepal and paints a risk map.

## Why each major choice

| Choice | Why |
|---|---|
| **Jan 1 – May 31 only** | You are right. Literature says 91% of burned area is Mar–May; your data says 89.5% of detections are Jan–May. Training on the monsoon adds ~120 days a year of trivially-negative samples that teach the model nothing except "it doesn't burn when it rains" — which it learns from January anyway. Cutting to 151 days/year makes the dataset 2.4× smaller and *better*. |
| **Daily timesteps** | This is the one place I push back — see Part 3. Short version: fire risk is a weather phenomenon, and daily costs you almost nothing extra here. |
| **1 km grid, 465 × 912** | You already have it, the Nepal mask is built, and it matches MODIS label precision. No reason to change. |
| **Forest and shrub pixels only** (~80k of 168k) | Fires happen in vegetation. Including snow, rock, water, and urban cells inflates your metrics with free negatives. Every Nepal study restricts to forest. This halves the dataset *and* makes the numbers honest. |
| **VIIRS + MODIS labels** | VIIRS is 375 m vs MODIS 1 km and detects the small understory fires typical of Nepal. Nepali researchers explicitly prefer it ("higher resolution and improved nighttime performance make it a superior tool... especially in smaller areas"). Using both gives ~3–4× more positives. |
| **LightGBM as the primary model** | On tabular geospatial problems it usually matches or beats CNNs, trains in minutes on an M4, gives you SHAP explanations for free, and the entire Nepal literature uses tree ensembles — so you are directly comparable. |
| **A CNN as the second model, not the first** | Deep learning is the interesting comparison, not the foundation. If it beats LightGBM, great result. If it doesn't, that's also a result. |
| **Leave-one-year-out CV** | Nepal's interannual fire variance is ~4.5× (2020: 2,436 detections vs 2021: 10,959). A single test year tells you nothing. Ten folds give you error bars. |

## What you will have at the end

1. A daily wildfire risk forecasting model for Nepal, evaluated honestly across ten fire seasons with error bars.
2. A results table comparing it against climatology, persistence, and a tree-ensemble baseline — the comparison almost nobody publishes.
3. A SHAP analysis showing which drivers matter in which physiographic region.
4. A FastAPI service producing a daily national risk map.
5. A React web app showing the map, district drill-downs, and a forecast-verification page.
6. A report whose claims are all backed by the results table.

---

# Part 2 — The data, chosen for Nepal and for a small laptop

Nepal is 885 km east–west and 193 km north–south — about 147,000 km². At 1 km that is ~168,000 grid cells, of which roughly **80,000 are forest or shrub**. This smallness is your advantage: the entire country fits in memory.

## The dataset, final

| # | Layer | Source | Native | We use | Why this one |
|---|---|---|---|---|---|
| **Labels** ||||||
| 1 | Active fire | FIRMS **VIIRS S-NPP + NOAA-20** | 375 m, 2×/day | daily binary per cell | Best available for small Nepali fires |
| 2 | Active fire | FIRMS **MODIS C6.1** | 1 km, 4×/day | daily binary per cell | Long record; union with VIIRS |
| **Weather — daily, the core signal** ||||||
| 3 | Temperature max/min/mean | ERA5-Land daily | ~9 km | daily | Universal fire driver |
| 4 | Dewpoint → **RH, VPD** | ERA5-Land daily | ~9 km | daily | Humidity is the top climatic driver in Nepal studies |
| 5 | **Wind u10, v10 → speed** | ERA5-Land daily | ~9 km | daily | Top-3 driver; **completely missing from your v1** |
| 6 | Precipitation | ERA5-Land daily | ~9 km | daily | Negative relationship with fire in every Nepal region |
| 7 | **Soil moisture L1** | ERA5-Land daily | ~9 km | daily | Named as a key driver by Hamal et al.; **missing from v1** |
| **Vegetation and thermal** ||||||
| 8 | NDVI, EVI | MODIS **MOD13Q1** | 250 m, 16-day | interpolated to daily | Fuel amount and cure state |
| 9 | LST day, night | MODIS **MOD11A2** | 1 km, 8-day | interpolated to daily | LST is in nearly every Nepal fire model |
| **Static — fuel, terrain, humans** ||||||
| 10 | Land cover | **ESA WorldCover v200** | 10 m | fractions per 1 km cell | Fuel type; also defines our forest mask |
| 11 | Elevation, slope, **aspect**, **TWI** | SRTM 30 m | 30 m | 1 km | Exact predictor set used by Rasuwa / CTML studies |
| 12 | **Distance to road** | OpenStreetMap Nepal | vector | 1 km raster | Top anthropogenic predictor in Nepal literature |
| 13 | **Distance to settlement**, population | WorldPop / OSM | 100 m | 1 km | "Proximity to settlements" was the #4 predictor in Rasuwa |
| 14 | Physiographic region | Terai / Chure / Middle Mtn / High Mtn | vector | categorical | Drivers differ by region — lets you analyse per region |
| **Derived — free signal, no download** ||||||
| 15 | Consecutive dry days, days since rain | from ERA5 precip | — | daily | The strongest cheap dryness feature |
| 16 | Rolling 7/30/90-day precip + anomaly | from ERA5 precip | — | daily | Captures drought build-up |
| 17 | Rolling 7-day max temp, min RH, max wind | from ERA5 | — | daily | Fire weather memory |
| 18 | NDVI anomaly vs day-of-year climatology | from NDVI | — | daily | Is vegetation drier than normal *for this date* |
| 19 | **Fire climatology per cell per day-of-year** | MODIS 2003–2015 points | — | static-ish | Cheap, powerful, and gives an out-of-sample prior |
| 20 | Days since last fire, fires in prior 1/3/5 yr | FIRMS history | — | daily | Fire is strongly autocorrelated |
| 21 | Day-of-year sin/cos | — | — | daily | Seasonal shape |

**~28 features.** That's the sweet spot: enough to cover every driver the Nepal literature identifies, few enough to train in minutes and explain in a report.

## What we deliberately skip, and why

| Skipped | Reason |
|---|---|
| Full Canadian FWI implementation | ~150 lines of fiddly code. Features 15–17 capture most of the same signal. Add later if time allows. |
| Sentinel-2 / Landsat | 10 m data over 10 years is hundreds of GB. Not feasible on a laptop, and not needed for 1 km forecasting. |
| MCD64A1 burned area | Only needed if you want to predict burned *area* rather than ignition. Out of scope for 3 weeks. |
| Himawari geostationary | Real-time detection, not prediction. Different project. |
| Foundation models (Prithvi/Clay) | Genuinely exciting, genuinely a 2-week detour. Note it as future work. |
| Nov–Dec season | ~8% of detections. Adds complexity (seasons spanning two calendar years) for little gain. Document as a limitation. |
| 2012–2015 predictor rasters | We use fire *points* from 2003 onward for the climatology feature (a 10 MB CSV), but only download predictor rasters for 2016–2025. Best of both. |

## Download budget — the trick that makes this laptop-feasible

**Export every layer at its native resolution, and resample locally.** Your v1 exported ERA5 at 1 km, which inflated a 41 × 82 grid into a 465 × 912 grid — 126× more bytes for zero extra information.

| Layer | Naive (1 km export) | Native export | Files |
|---|---|---|---|
| ERA5-Land daily, 9 bands, 151 d × 10 yr | ~23 GB | **~190 MB** (41 × 82 grid) | 50 monthly stacks |
| MOD11A2 LST, 8-day, 2 bands | 640 MB | **640 MB** (already 1 km) | 190 files |
| MOD13Q1 NDVI/EVI, 16-day | 340 MB | **340 MB** | 100 files |
| WorldCover, SRTM, static | — | **~120 MB** | 6 files |
| FIRMS CSVs | — | **~40 MB** | 14 files |
| **Total** | ~24 GB | **~1.3 GB** | ~360 files |

**1.3 GB.** That downloads over a normal connection in under an hour.

## Compute budget on an M4

| Stage | Size | Where | Time |
|---|---|---|---|
| Label cube (151 d × 10 yr × 465 × 912, uint8) | 640 MB | Local | 10 min |
| Feature cube, dynamic only, float16, masked to forest | **~4.5 GB** on disk | Local | 30 min |
| Training table: ~120k positives + 2M sampled negatives × 28 features | **~250 MB in RAM** | Local | 5 min |
| LightGBM train, one fold | — | **Local** | ~90 sec |
| Leave-one-year-out CV, 10 folds | — | **Local** | ~15 min |
| SHAP on 100k rows | — | **Local** | ~5 min |
| U-Net, 128×128 patches, ~20 epochs | — | **Kaggle T4** (or local MPS) | ~40 min |

Everything except the optional CNN runs comfortably on the M4. Peak RAM stays under 6 GB.

**Kaggle is pre-defined for exactly one step: Step 10, the CNN.** Free tier gives 30 GPU-hours/week, which is 40× what you need. Upload the training table and patch cache as a Kaggle Dataset (20 GB limit — you'll use ~5 GB) and run the notebook there. Nothing else needs cloud.

---

# Part 3 — Where I disagree, and the compromise

You said daily data probably isn't needed. Here is the case, and then the compromise.

**Why daily matters.** Your v1 got a test PR-AUC of 0.505 against a base rate of 0.279 — a 1.8× lift. The reason it wasn't 50× is that averaging weather over 16 days destroys the thing that distinguishes a fire day from a non-fire day. Within a single April fortnight in Nepal, some days have 15% afternoon humidity and 8 m/s wind, and some have 60% humidity and calm air. Fires happen on the first kind. A 16-day mean makes both look identical. The Nepal literature is explicit that low humidity, low soil moisture, and high wind drive fire — those are *daily* quantities.

**Why it's nearly free here.** Because we cut to Jan–May, cut to 10 years, cut to forest pixels, and export ERA5 at native resolution:

- 16-day version: 10 timesteps/yr × 10 yr = 100 steps → ~0.3 GB
- **Daily version: 151 days/yr × 10 yr = 1,510 steps → ~4.5 GB**

4.5 GB on disk, 250 MB in RAM for training, ~90 seconds per LightGBM fit. The GEE export effort is *identical* — you export monthly multi-band stacks either way, five per year, same number of clicks.

**The compromise — only weather is daily.** Everything else stays at its natural cadence and gets interpolated:

| Layer | Cadence |
|---|---|
| ERA5 weather | **daily** (it's the only thing that truly varies daily) |
| LST | 8-day, interpolated |
| NDVI/EVI | 16-day, interpolated |
| Terrain, land cover, roads, settlements | static, computed once |

So you are only *actually* handling daily data for 9 weather bands on a 41 × 82 grid. That is 190 MB. There is no version of this project where that is the expensive part.

**If you still want to avoid daily:** the fallback is a **3-day timestep** (50 steps/season instead of 151). It keeps most of the weather signal, cuts the cube to 1.5 GB, and everything else in this plan is unchanged. I'd take daily, but 3-day is defensible and I won't argue past that.

---

# Part 4 — The 3-week plan

18 working days. Each day has a **Goal**, **Do**, and **Done when**. `[LOCAL]` = M4. `[KAGGLE]` = free GPU. `[GEE]` = runs in Google's cloud, you just download.

---

## Week 1 — Data and your first working model

### Day 1 — Scaffold `[LOCAL]`

**Goal:** a clean package with one source of truth.

```bash
git checkout -b v2
mkdir legacy && git mv source GEE_code models legacy/
# .gitignore: data/, *.tif, *.zarr, runs/, venv/, node_modules/
git rm -r --cached data_raw data_processed data_processed_normalized

pip install uv && uv init . 
uv add numpy pandas xarray zarr rasterio rioxarray geopandas shapely \
       earthengine-api lightgbm scikit-learn shap matplotlib tqdm \
       pydantic-settings pyyaml requests
uv add --dev pytest ruff
```

Create `src/prometheus/` with `config.py`, `grid.py`, `data/`, `features/`, `models/`, `eval/`. Put every constant — years, months, paths, feature list, CV folds — in `configs/base.yaml`. `grid.py` holds the canonical transform, shape, CRS, and Nepal mask, defined once.

**Done when:** `python -c "from prometheus.config import cfg; print(cfg.years, cfg.season_months)"` prints `[2016..2025] [1,2,3,4,5]`.

---

### Day 2 — Fire labels `[LOCAL]`

**Goal:** the label cube. Labels first — they define the task.

1. Get a free FIRMS API key at `firms.modaps.eosdis.nasa.gov/api/`.
2. Download for bbox `80.018,26.347,88.201,30.447`:
   - `VIIRS_SNPP_SP` 2016–2025
   - `VIIRS_NOAA20_SP` 2018–2025
   - `MODIS_SP` **2003–2025** (the extra years are only for the climatology feature — points only, tiny)
3. Clean once, in `data/firms.py`: MODIS `confidence >= 50`; VIIRS `confidence in {nominal, high}`; drop `type != 0` (removes gas flares and volcanoes); dedupe on `(round(lat,4), round(lon,4), acq_date, satellite)`.
4. Rasterize Jan–May 2016–2025 to a daily `uint8` cube, **dilated 1 pixel** (3×3) to absorb geolocation error. Save `data/cube/fire_daily.zarr`.

**Done when:** a printed year × month detection table shows the April spike, total VIIRS+MODIS detections exceed **120,000**, and a test asserts the cube is zero outside the Nepal mask.

---

### Day 3 — Evaluation harness + climatology `[LOCAL]`

**Goal:** a working model and a scoring table by day three. **This is the most important day in the plan.**

1. `eval/metrics.py`: `pr_auc`, `roc_auc`, `brier`, **`skill_vs_climatology`**, **`top_k_capture(k=0.10)`** (what fraction of real fires land in the top 10% of predicted-risk area — the metric a fire officer actually cares about), `reliability_curve`, `ece`.
2. `eval/cv.py`: leave-one-year-out over 2016–2025, returns a per-year table with mean ± std.
3. **Climatology baseline:** for each cell and day-of-year, historical fire frequency from MODIS 2003–2015, smoothed ±7 days temporally and with a small spatial Gaussian.
4. **Persistence baseline:** fire in this cell or its 8 neighbours in the last 1/3/7 days.
5. Run both through CV, print the table, **save it**.

**Done when:** one command prints:

```
model         PR-AUC   ROC-AUC   Brier    top10%-capture
climatology   0.0XX    0.8XX     0.00XX   0.XX
persistence   0.0XX    0.7XX     0.00XX   0.XX
```

Every model from now on gets a row in this table. Anything that doesn't beat climatology doesn't ship.

---

### Days 4–5 — GEE exports `[GEE]`

**Goal:** kick off all downloads. These run in Google's cloud — start them and work on other things.

Write one script per family, all sharing the ROI from config. **Critical: use each layer's native scale.**

```javascript
// ERA5 — export at NATIVE ~9 km, one multi-band stack per month
Export.image.toDrive({
  image: monthlyStack,        // 9 bands × ~31 days
  scale: 11132,               // 0.1° — do NOT use 1000
  region: roi, crs: 'EPSG:4326', maxPixels: 1e13
});
```

| Script | Collection | Bands | Scale |
|---|---|---|---|
| `era5_daily.js` | `ECMWF/ERA5_LAND/DAILY_AGGR` | t2m_max, t2m_min, t2m, d2m, precip_sum, u10, v10, soil_water_L1, surface_pressure | **11132** |
| `lst_8day.js` | `MODIS/061/MOD11A2` | LST_Day_1km, LST_Night_1km | 1000 |
| `ndvi_16day.js` | `MODIS/061/MOD13Q1` | NDVI, EVI | 1000 |
| `static.js` | SRTM, WorldCover v200 | elev, slope, aspect, TWI, landcover fractions | 1000 |

While exports run, build the non-GEE static layers locally: download the Geofabrik Nepal OSM extract, compute distance-to-road and distance-to-settlement with `scipy.ndimage.distance_transform_edt`, and rasterize the four physiographic regions.

**Done when:** ~1.3 GB is in `data/raw/`, and a test asserts every static layer matches the canonical grid.

---

### Days 6–7 — Build the cube `[LOCAL]`

**Goal:** one aligned feature cube.

1. Resample ERA5 from 9 km to 1 km with **lapse-rate-corrected temperature** (`T_1km = T_era5 + 0.0065 * (elev_era5 - elev_1km)`). This is a few lines and it is a legitimate, citable methods contribution — far better than nearest-neighbour.
2. Interpolate LST (8-day) and NDVI (16-day) to daily, linearly in time.
3. **Build the forest mask** from WorldCover: keep tree cover, shrubland, grassland. Drop water, snow, bare rock, built-up. Expect ~80,000 of 168,000 cells.
4. Write `data/cube/features_daily.zarr`, float16, chunked `(time: 32, y: 256, x: 256)`.

**Done when:** all layers share one shape/transform/CRS (tested), no variable exceeds 5% NaN inside the forest mask, and a plotted 2021 time series of temperature, RH, and wind for one Terai cell looks physically sensible.

---

## Week 2 — Modeling

### Day 8 — Features and training table `[LOCAL]`

Compute derived features 15–21 from Part 2 (dry-day counters, rolling windows, anomalies, fire history, day-of-year encoding). Then build the tabular set: **all positive cell-days**, plus negatives sampled **1:20**, stratified by year and month, restricted to the forest mask.

Normalization stats come from **training folds only**, saved to a versioned JSON.

**Done when:** `train_table.parquet` exists (~2.1M rows × 28 columns, ~250 MB) and a correlation matrix plus per-feature fire/no-fire distributions are plotted. You should already be able to *see* which features matter.

---

### Day 9 — LightGBM `[LOCAL]`

Train with `scale_pos_weight` and early stopping. Light hyperparameter search — `num_leaves`, `min_data_in_leaf`, `learning_rate`, `feature_fraction`. 20–30 configs is plenty; each fit is ~90 seconds.

**Done when:** a single fold trains in under 2 minutes and beats climatology on PR-AUC.

---

### Day 10 — Full evaluation `[LOCAL]`

1. Leave-one-year-out CV, all 10 folds. Add the row to the results table with mean ± std.
2. **SHAP** — global importance plus dependence plots for the top 6 features. This drives your report's narrative.
3. **Family ablations** — drop weather / drop human / drop terrain / drop fire-history, report the PR-AUC delta for each. A strong results section on its own.
4. **Per-region breakdown** — Terai, Chure, Middle Mountains, High Mountains. The literature says drivers differ by region; check whether your model agrees.

**Done when:** you have the results table, a SHAP plot you can explain out loud, an ablation table, and a per-region table.

---

### Day 11 — Calibration and the 7-day horizon `[LOCAL]`

1. Isotonic calibration on a held-out inner fold; plot reliability before/after; report ECE.
2. Train a second model for the **next-7-days** target. Same pipeline, different label window.
3. Define 5 risk classes by quantile — Low / Moderate / High / Very High / Extreme — matching operational fire-danger convention.
4. Freeze both models, version them, write a short model card.

**Done when:** `predict(date) -> (465, 912) calibrated probability` runs in under a second, and the reliability diagram is close to the diagonal.

---

### Day 12 — CNN comparison `[KAGGLE]` *(optional but recommended)*

The only step that wants a GPU.

1. Cache 128 × 128 patches for the forest region to `.npy`, upload as a Kaggle Dataset (~5 GB, well under the 20 GB limit).
2. `smp.Unet(encoder_name="resnet18", encoder_weights="imagenet", in_channels=28, classes=1)`, Focal + Tversky loss (β = 0.7), AdamW at 3e-4, ~20 epochs. Roughly 40 minutes on a T4.
3. Download predictions, score them with **the same local CV harness**, add the row.

**Done when:** the U-Net row sits next to LightGBM in the results table — whichever wins. *If LightGBM wins, say so in the report.* On tabular geospatial problems it very often does, and reporting that honestly is a stronger result than hiding it.

> **Skip this day if you're behind.** LightGBM alone with rigorous evaluation is a complete project.

---

## Week 3 — System and report

### Day 13 — Inference pipeline `[LOCAL]`

`scripts/forecast.py --date YYYY-MM-DD` → fetches recent ERA5/MODIS, builds features, runs both models, writes `risk_{date}_h1.tif` and `risk_{date}_h7.tif` as **Cloud-Optimized GeoTIFF**, plus `districts_{date}.geojson` with mean and max risk per district.

Make it idempotent and backfillable. Backfill the 2024–2025 seasons so the app has history on day one. Add a **verification job** that scores yesterday's forecast against today's detections into `verification.csv`.

**Done when:** `make forecast DATE=2025-04-12` produces COGs that open correctly in QGIS.

---

### Day 14 — Backend `[LOCAL]`

```bash
uv add fastapi "uvicorn[standard]" titiler.core
```

| Route | Returns |
|---|---|
| `GET /api/risk/tiles/{z}/{x}/{y}.png?date=&horizon=` | raster tiles via TiTiler over your COGs |
| `GET /api/districts?date=&horizon=` | GeoJSON, 77 districts with risk class |
| `GET /api/districts/{id}/timeseries` | risk history |
| `GET /api/fires/active` | recent FIRMS detections |
| `GET /api/verification` | forecast-vs-observed accuracy |
| `GET /api/explain?lat=&lon=&date=` | top SHAP contributions for one cell |

**Skip PostGIS for now** — read COGs and GeoJSON straight from disk. A database adds a day of work and buys you nothing at this scale. Note it as future work.

**Done when:** `/docs` lists every endpoint and each returns real data.

---

### Day 15 — Frontend `[LOCAL]`

Reuse your existing React + Vite + Tailwind app. Keep **Leaflet** — at 1 km over Nepal it's fine, and swapping to MapLibre isn't worth a day right now.

Four views: national risk map with a date scrubber and 1-day/7-day toggle; district drill-down with a risk time series; historical fire explorer; and a **verification page** showing yesterday's forecast against today's detections. Use a colourblind-safe ramp — yellow→orange→red→purple, not red-green.

**Done when:** the app loads a real forecast from the API and the date scrubber animates.

---

### Days 16–18 — Report and polish

1. **Purge every flood reference** — §1.2, §1.3, §3.2.1, §3.2.2, §4.1 all contain pasted flood text from a different project.
2. **Verify every citation against a DOI.** Several current references appear fabricated, which is an integrity issue rather than a formatting one. Replace with the real Nepal literature — Hamal et al. 2022 (*Atmos. Sci. Lett.* 10.1002/asl.1096), the Rasuwa RF study (10.5194/egusphere-2025-2492), the CTML fuzzy-AHP + RF study (10.1080/19475705.2024.2436540), Matin et al. 2017 (*Int. J. Wildland Fire*), and Mishra et al. 2023 (*Fire Ecology*).
3. Rewrite Chapter 3 to describe the actual pipeline; rewrite Chapter 4 around the results table, SHAP plot, ablation table, per-region table, and reliability diagram.
4. **Frame the contribution correctly:** "existing Nepal fire ML produces static susceptibility maps; we produce a dynamic daily forecast." That sentence is your thesis.
5. **Always print base rates next to metrics**, and lead with skill-vs-climatology.
6. Fill in §4.3, §4.4, and Chapter 5 — still template boilerplate. Fix the two-supervisors inconsistency.

---

# Part 5 — Expectations and risk

## Realistic numbers

Set these in the report up front so nobody expects magic:

| Metric | Expect |
|---|---|
| ROC-AUC | **0.85 – 0.93** |
| PR-AUC | **0.08 – 0.25** against a ~0.3% base rate on forest cells — a **30–80× lift** |
| Top-10%-area fire capture | **55 – 75%** |
| Skill vs climatology | **+20 – 60%** relative PR-AUC improvement |

Anyone reporting pixel-level next-day PR-AUC above 0.5 is leaking. State that.

## What could go wrong

| Risk | Mitigation |
|---|---|
| GEE exports are slow or quota-limited | Start them on Day 4 and work on other things. Export monthly stacks, not daily files. |
| Download is bigger than expected | Drop to 2018–2025 (8 years). Costs one CV fold. |
| Feature engineering overruns | Cut features 18–20 (anomalies, fire history). Keep 15–17 — dry-day counters are the highest-value derived features. |
| Behind at Day 12 | Skip the CNN entirely. |
| Behind at Day 14 | Ship a static demo — pre-rendered risk maps in the React app, no live API. |

## The non-negotiables

**Never cut Days 2, 3, 8, or 10.** Labels, the evaluation harness with a climatology baseline, the feature table, and honest cross-validated results *are* the project. A well-evaluated LightGBM model beats five deep models evaluated badly, every time.

---

# Progress tracker

**Week 1 — data**
- [ ] Day 1 — Scaffold
- [ ] Day 2 — Fire labels (VIIRS + MODIS)
- [ ] Day 3 — Eval harness + climatology ★
- [ ] Days 4–5 — GEE exports + static layers
- [ ] Days 6–7 — Feature cube

**Week 2 — modeling**
- [ ] Day 8 — Features + training table
- [ ] Day 9 — LightGBM
- [ ] Day 10 — LOYO CV + SHAP + ablations ★
- [ ] Day 11 — Calibration + 7-day horizon
- [ ] Day 12 — U-Net on Kaggle *(optional)*

**Week 3 — system**
- [ ] Day 13 — Inference pipeline
- [ ] Day 14 — FastAPI backend
- [ ] Day 15 — Frontend
- [ ] Days 16–18 — Report

★ = do not skip
