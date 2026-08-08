# Prometheus — Technical Audit and Rebuild Roadmap

Audit date: 2026-08-09
Scope: full repository (`GEE_code/`, `source/`, `models/`, `frontend/`, `data_raw/`, `data_processed/`, `data_processed_normalized/`, `reports/`, `report.md`)

---

## 0. Verdict in one paragraph

The project is **feasible and worth continuing**, but it is currently solving the wrong problem with the wrong data cadence, and the evaluation does not support the claims in the report. Fire *label* data for Nepal is abundant and free — that is not your bottleneck. Your bottleneck is that you compressed everything to a **16-day cadence**, which leaves you with **80 total timesteps across 8 years**, and you defined the target as *"will any fire occur somewhere in a 32 km × 32 km box in the next 16 days"* — a question that is **27–40% "yes" by default during fire season** and has essentially no operational value. The model does beat chance (test PR-AUC 0.505 against a 0.279 base rate, a real ~1.8× lift), but that lift is small and the headline framing in `report.md` is not defensible. The fix is not a bigger model. The fix is redefining the task to **daily, 1–7 day lead, 1 km grid-cell ignition probability**, adding the ~10 predictors you are missing that actually drive fire (wind, soil moisture, land cover, human proximity, fire weather index), and building an honest evaluation protocol against real baselines. That version of this project is genuinely strong, publishable at student-conference level, and a superb portfolio piece.

---

## 1. What exists right now — factual inventory

### 1.1 Rasters on disk

| Directory | Variables | Files | Grid | CRS |
|---|---|---|---|---|
| `data_raw/` | ndvi16, temp16, precip16, rh16, vpd16, fire16, static, mask | ~82 per var | 465 × 912 | EPSG:4326, ~0.00898° (~1 km) |
| `data_processed/` | same, Nepal-masked (nodata −9999; fire uses 255) | 80 per var | 465 × 912 | EPSG:4326 |
| `data_processed_normalized/` | same minus fire, min–max scaled to [0,1] | 80–81 per var | 465 × 912 | EPSG:4326 |

Static layers: `elevation_static_srtm.tif`, `slope_static_srtm.tif`. Nepal mask: 168,064 valid 1 km pixels.

### 1.2 The temporal coverage problem, stated precisely

Every variable has exactly **10 composites per year** (2024 has 9), dated:

```
20180101, 20180117, 20180202, 20180218, 20180306,
20180322, 20180407, 20180423, 20180509, 20180525
```

That is Jan 1 through the window starting May 25. Across 2018–2025 that is **80 timesteps in total**. After building sequences with T=3 inputs and restricting the label month to March–May, you get roughly **6 usable windows per year, 48 in total**. Every one of your 28,486 "samples" is a spatial re-slice of those 48 moments in time. This is the single most important fact about the project: **you have 48 independent temporal observations, not 28,486 samples.** Everything downstream — the optimistic validation, the year-to-year instability, the inability to learn weather-driven dynamics — follows from this.

### 1.3 Fire label statistics (measured, not estimated)

From `GEE_code/Fire-Data/fire_archive_M-C61_701611.csv` (NASA FIRMS, MODIS Collection 6.1, Nepal ROI, 2018-01-01 → 2025-07-24):

- **53,348** raw detections; **41,232** after `confidence >= 50`.
- Detections per year: 2018: 5,780 · 2019: 7,301 · **2020: 2,436** · **2021: 10,959** · 2022: 4,354 · 2023: 6,665 · 2024: 10,794 · 2025: 5,059
- Detections per month (all years pooled): Jan 2,363 · Feb 3,233 · Mar 9,210 · **Apr 26,524** · May 6,418 · Jun 609 · Jul–Sep 103 · Oct 457 · **Nov 2,327** · **Dec 2,104**
- **89.5%** of all detections fall in Jan–May. April alone is **~50%** of the annual total.
- Rasterized to the 1 km / 16-day grid: **20,595 fire pixels out of 13.7 million valid pixel-timesteps = 0.150% positive rate**. Per timestep: min 1, median 107, max 2,318 fire pixels.

Three consequences you must design around:

1. **Interannual variance is enormous** — 2021 has 4.5× the fires of 2020. With one validation year and one test year, your metric is dominated by which year you happened to draw. 2024 (val) is a heavy year, 2025 (test) is a light year. This alone explains most of the "val → test collapse."
2. **You are discarding a real second season.** Nov–Dec carries ~4,400 detections that your Jan–May window excludes entirely. Oct–Dec is a genuine post-monsoon burning period in the Terai and mid-hills.
3. **Class imbalance at pixel level is 1:665**, not the ~1:2 your patch labels suggest. The two framings are not comparable and the report conflates them.

### 1.4 Code inventory

- `source/` — 30+ standalone scripts, no package structure, no config system, no tests, no CLI. Includes `train_convlstm.py` (pixel segmentation, 6 channels), `train_3dcnn_baseline.py` (pixel segmentation, 7 channels), `train_fire_unet.py` (legacy, reads NPZ that the rest of the pipeline no longer produces).
- `models/prometheus-2.ipynb` — the actual production ConvLSTM (patch-level classification, 8 channels). This is the only pipeline with recorded results. The notebook itself ends in a `KeyboardInterrupt`; the reported numbers come from saved artifacts in `source/runs/patch_convlstm_20260111_145930/`.
- `models/prometheus-3d-cnn.ipynb` — pixel segmentation 3D-CNN.
- `frontend/` — React 19 + Vite + Leaflet + Tailwind, ~800 lines. **There is no backend anywhere in the repository.** No FastAPI, no Flask, no inference server. The frontend cannot be connected to the model as things stand.
- `GEE_code/` — six per-variable JS export scripts plus fire rasterization, and three mutually inconsistent FIRMS cleaning scripts producing seven overlapping CSVs.

---

## 2. Root-cause analysis — why the results are weak

I have grouped these by severity. The first four are the ones that actually determine whether this project works.

### CRITICAL

**C1 — The prediction target is not a useful question.**
`build_dataset_index_p32_s16.py` labels a patch `has_fire = np.any(fire_patch == 1)` over a 32 × 32 pixel patch. At 1 km resolution that is a **1,024 km² area**, roughly the size of an average Nepali district, over a **16-day window**, during peak fire season. Of course it is positive 30% of the time. Measured base rates:

| Split | Years | n | Base rate |
|---|---|---|---|
| train | 2018–2023 | 21,117 | 0.304 |
| val | 2024 | 3,673 | 0.405 |
| test | 2025 | 3,696 | 0.279 |

A model that always outputs "fire" scores precision 0.279 / recall 1.00 / **F1 0.436** on the test set. Your model scores F1 0.581. That is a real improvement, but it is not a wildfire prediction system — it is a mildly informed seasonal prior. No fire officer can act on "somewhere in this district, sometime in the next two weeks."

**C2 — The temporal cadence throws away 94% of the available signal.**
Fire risk is driven by *weather on the day*: wind, relative humidity in the afternoon, days since last rain, temperature spike. ERA5-Land is **daily** (hourly, in fact). You averaged it into 16-day means before the model ever saw it, which destroys exactly the high-frequency variation that separates a fire day from a non-fire day in the same fortnight. Averaging RH over 16 days makes a dangerous 20%-RH afternoon look identical to a benign one. This is the largest single source of lost predictive power in the project.

**C3 — The dominant fire predictors are absent.**
Your feature set is NDVI, temperature, precipitation, RH, VPD, elevation, slope. Missing, in rough order of importance for Nepal:

- **Wind speed and direction** (ERA5-Land `u_component_of_wind_10m`, `v_component_of_wind_10m`) — a top-three driver, completely absent.
- **Land cover / fuel type** (ESA WorldCover 10 m, or MODIS MCD12Q1) — you have no idea whether a pixel is forest, cropland, or rock. A model cannot distinguish "won't burn" from "didn't burn."
- **Soil moisture** (ERA5-Land `volumetric_soil_water_layer_1`) — the standard proxy for dead fuel moisture.
- **Human ignition proxies** — distance to roads (OSM), distance to settlements (WorldPop / GHSL), distance to forest edge, population density. **Roughly 90% of Nepali wildfires are human-caused** (agricultural residue burning, grazing-land management, NTFP collection). You have modeled only the biophysical half of the problem.
- **Fire Weather Index components** (Copernicus CEMS / ERA5-derived FWI, available in GEE) — FFMC, DMC, DC, ISI, BUI, FWI. These are decades of fire-science feature engineering, free, and they belong in your baseline.
- **Antecedent dryness** — consecutive dry days, days since last ≥1 mm rain, SPI-1/SPI-3, KBDI.
- **Terrain derivatives** — aspect (south-facing slopes burn more in the Himalaya), TPI, TWI, curvature. You have elevation and slope only.
- **Fire history** — time since last burn at that pixel, count of fires in the previous N years. Strongly autocorrelated and trivially cheap.

**C4 — Resolution laundering.**
Every GEE export is forced to `scale: 1000`. ERA5-Land is natively ~9 km and is being **upsampled 9×** to 1 km, creating 81 near-identical pixels per real observation. The model sees spatial detail in the climate channels that does not exist. Conversely MOD13Q1 NDVI is natively 250 m and is **downsampled 4×**, throwing away real detail. You are simultaneously inventing fake resolution and discarding real resolution. The honest approach is to keep each variable at its native grid and let the model handle multi-resolution fusion, or to downscale climate physically (lapse-rate correction against the DEM) rather than by nearest-neighbour resampling.

### HIGH

**H1 — Evaluation protocol cannot support the report's claims.**
- No baselines at all. There is no comparison against climatology, persistence, a raw FWI threshold, or logistic regression. Without these, PR-AUC 0.505 is uninterpretable.
- Patches use `stride 16` on `patch 32`, so adjacent samples share 75% of their pixels. Metrics are computed on heavily autocorrelated samples; confidence intervals are far wider than the numbers suggest.
- The decision threshold is re-tuned on validation *every epoch* and the best epoch is selected on validation PR-AUC — a double-dip that inflates validation numbers.
- One validation year and one test year, in a domain with 4.5× interannual variance. There are no error bars.

One point in your favour that the report misses: the val→test drop from 0.718 to 0.505 is **not** primarily overfitting. Normalizing by base rate, val lift is 0.718/0.405 = **1.77×** and test lift is 0.505/0.279 = **1.81×**. The model generalizes essentially identically across years; the raw PR-AUC just looks worse because 2025 was a lighter fire year. Stating this correctly is a genuine strength — it shows you understand your metric.

**H2 — Four pipelines, four incompatible definitions.**

| Pipeline | Task | Channels | Split (train/val/test) |
|---|---|---|---|
| `prometheus-2.ipynb` ConvLSTM | patch binary | 8 (7 + mask) | 2018–23 / 2024 / 2025 |
| `train_convlstm.py` | pixel segmentation | 6 (no VPD) | 2018–22 / 2023 / 2024 |
| `train_3dcnn_baseline.py` | pixel segmentation | 7 | 2018–23 / 2024 / 2025 |
| `train_fire_unet.py` | pixel segmentation | 7 (from stale NPZ) | random 25% split |

Nothing here is comparable to anything else. `train_fire_unet.py` additionally does a **random** validation split with no spatial blocking, which leaks overlapping patches between train and val.

**H3 — Point labels used as segmentation ground truth.**
`make_fire_labels.py` snaps each FIRMS lat/lon to a single 1 km pixel with **no buffer**. FIRMS geolocation error is on the order of a pixel, and MODIS pixel size grows to several km off-nadir. Training pixel-wise Dice/IoU against single-pixel ground truth means the model is penalized for being off by one pixel on a fundamentally imprecise label. Either buffer/dilate the labels (~1–2 km) or use MCD64A1 burned-area polygons as the segmentation target.

**H4 — No backend exists.**
The report describes a FastAPI inference service and an interactive ROI-selection map. Neither exists. The frontend is a standalone React app with no API layer. This is the largest gap between the report and reality.

### MEDIUM

- **M1** — `source/train_convlstm.py:163` contains a stray `\` line-continuation that breaks the file.
- **M2** — `organise_ndvi_by_year.py:7` points at `data_raw/ndvi_16`; the real folder is `data_raw/ndvi16`.
- **M3** — `organize_fire_by_year.py` uses `NDVI_ROOT` variables and prints "No NDVI files" for fire data.
- **M4** — `fire_checker.py:98` hardcodes `"aligned": True` for fire rasters without actually checking grid alignment.
- **M5** — `normalize_with_reports.py` has 479 lines of dead commented-out code with a *different* `TRAIN_YEARS` list than the live code at line 503.
- **M6** — Nodata conventions are inconsistent: environmental vars use `−9999`, fire masking uses `255`, `make_fire_labels.py` writes `0` for both "no fire" and "outside Nepal." A loader cannot distinguish masked from negative.
- **M7** — The last composite of each year is temporally asymmetric: ERA5 scripts filter to May 31 but the May 25 window requests 16 days through June 10, so only 7 days exist. NDVI's May 25 composite covers the full 16 days. Variables disagree on what the last timestep means.
- **M8** — Seven FIRMS CSVs with three different cleaning scripts. `firms_clean_2018_2025.csv` — the one actually consumed by `make_fire_labels.py` — has **no generator script in the repo**. Its provenance is unreproducible.
- **M9** — `dataset_loader.py:40` reads the entire 465 × 912 band on every `__getitem__` to extract a 32 × 32 patch.
- **M10** — Duplicate Aqua/Terra detections of the same fire are retained; dedup key excludes satellite.

### REPORT ISSUES (`report.md`)

- **Flood project text copy-pasted throughout.** §1.2 objective says "predict potential flood risks." §1.3 says "actionable flood information." §3.2.1 opens "The implementation of the flood detection and prediction system." §3.2.2 references the "Sen1Floods11 Essentials dataset." §4.1 headline feature is "Flood Prediction Module ... using BiLSTM." This will be the first thing an examiner notices.
- **Datasets cited do not match the code.** Report says MOD13**A2**, CHIRPS/GPM, MOD11A2. Code uses MOD13**Q1** and ERA5-Land for temp/precip/RH/VPD.
- **VIIRS is claimed (§3.2.1) but never used.** All fire data is MODIS M-C61.
- **Contradictory metrics.** §3.2.1 states AUC 0.718 and val loss 0.157; §4.2.1 states AUC 0.505 and loss 0.15; Table 3.1 states val loss 0.0930. Three different numbers for the same run.
- **Confusion matrix does not match the stated recall.** TP 926 / FN 105 gives recall 0.898 ✓, but TP+FP+TN+FN = 3,696 ✓ with 1,031 positives = base rate 0.279 ✓. The numbers are internally consistent — but the text calls PR-AUC 0.505 "close to random performance," when random is 0.279. Undersells your own result.
- §4.2.3 and §4.2.4 are duplicate headings with duplicate body text.
- §4.3, §4.4, Ch. 5 are **unfilled template boilerplate** ("Guidelines by project type: ...", "Students should also explain ...").
- Two supervisors named (Mr. Suman Shrestha on the title page, Dr. Rabindra Bista on the certificate).
- References appear fabricated — "Prapas et al. (2023), *Fire Safety Journal* 134:104634" does not correspond to the real Prapas et al. wildfire-danger work (which is a Copernicus/NeurIPS-workshop and *Nat. Hazards Earth Syst. Sci.* line of publication). **Verify every citation against DOI before submission.** Fabricated references are an academic-integrity issue, not a formatting one.

---

## 3. Fire data availability in Nepal — the definitive answer

You asked specifically about this. Short version: **fire labels are the most abundant part of your data. Stop worrying about them and start worrying about predictors and cadence.** Here is everything available, ranked by what I would actually use.

### 3.1 Tier 1 — use these

| Product | Resolution | Revisit | Coverage | Access | Why |
|---|---|---|---|---|---|
| **VIIRS S-NPP + NOAA-20 + NOAA-21 active fire (VNP14IMGML / VJ114IMGML)** | **375 m** | 2–4×/day | 2012–present (NOAA-20 from 2018) | NASA FIRMS API/CSV download | **The single biggest free upgrade available to you.** ~3–5× more detections than MODIS, 375 m instead of 1 km, detects smaller/cooler fires. Nepal's fires are mostly small understory burns that MODIS misses. |
| **MODIS Terra+Aqua C6.1 (MCD14ML)** | 1 km | 2–4×/day | 2000–present | NASA FIRMS (you already have this) | Keep it — the long record back to 2000 lets you build fire-history features and gives 26 years for climatology. **You are only using 8 of 26 available years.** |
| **MCD64A1 Burned Area** | 500 m | monthly | 2000–present | GEE `MODIS/061/MCD64A1` | Gives actual *burned area polygons* + burn date, not just hotspots. This is the correct target if you want segmentation rather than ignition detection. |
| **FireCCI51 Burned Area** | 250 m | monthly | 2001–2020 | GEE `ESA/CCI/FireCCI/5_1` | Finer burned area; good for validating MCD64A1. |

### 3.2 Tier 2 — valuable additions

| Product | Resolution | Use |
|---|---|---|
| **Sentinel-2 L2A → dNBR/NBR** | 10–20 m, 5-day | High-resolution burn-scar delineation for a case-study chapter. GEE `COPERNICUS/S2_SR_HARMONIZED`. |
| **Himawari-9 AHI** | 2 km, **10-minute** | Geostationary, covers Nepal. Sub-hourly detection → enables genuine near-real-time alerting rather than 12-hour-latency polar orbiters. |
| **Landsat 8/9 thermal** | 30/100 m, 16-day | Burn severity mapping. |
| **GEE `FIRMS` collection** | 1 km daily | Pre-rasterized MODIS daily fire mask; convenient but MODIS-only. |

### 3.3 Nepal-specific and institutional sources

- **ICIMOD SERVIR-HKH Forest Fire Detection and Monitoring System** (`bipad.gov.np`, `rds.icimod.org`) — operational MODIS/VIIRS alerts for Nepal, plus district-level historical summaries. Worth citing as the incumbent system your work complements.
- **BIPAD Portal (NDRRMA)** — Nepal's official disaster incident database. Contains **reported** fire incidents with damage, casualties, and response. Extremely valuable as an independent validation set: satellite hotspots ≠ incidents that mattered.
- **DFRS / FRTC forest cover and forest type maps** — the fuel-type layer for Nepal specifically, better than global land cover.
- **DHM (Department of Hydrology and Meteorology)** station data — ~280 stations for validating your ERA5 downscaling. Access is by request but a student request is usually granted.
- **Nepal MoFE annual fire reports** — ground truth on cause attribution (confirms the ~90% anthropogenic figure).

### 3.4 What this means for your data budget

If you move to **daily, VIIRS + MODIS, 2012–2025, full calendar year**:

| | Current | Proposed |
|---|---|---|
| Timesteps | 80 | ~5,100 (14 years × 365) |
| Fire detections | 41,232 (MODIS, conf≥50, 8 yr) | ~250,000+ (VIIRS+MODIS, 14 yr) |
| Label resolution | 1 km | 375 m |
| Season coverage | Jan–May (89.5% of fires) | full year (100%, incl. Nov–Dec season) |
| Independent temporal samples | ~48 | ~5,100 |

That is a **100× increase in independent temporal observations** for zero additional cost, using data you can download today. This is the change that makes the project work.

---

## 4. The rebuild — step by step

I have phased this so that each phase produces something demonstrable. Do not skip Phase 3.

### Phase 0 — Foundation (3–5 days)

Goal: make the repo a project rather than a folder of scripts.

```
prometheus/
├── pyproject.toml              # uv or poetry; pin everything
├── configs/                    # Hydra or pydantic-settings YAML
│   ├── data/
│   ├── model/
│   └── experiment/
├── src/prometheus/
│   ├── data/                   # ingestion, gee, firms, rasterio io
│   ├── features/               # feature engineering, one module per family
│   ├── datasets/               # torch Dataset + samplers
│   ├── models/                 # architectures
│   ├── training/               # loops, losses, metrics, calibration
│   ├── eval/                   # baselines, spatial CV, reporting
│   └── serving/                # FastAPI app, inference
├── notebooks/                  # exploration only, never authoritative
├── tests/
├── dvc.yaml                    # or a Makefile DAG
└── docker/
```

Concrete actions:
1. Delete `train_fire_unet.py`, `dataset_loader.py`, `verify_rh_scale.py`, the four redundant FIRMS CSVs, and the 479 dead lines in `normalize_with_reports.py`. Dead code that contradicts live code is worse than no code.
2. One config object defines `TRAIN_YEARS / VAL_YEARS / TEST_YEARS / CHANNELS / PATCH / STRIDE`. Every script imports it. Zero hardcoded splits.
3. Add `.gitattributes` with DVC or Git-LFS for rasters — you currently have ~500 MB of GeoTIFFs tracked in git, which is why your `git status` is 200 lines of modified `.tif` files.
4. `pytest` with at least: a grid-alignment test, a no-temporal-leakage test (assert `max(input_date) < label_window_start`), a normalization-range test, and a label-count regression test.
5. Pre-commit with ruff + black.

### Phase 1 — Redefine the task (decide before writing code)

Commit to this specification and put it in the report:

> **Given daily environmental and anthropogenic conditions up to and including day _d_, predict for each 1 km grid cell in Nepal the probability of at least one active-fire detection occurring in that cell during day _d+1_ (and separately _d+1..d+3_, _d+1..d+7_).**

Why each choice:
- **Daily** — matches how fire risk actually varies and how decisions are actually made.
- **1 km cell** — matches MODIS label precision; you can go to 375 m once VIIRS is in.
- **1 / 3 / 7-day horizons** — 1-day is the operational alert, 7-day is the planning product. Reporting all three shows you understand the accuracy/lead-time tradeoff.
- **Probability, calibrated** — not a binary. Fire management runs on graded risk levels, exactly like the existing 5-class fire danger ratings.

Also produce the **district-aggregated daily risk** (77 districts) as a second output head. It is the product a Nepali official would actually consume, and it is far easier to evaluate meaningfully.

### Phase 2 — Rebuild the data layer (2–3 weeks)

**2a. Labels.** Download VIIRS (SNPP + NOAA-20) and MODIS archives for Nepal 2012–2025 from the FIRMS API. Build a daily 1 km binary raster stack. Buffer each detection to a 1-pixel radius to absorb geolocation error, and keep a separate unbuffered version for ablation. Store as a single Zarr array `(time, y, x)` — not 5,000 GeoTIFFs.

**2b. Predictors.** Export daily, at native resolution where possible:

| Family | Source | Variables |
|---|---|---|
| Weather (daily) | ERA5-Land `ECMWF/ERA5_LAND/DAILY_AGGR` | t2m max/min/mean, d2m → RH & VPD, **u10, v10 → wind speed + direction**, total_precipitation_sum, **volumetric_soil_water_layer_1**, surface_pressure |
| Fire weather | Copernicus CEMS FWI (or compute FFMC/DMC/DC/ISI/BUI/FWI yourself from ERA5 — this is a great, citable contribution) | 6 indices |
| Vegetation | MOD13Q1 NDVI/EVI (250 m, 16-day, interpolate to daily), MOD09/MCD43 → **NDMI/NDWI** | NDVI, EVI, NDMI, NDVI anomaly vs 2000–2020 climatology |
| Thermal | MOD11A1 daily LST **day and night** | LST_day, LST_night, **day−night difference** (a strong dryness proxy) |
| Fuel | ESA WorldCover 10 m, MCD12Q1, DFRS forest type | land cover class (one-hot), tree cover fraction, forest type |
| Terrain | SRTM 30 m | elevation, slope, **aspect (sin/cos)**, TPI, TWI, curvature |
| Human | OSM, WorldPop, GHSL | dist. to road, dist. to settlement, dist. to forest edge, population density, cropland fraction |
| Antecedent | derived | consecutive dry days, days since ≥1 mm rain, 7/30/90-day rainfall anomaly, SPI-1/3, KBDI |
| Fire history | MODIS 2000–2011 (out-of-sample years) | days since last fire, fire count in prior 1/3/5 yr, seasonal climatology of fire at that cell |
| Calendar | derived | day-of-year sin/cos, and **Nepali festival/agricultural calendar flags** — pre-monsoon crop-residue burning is calendar-driven and this is a genuinely novel, locally-grounded feature |

Roughly 45–55 channels instead of 7.

**2c. Downscaling, done honestly.** Do not bilinearly upsample ERA5 to 1 km and call it 1 km data. Either (a) keep climate at 9 km as a separate coarse input branch and fuse in the model, or (b) apply physical downscaling — lapse-rate temperature correction using the 30 m DEM, terrain-shading for radiation. Option (b) is a defensible methods contribution and looks excellent in a report.

**2d. Storage.** Zarr + xarray, chunked `(time: 32, y: 256, x: 256)`. Cloud-Optimized GeoTIFF for anything the web app serves. Your current one-file-per-timestep-per-variable layout will not survive a 5,000-timestep dataset.

**2e. Reproducibility.** Every download step becomes a DVC stage or a Prefect flow. `dvc repro` should rebuild the entire dataset from nothing.

### Phase 3 — Baselines first (1 week) — DO NOT SKIP

Before any deep learning, produce a table of these on the exact same evaluation protocol:

| # | Baseline | What it proves |
|---|---|---|
| 0 | Always-positive / always-negative | Floor |
| 1 | **Climatology** — P(fire \| cell, day-of-year) from 2000–2017 | The seasonal+spatial prior. Very hard to beat. Most published "wildfire AI" papers quietly fail to beat this. |
| 2 | **Persistence** — fire yesterday in this cell or its 8 neighbours | Autocorrelation floor |
| 3 | **FWI threshold** — raw Canadian FWI with a tuned cutoff | 50 years of fire science, zero ML |
| 4 | **Logistic regression** on all features | Linear signal |
| 5 | **LightGBM / XGBoost** on all features, per-cell tabular | The real competitor. On tabular geospatial problems GBDTs frequently match or beat CNNs. |

Report PR-AUC, ROC-AUC, Brier score, and **skill score relative to climatology** for every one. If your ConvLSTM does not beat #5, say so honestly in the report — a rigorous negative result is a better project than an unsupported positive one, and examiners reward it.

### Phase 4 — Deep models (3–4 weeks)

Build in this order, keeping every earlier model as a comparison point:

1. **U-Net / U-Net++ on daily stacked features** — spatial only, no temporal recurrence. Strong, fast, easy to debug. Use an ImageNet-pretrained ResNet-34 or EfficientNet encoder via `segmentation-models-pytorch` and adapt the first conv for 50 channels.
2. **ConvLSTM / ConvGRU U-Net hybrid** — encoder-decoder with ConvLSTM at the bottleneck, T = 7–14 daily steps. This is the correct version of what you tried.
3. **3D U-Net** — spatiotemporal convolution alternative.
4. **Swin-UNETR / SegFormer** — transformer backbones; strong and very current.
5. **Geospatial foundation model fine-tuning** — this is the highest-signal thing you can do for a 2026 portfolio. **NASA/IBM Prithvi-EO-2.0** (HuggingFace `ibm-nasa-geospatial/Prithvi-EO-2.0-300M`) ships with an official **burn-scar segmentation** fine-tuning recipe. Also consider **Clay v1**, **SatMAE**, **TerraMind**. Fine-tuning a geospatial FM on Nepal fire data and comparing it against your from-scratch models is a complete, defensible, genuinely modern research contribution.

Losses: Focal + Tversky (or Focal-Dice) for extreme imbalance; tune the Tversky β to trade recall against precision explicitly. Use `pos_weight` derived from the *actual pixel* imbalance (~1:665), not the patch imbalance.

Multi-task heads worth adding (cheap, and they improve the shared representation):
- ignition probability (primary)
- expected FRP / fire intensity (regression)
- expected burned area from MCD64A1 (regression)
- district-level aggregate risk (auxiliary, from a pooled head)

### Phase 5 — Evaluation done properly (1 week)

- **Blocked spatio-temporal CV.** Leave-one-year-out over all 14 years, reporting mean ± std. This kills the "one lucky test year" problem outright.
- **Spatial block CV** as a second axis: hold out entire physiographic regions (Terai / Siwalik / Middle Mountains / High Mountains / High Himalaya) to test geographic generalization.
- **Stride = patch size** for evaluation patches, eliminating overlap-induced autocorrelation.
- **Threshold selection on a dedicated inner validation fold**, never on the reported fold.
- **Metrics:** PR-AUC (primary), **skill score vs climatology** (the honest headline), ROC-AUC, Brier score, **reliability diagram + Expected Calibration Error**, and an operational metric: *what fraction of actual fires fall in the top 5% / 10% of predicted-risk area?* That last one is what a fire manager cares about and almost nobody reports it.
- **Calibration:** isotonic regression or temperature scaling on the validation fold. An uncalibrated probability is not a risk product.
- **Uncertainty:** a 5-member deep ensemble, or MC-dropout. Report predictive intervals.
- **Conformal prediction** for distribution-free coverage guarantees on the risk estimate. Rare in this domain and very impressive.
- **Explainability:** SHAP on the GBDT baseline (feature-importance narrative), Integrated Gradients / attention maps on the deep model, and per-region ablations (drop wind, drop human features, drop FWI) to quantify each family's contribution.

### Phase 6 — Backend (2 weeks)

This does not exist yet and is the biggest fullstack gap.

```
FastAPI (async, Pydantic v2)
  ├─ GET  /api/risk?date=&bbox=&horizon=      → GeoJSON / vector tiles
  ├─ GET  /api/risk/tiles/{z}/{x}/{y}.png     → raster tiles (TiTiler over COG)
  ├─ GET  /api/districts/{id}/risk            → district timeseries
  ├─ GET  /api/fires/active                   → live FIRMS passthrough
  ├─ POST /api/predict                        → on-demand inference for an ROI
  ├─ WS   /ws/alerts                          → live push
  └─ GET  /api/explain/{cell}                 → SHAP contributions for one cell

PostgreSQL + PostGIS + TimescaleDB   → fire history, predictions, districts
Redis                                → tile + response cache
Prefect (or Airflow)                 → daily 02:00 NPT ingestion → inference → publish
MinIO / S3                           → COG + Zarr store
TorchServe or in-process ONNX Runtime → inference
```

Details that will differentiate you:
- Export the model to **ONNX**, quantize to int8, benchmark latency. Nepal-wide 1 km inference is a 465 × 912 forward pass — it should run in well under a second on CPU. Prove it with numbers.
- Serve predictions as **Cloud-Optimized GeoTIFF** and let TiTiler do dynamic tiling. Do not send arrays to the browser.
- Daily pipeline must be **idempotent and backfillable** — `prefect deployment run --param date=2025-04-12`.
- **Alerting**: SMS via a Nepal-local gateway (Sparrow SMS) or Viber, with district-officer subscriptions and configurable thresholds. This is the feature that turns a demo into a system.

### Phase 7 — Frontend (1.5 weeks)

Keep React + Vite + Tailwind; upgrade the rest.

- **MapLibre GL JS** instead of Leaflet — GPU-accelerated, handles a 465 × 912 raster overlay plus 400k historical fire points smoothly. Leaflet will not.
- **Deck.gl** layers for the fire-point heatmap and the animated time slider.
- **TanStack Query** for server state, **Zustand** for UI state.
- Core views: national risk map with a date/horizon scrubber, district drill-down with a risk timeseries, historical fire explorer, model-explanation panel (per-cell SHAP), and a "how accurate has this been?" verification page showing yesterday's forecast against today's detections. **That last page is rare and demonstrates real integrity.**
- **PWA with offline caching** and a **Nepali (नेपाली) locale**. Rural districts have intermittent connectivity; this is not a gimmick, it is a requirement.
- Accessibility: colourblind-safe risk ramp (do not use red-green), keyboard navigation, screen-reader labels on the map controls.

### Phase 8 — MLOps and the AI-engineering layer (2 weeks)

This is where you differentiate as an *AI engineer* rather than someone who trained a model.

- **Experiment tracking**: Weights & Biases or MLflow. Every run logged with config hash, git SHA, dataset version.
- **Model registry** with staging → production promotion gated on the eval suite.
- **CI/CD**: GitHub Actions running tests, lint, a smoke-train on a tiny fixture dataset, and Docker build/push.
- **Drift monitoring**: Evidently AI on input feature distributions; alert when 2026 conditions drift from the 2012–2025 training distribution. Climate change makes this concrete, not theoretical.
- **Observability**: Prometheus metrics (the name is right there) + Grafana dashboards for latency, prediction volume, daily positive rate.
- **Continual learning**: an automated monthly retrain that appends the last month's verified fires, re-runs the eval suite, and only promotes if it beats the incumbent.
- **An LLM layer that is actually useful**, not bolted on:
  - **Automated daily district briefings** — a small model (Llama 3.x 8B / Qwen, or an API) converts the numeric forecast plus SHAP attributions into a two-paragraph Nepali/English brief: which districts, why, what conditions drove it. This is genuine narrative generation over structured model output, and it is exactly the kind of thing that is valued right now.
  - **RAG over Nepal's fire policy corpus** — MoFE guidelines, NDRRMA response protocols, district preparedness plans — so an officer can ask "what is the protocol for a Category-4 forest fire in a community forest in Bardiya?"
  - **An agentic query interface** with tool-calling over your own API: "show me districts where risk exceeded 0.7 for three consecutive days last April" → the agent composes API calls and returns a map. Constrain it to your typed API surface; do not let it write SQL.
  - Guardrails: never let the LLM invent a risk number. It may only narrate values the model produced. State this explicitly in the report.

### Phase 9 — Optional differentiators, in order of impact-per-effort

1. **Fire spread simulation.** Seed a cellular-automaton or level-set spread model (Rothermel rate-of-spread, wind- and slope-driven) from your predicted ignition points. Turns "where might a fire start" into "and here is where it would go in 6 hours." This is the single most visually compelling feature you can build, and it composes cleanly with the ignition model.
2. **Counterfactual / scenario mode.** "What if RH drops 15% and wind doubles?" — slide the inputs, re-run inference live. Trivial to build on top of an ONNX model, extremely effective in a demo.
3. **Smoke and air-quality coupling.** Nepal's fire season is also its air-pollution crisis. Couple predicted fire to a simple HYSPLIT-style dispersion estimate or just correlate against Sentinel-5P aerosol index / ground PM2.5 to forecast Kathmandu valley AQI. Directly addresses the motivation in your Chapter 1 and nobody else will have done it.
4. **Active learning against BIPAD.** Use reported incidents as high-value labels and mine hard negatives — high-risk-predicted cells that did not burn.
5. **Prithvi vs from-scratch ablation as a formal study.** Well-run foundation-model transfer learning on a data-scarce region is a legitimate paper.

---

## 5. Feasibility, honestly

| Dimension | Assessment |
|---|---|
| Data availability | **Excellent.** 26 years of MODIS, 14 of VIIRS, full ERA5-Land archive, all free. Nothing blocks you. |
| Compute | **Fine.** Nepal-wide 1 km daily for 14 years ≈ 5,100 × 465 × 912 × 50 channels ≈ 400 GB float32, or ~100 GB as float16 Zarr. Training a U-Net on this fits comfortably on a Kaggle P100/T4 in a few hours per run. |
| Scientific difficulty | **High but tractable.** Next-day ignition prediction is genuinely hard. Realistic targets: **ROC-AUC 0.88–0.95** and **PR-AUC 0.10–0.35** at pixel level (against a 0.15% base rate that means a **60–200× lift**), with **60–80% of actual fires captured in the top 10% of predicted-risk area**. Anyone claiming pixel-level PR-AUC above 0.5 on next-day ignition is leaking. Set these expectations in the report up front. |
| Engineering scope | **Large but phaseable.** Phases 0–5 alone are a complete, defensible ML project. Phases 6–7 make it a system. Phase 8–9 make it a standout. |
| Timeline | **12–16 weeks** for Phases 0–7 at part-time student pace; ~20 weeks including 8–9. |

### If you have limited time, cut in this order

Keep, non-negotiable: Phase 0 (foundation), Phase 1 (task redefinition), Phase 2a–2b (daily data + missing predictors), Phase 3 (baselines), one deep model, Phase 5 (evaluation), a minimal FastAPI + existing frontend.

Cut first: fire spread simulation, air-quality coupling, conformal prediction, the LLM layer, foundation-model fine-tuning, transformer backbones, multi-task heads.

**A rigorous LightGBM baseline plus one honest U-Net, evaluated with leave-one-year-out CV against climatology, is a far better project than five deep models evaluated badly.** Rigor is the differentiator, not architecture count.

---

## 6. Immediate next actions (this week)

1. **Fix `report.md`'s flood text and the citation problem.** These are hours of work and they are currently the most damaging things in the repository. Verify every reference against a DOI.
2. **Reconcile the metrics in the report to one number per run** (val PR-AUC 0.718 / test PR-AUC 0.505), and **add the base rates** (0.405 / 0.279) next to them so the ~1.8× lift is visible. Reframe §4.2.3 — 0.505 is not "close to random," random is 0.279.
3. **Download VIIRS 375 m for Nepal 2012–2025** from the FIRMS API. One afternoon, and it is the highest-leverage change available.
4. **Re-export ERA5-Land as daily, and add u10/v10 wind and soil moisture.** Two hours of GEE script edits.
5. **Write the climatology baseline** — `P(fire | cell, day-of-year)` from MODIS 2000–2017. Fifty lines of code, and it gives you the yardstick every subsequent number should be measured against.
6. **Fix `train_convlstm.py:163`, `organise_ndvi_by_year.py:7`, and the fire nodata convention.**
7. **Move rasters out of git** into DVC or LFS.

---

## 7. Reference — corrected dataset table for the report

Replace §3.6 "Datasets" with the following, which reflects what the code actually uses today:

| Variable | Product | Native res. | Native temporal | Export |
|---|---|---|---|---|
| NDVI | `MODIS/061/MOD13Q1` | 250 m | 16-day composite | 1 km, EPSG:4326 |
| Temperature | `ECMWF/ERA5_LAND/DAILY_AGGR` `temperature_2m` | ~9 km | daily → 16-day mean | 1 km, EPSG:4326 |
| Precipitation | `ECMWF/ERA5_LAND/DAILY_AGGR` `total_precipitation_sum` | ~9 km | daily → 16-day sum | 1 km, EPSG:4326 |
| Relative humidity | ERA5-Land `temperature_2m` + `dewpoint_temperature_2m`, Magnus formula | ~9 km | daily → 16-day mean | 1 km, EPSG:4326 |
| VPD | Derived from ERA5-Land T and Td | ~9 km | daily → 16-day mean | 1 km, EPSG:4326 |
| Elevation | `USGS/SRTMGL1_003` | 30 m | static | 1 km, EPSG:4326 |
| Slope | `ee.Terrain.slope(SRTM)` | 30 m | static | 1 km, EPSG:4326 |
| Nepal mask | `FAO/GAUL/2015/level0` | vector | static | 1 km, EPSG:4326 |
| Fire labels | NASA FIRMS **MODIS C6.1 (MCD14ML)**, confidence ≥ 50 | 1 km | 2–4 passes/day → 16-day binary | 1 km, EPSG:4326 |

Note explicitly in the report that **ERA5-Land is upsampled from ~9 km to 1 km** and that **VIIRS was not used** in this iteration. Stating a limitation plainly is worth more marks than hiding it.
