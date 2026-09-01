# Prometheus: Daily 1 km Wildfire Risk Forecasting for Nepal

**Baldeep Karki** (03230922) and **Sammit Poudyal** (03232922)  
Department of Computer Science and Engineering  
Kathmandu University, Dhulikhel, Nepal  
COMP 308 — Year III / Semester II  
Supervisor: Dr. Rabindra Bista

---

## Abstract

Published machine-learning studies of fire in Nepal almost uniformly answer a *spatial* question: where, over a decade, can fires occur? That answer is a static susceptibility map. It does not change between January and April, or between a drought year and a wet one, and it cannot tell a district officer that *this week* is dangerous. This paper presents **Prometheus**, a national, daily, 1 km wildfire *forecast* for Nepal’s pre-monsoon season (1 January–31 May). The system predicts the probability that each forest cell will host a NASA FIRMS detection tomorrow (horizon *h* = 1) and over the next seven days (*h* = 7).

We assemble an aligned datacube on a canonical 465 × 912 grid (EPSG:4326): 821,478 seasonal FIRMS detections (2016–2026), ERA5-Land weather lapse-rate downscaled from 9 km to 1 km, MODIS vegetation and land-surface temperature interpolated to daily, and static terrain, land-cover, and human-proximity layers. A LightGBM ranker trained with leave-one-year-out (LOYO) cross-validation attains mean PR-AUC **0.1548 ± 0.0481** against a forest-mask climatology of **0.0425 ± 0.0148** (**+281% skill**) and beats both climatology and a 7-day spatial persistence baseline in **all ten** holdout seasons. Family ablations show that fire history carries the model (removing it costs **−56%** PR-AUC) while distance to roads and settlements—the dominant predictors in Nepal’s susceptibility literature—add **nothing measurable** to *daily* skill. A U-Net trained under the same protocol loses **10/10** folds. Isotonic calibration on a held-out year reduces expected calibration error by roughly **950×** on the one-day head. The frozen bundle is served as Cloud-Optimized GeoTIFFs, a FastAPI, and a React map with district drill-down, a what-if weather sandbox, a statistical cell panel (calibrated chance, comparison chart, grouped SHAP shares), and a public verification ledger (451 days, 2024–2026; mean daily PR-AUC 0.0808, top-10% capture 0.557).

**Keywords:** wildfire forecasting, Nepal, LightGBM, leave-one-year-out, PR-AUC, calibration, FIRMS, ERA5-Land, explainability

---

## I. Introduction

### A. Motivation

Forest occupies roughly 45% of Nepal. Pre-monsoon fire is not an occasional shock: about **91% of annual burned area** falls in March–May, with April the single most active month [1]. Matin *et al.* report that ~89% of recorded incidents occur in those three months [2]. The climatic mechanism is well established for the southern lowlands: low precipitation, low humidity, low soil moisture, and high temperature, amplified in El Niño years by weakened westerly moisture transport [1]. Ignition is overwhelmingly anthropogenic. The operational consequence is a short, intense window in which suppression is hard, air quality in the valleys collapses, and a static “high-risk district” list is already known to every ranger.

What is *not* known, and what existing Nepali ML products do not provide, is a **day-specific probability** at 1 km. ICIMOD’s SERVIR Hindu Kush Himalaya Forest Fire Detection and Monitoring System, NASA FIRMS, and related viewers report fires *after* a satellite has seen them [2]. That is detection, not forecast.

### B. The gap

Almost every published Nepal fire-ML study produces a **static susceptibility map**. Representative examples include weighted-index and GIS ranking at national scale [2], MaxEnt and deep neural nets trained on pooled MODIS history [3], and a large family of district- and landscape-scale Random Forest maps (Rasuwa, Chure–Tarai–Madhesh, Terai Arc Landscape, Palpa). They aggregate detections over 10–20 years and answer *where fires can occur*. Roads, settlements, and slope dominate those models because roads do not move: they are excellent predictors of the *long-run spatial cluster* and nearly silent about *which Tuesday burns*.

Prometheus answers the complementary question:

> Given today’s weather, vegetation, and fire history, what is the probability that each 1 km forest cell in Nepal sees a satellite fire detection tomorrow, and over the next week?

That is the question an operational system actually needs, and it is under-served for Nepal.

### C. Contributions

1. **A daily national forecast, not a susceptibility poster.** Two calibrated heads (*h* = 1 and *h* = 7) on a 1 km grid, restricted to burnable land below 4500 m.
2. **An honest evaluation protocol.** Ten-fold leave-one-year-out on every forest cell of every day of the held-out season; climatology and persistence recomputed on the *same* pixel population; base rate printed beside every metric; a shipping rule that a model which does not beat climatology does not ship.
3. **A methods detail that is not cosmetic.** ERA5-Land is exported at native ~9 km and downscaled with a dry-adiabatic lapse correction rather than bilinear invention of 1 km weather.
4. **An ablation that contradicts the susceptibility narrative.** Fire history is the model (−56% PR-AUC when removed). Human proximity and terrain are worth less than fold-to-fold noise for *daily* prediction.
5. **A fair deep-learning comparison that LightGBM wins.** A ResNet-18 U-Net, same years, same mask, same metrics, loses 10/10 folds. We report that rather than hide it.
6. **A working product.** Frozen bundle `v1`, 303 daily forecast packages (2024–2025 backfill), FastAPI, and a four-view React application with verification against next-day FIRMS.

### D. Scope

We model **January–May, 2016–2026** (11 seasons, 1664 days). Training on the monsoon would add ~120 trivially negative days per year and teach the model nothing it does not already learn from January dryness. November–December fires (~8% of detections) are out of scope and stated as a limitation. Labels are satellite detections, not ground-truthed perimeters.

---

## II. Related Work

### A. Fire climate and monitoring in Nepal

Hamal *et al.* [1] quantify the spring concentration of burned area and the ENSO-modulated moisture pathway. Matin *et al.* [2] map historical incidence and a linear risk index used for district ranking; ICIMOD’s operational system built on that lineage is a *detection* service. Mishra *et al.* [3] compare MaxEnt and a deep net for national *vulnerability* from pooled MODIS history—again a static surface. These papers establish seasonality, drivers, and the human-ignition regime. They do not produce a next-day probability field.

### B. Static susceptibility versus dynamic danger

Jain *et al.* [4] review ML in wildfire science and distinguish occurrence, spread, and danger rating. The Nepal literature sits almost entirely in occurrence-as-susceptibility. Internationally, daily danger forecasting has moved to datacubes and learned models. Prapas *et al.* [5] and Kondylatos *et al.* [6] forecast next-day danger over Greece from a 1 km cube (FireCube), comparing Random Forest, CNN, LSTM, and ConvLSTM against the Fire Weather Index. ConvLSTM is a strong spatial–temporal baseline *there*. We treat it as related work, not as our architecture: Nepal at 1 km with a rich tabular history is a different inductive bias, and Section VI shows a convolutional U-Net losing to gradient-boosted trees under a matched protocol.

### C. Operational indices and global viewers

The Canadian Fire Weather Index, NFDRS, and FFDI are hand-designed weather formulae. GWIS and EFFIS are excellent operational viewers whose danger layers are largely FWI-style or burned-area monitoring. They are not a Nepal-specific, calibrated, 1 km next-day model with a public accuracy ledger. Prometheus is closer to a **learned local fire-weather index** with satellite labels.

### D. Gradient boosting and calibration

LightGBM [7] is the workhorse of tabular geospatial problems: fast on CPU, native missing-value handling, and SHAP [8] at negligible extra cost. Raw boosted margins from imbalanced training are not probabilities; isotonic regression on a held-out set is the standard repair [9]. We treat calibration as load-bearing, not cosmetic (Section V-E).

---

## III. Study Area, Grid, and Data

### A. Canonical grid

Every raster in the project is asserted against one grid: **465 × 912** cells, EPSG:4326, pixel size 0.008983152841195215°, origin (80.01294235652578°E, 30.515770201540146°N). There are **168,064** valid Nepal cells. Zero fire pixels fall outside the Nepal mask (tested).

### B. Season and years

Labels and predictors cover **1 January–31 May, 2016–2026**. 2016 trains but is never a holdout fold: it is the first year with fire labels, so fire-history features would be blank by construction. That yields **ten** LOYO folds (2017–2026). Interannual variance is the reason for ten folds rather than one test year: forest-mask base rate swings from **0.26% (2020)** to **1.86% (2021)**.

### C. Labels: NASA FIRMS

Detections come from the FIRMS Area API (country endpoints were unavailable). Sensors: VIIRS S-NPP, VIIRS NOAA-20, and MODIS Collection 6.1. Cleaning: MODIS confidence ≥ 50; VIIRS confidence in {nominal, high}; drop `type ≠ 0` (flares, volcanoes); dedupe on rounded lat/lon, date, and satellite. Raw 1,040,071 rows compress to **863,807** cleaned detections; **821,478** fall in Jan–May 2016–2026 (VIIRS SNPP 433,320; NOAA-20 336,534; MODIS 51,624). Rasterisation produces a cube of shape `(1664, 465, 912)` with **1,876,970** fire pixel-days and fire on 1604 of 1664 days. Labels are **dilated 1 pixel (3 × 3)** to absorb geolocation error. Targets: `label_h1` = any detection on day *t*+1; `label_h7` = any detection in (*t*+1 … *t*+7). The last *H* days of each season are dropped so an incomplete lookahead is never treated as a zero.

### D. Weather: ERA5-Land, honestly downscaled

Nine daily fields are exported from `ECMWF/ERA5_LAND/DAILY_AGGR` at **native scale 11,132 m**, not 1 km. Exporting ERA5 at 1 km would invent detail that does not exist and inflate the download ~121×. Temperature is lapse-rate corrected:

\[
T_{1\mathrm{km}} = T_{\mathrm{ERA5}} + 0.0065\,(z_{\mathrm{ERA5}} - z_{1\mathrm{km}})
\]

where \(z_{\mathrm{ERA5}}\) is 1 km SRTM block-averaged onto the ERA5 cell and interpolated back, so the correction carries **only** the terrain ERA5 cannot see. On 15 April 2021 versus plain bilinear this is **−2.3 °C above 5000 m** and +0.3 °C in the Terai. Dewpoint uses **2 K km⁻¹**, not 6.5: applying the dry-air rate to both temperature and dewpoint would leave relative humidity unchanged and discard the signal being downscaled. Surface pressure is hypsometrically adjusted and stored in **hPa** (Pa exceeds the float16 maximum of 65,504). Derived fields: relative humidity, vapour-pressure deficit, wind speed from daily-mean *u*/*v*.

### E. Vegetation, thermal, and static layers

MODIS MOD13Q1 NDVI/EVI (16-day) and MOD11A2 LST day/night (8-day) are interpolated to daily. Cloud gaps in LST (up to 22% per composite) are closed along time **before** interpolation. Static layers: SRTM elevation, slope, aspect (sin/cos), a slope-based TWI approximation; ESA WorldCover v200 fractions; OSM distance to road and settlement; an elevation-band proxy for physiographic region (Terai / Chure / Middle / High Mountains)—declared as a limitation.

### F. Forest mask

A cell is modelled if burnable fraction (tree + shrub + grass) ≥ 0.25 **and** elevation ≤ 4500 m: **126,622** of 168,064 Nepal cells, retaining **96.1%** of fire pixel-days. Tighter masks (burnable ≥ 0.5, or 50–3500 m) discard real positives; the 4500 m treeline ceiling removes 3,839 alpine cells at **zero** cost in positives.

### G. Feature cube and training table

`features_daily.zarr` is **1664 × 465 × 912**, float16, 5.69 GB, max NaN inside the forest mask **0.000%**. The training table is **2,065,833 rows × 53 columns** (44 features + 2 labels + 7 metadata). Sampling: all positive cell-days up to a 100k budget spread over year×month strata, plus negatives at **1:20** in the same stratum, forest mask only. Capping positives thins *training*; evaluation scores **every** forest cell. Leakage controls (each tested): labels look strictly forward; NDVI-anomaly climatology is leave-one-year-out; normalisation statistics are **per fold, training years only**; `fire_clim` is built from MODIS **2003–2015**, entirely outside the modelling window.

**44 features** in seven families: weather (12), rolling dryness (7), vegetation and thermal (6), fire history (5), terrain (5), land cover (4), human (3), plus day-of-year sin/cos. Univariate Cohen’s *d* on the table already states the thesis: `fire_clim` +0.84, `vpd` +0.82, `rh` −0.76, recent-fire counts +0.68 to +0.73; `dist_road` −0.06 and `dist_settlement` +0.02 are flat. Collinear twins to disclose: `t2m_max`–`t2m` *r* = 0.99, `surface_pressure`–`elevation` *r* = 0.98, `fires_3yr`–`fires_5yr` *r* = 0.96. Harmless for LightGBM accuracy; SHAP credit must be read at the pair level.

---

## IV. Methods

### A. Task

For each forest cell and day *t* in season, predict \(P(y_{t+H}=1 \mid x_t)\) with *H* ∈ {1, 7}. This is extreme class imbalance (mean forest base rate **0.83%** on *h* = 1). Accuracy and even ROC-AUC flatter the problem; we lead with **PR-AUC**, **top-10% capture** (share of true fires falling in the highest-risk tenth of the map—the quantity a fire officer can act on), **Brier score**, and **skill versus climatology**.

### B. Baselines

- **Climatology.** Per cell and day-of-year, MODIS 2003–2015 frequency, smoothed ±7 days temporally and with a small spatial Gaussian. On *all* Nepal-mask pixels, mean PR-AUC **0.0416**, ROC-AUC 0.81, top-10% capture 0.56. On the *forest* population used by the model, climatology is recomputed to **0.0425 ± 0.0148** (LOYO mean). Comparing a forest model to an all-Nepal climatology would be meaningless; both numbers are kept, on their own populations.
- **Persistence.** Fire in the cell or its 8-neighbourhood in the last 7 days. Stronger PR-AUC on all-Nepal (0.0493) and 68% top-10% capture, but poor Brier because it emits hard 0/1 scores.

**Shipping rule.** A model that does not beat climatology PR-AUC on the same pixel population does not ship.

### C. LightGBM

One booster per horizon. Training uses `scale_pos_weight` and early stopping on the most recent *remaining* season (never a random row split: neighbouring cells on the same day are nearly the same sample). A 24-config random search over `num_leaves`, `min_data_in_leaf`, `learning_rate`, and `feature_fraction` spanned only 0.3695–0.3840 inner PR-AUC; full-grid PR-AUC was 0.2197 with defaults versus 0.2195 with the winner on the 2021 fold. **Tuning is not the lever; features and the evaluation protocol are.** Frozen production models use learning rate 0.05 rather than the search winner 0.02: inference cost is ~1.6 ms per tree over the mask, and 0.02 needed 553 / 797 trees (*h*1 / *h*7), putting *h*7 at 1.33 s. 0.05 halves the trees, costs ~2% relative PR-AUC (inside the ±0.048 fold spread), and meets a sub-second national map after season warm-up.

### D. U-Net comparison

Optional convolutional baseline, run locally on Apple Metal: `smp.Unet(resnet18, ImageNet, in_channels=44)`, Focal + Tversky (β = 0.7), AdamW 3×10⁻⁴, 20 epochs × 250 batches, batch size 16, ~11.4 min/fold. Same LOYO years, same forest-masked pixel population, same metrics. The U-Net is **research-only**; it is not served.

### E. Calibration and risk classes

The 1:20 negative downsample makes the raw booster a good *ranker* and a useless *probability*: mean raw score ~0.18 against a true rate of 0.53% on the 2026 report year—overconfident by a factor of thirty. **Isotonic regression is fitted on full-grid predictions**, not on the training table (a fit on the 1:20 sample would learn the sampled prevalence). Year split, chosen so nothing reported is anything the model touched: **fit 2016–2023 · early stopping 2024 · isotonic 2025 · report 2026.** ECE is reported with **equal-count bins**; at a sub-1% base rate, equal-width bins dump almost every pixel into the first bin. The calibrator is stored as ~500 interpolation breakpoints, not a pickled sklearn object.

Risk classes are **relative quantiles** [0.5, 0.75, 0.9, 0.95] of the predicted distribution on the calibration season—Extreme means the top 5% of place-days, matching operational fire-danger convention, not a fixed probability.

### F. Inference and product

`RiskPredictor("latest")` returns a (465, 912) calibrated field. First call of a season pays ~12 s and ~3.4 GB (rolling windows and fire history depend on the season to date); subsequent dates are **0.34 s median (*h*1)** and **0.60 s (*h*7)**. Daily artefacts: COG-like GeoTIFFs (`nodata = −1` so probability 0 stays valid), 77-district GeoJSON (OSM admin_level 6), and an append-only verification CSV. FastAPI serves XYZ risk tiles, district summaries and timeseries, active fires, verification, per-cell explanations (calibrated probability, snapshot, grouped SHAP), and a what-if scorer. The React application (Leaflet, light/dark) exposes five views: national map with date scrubber and a shared *h*1/*h*7 toggle (left card and cell panel); a what-if sandbox; district page; FIRMS explorer; verification page that always prints the base rate.

The public map backfill is **January–May 2024, 2025, and 2026**. The React map
opens on **12 April 2026**. There is no live “today” GEE hop; the site plays
history from the cube.

---

## V. Experimental Protocol

All model metrics below are computed on **every forest cell of every day** of the held-out season, never on the sampled training table. 2016 is used only to warm fire history. Ablations drop an entire family, including rolling aggregates when weather is dropped, so the weather ablation is honest. Day-of-year encoding belongs to no family and is never dropped. SHAP is summarised globally and by physiographic belt; collinear twins are reported as pairs.

---

## VI. Results

### A. Leave-one-year-out (horizon *h* = 1)

**Table I.** LightGBM versus climatology and persistence on the forest mask. Top-10% is capture of true fires. Base rate is the held-out season’s positive fraction.

| Holdout | PR-AUC | Climatology | Persistence | Top-10% | Base rate |
|--------:|-------:|------------:|------------:|--------:|----------:|
| 2017 | 0.1236 | 0.0271 | 0.0203 | 0.753 | 0.32% |
| 2018 | 0.1441 | 0.0351 | 0.0272 | 0.723 | 0.56% |
| 2019 | 0.2358 | 0.0441 | 0.0567 | 0.804 | 0.81% |
| 2020 | 0.0730 | 0.0153 | 0.0220 | 0.686 | 0.26% |
| 2021 | 0.2195 | 0.0566 | 0.0691 | 0.648 | 1.86% |
| 2022 | 0.1343 | 0.0460 | 0.0345 | 0.715 | 0.49% |
| 2023 | 0.1330 | 0.0458 | 0.0395 | 0.617 | 1.19% |
| 2024 | 0.1900 | 0.0687 | 0.0591 | 0.652 | 1.57% |
| 2025 | 0.1500 | 0.0458 | 0.0323 | 0.699 | 0.72% |
| 2026 | 0.1443 | 0.0408 | 0.0309 | 0.728 | 0.53% |
| **mean ± std** | **0.1548 ± 0.0481** | 0.0425 ± 0.0148 | 0.0392 ± 0.0168 | 0.703 ± 0.055 | 0.83% |

LightGBM beats both baselines in **every** fold (**+281%** mean skill vs climatology). Fold spread tracks the base rate: 2020 is the quietest year and the worst PR-AUC; 2019 and 2021 are busy and best. That is why climatology is a *column*, not a single global number. Mean top-10% capture is **70%**: seven in ten satellite fires sat in the reddest tenth of the map.

A single 2021 fold, scored on 18,993,300 pixel-days (353,637 positive), is the sanity check that preceded the full CV: LightGBM PR-AUC 0.2195 versus persistence 0.0691 and climatology 0.0566 (**+287%** skill), trained in 48 s.

### B. Ablations

**Table II.** Mean LOYO PR-AUC when a feature family is removed.

| Variant | Features | PR-AUC | Δ | Δ% |
|---|---:|---:|---:|---:|
| drop_terrain | 35 | 0.1552 ± 0.0467 | +0.0004 | +0.6% |
| **full** | **44** | **0.1548 ± 0.0481** | — | — |
| drop_human | 41 | 0.1543 ± 0.0486 | −0.0005 | −0.4% |
| drop_vegetation | 38 | 0.1521 ± 0.0471 | −0.0027 | −1.6% |
| drop_weather | 25 | 0.1404 ± 0.0462 | −0.0143 | −9.3% |
| drop_fire_history | 39 | 0.0695 ± 0.0294 | −0.0853 | **−56.2%** |

**Fire history is the model.** Remove it and PR-AUC more than halves, to 0.0695—barely above climatology. Weather is the only other family that matters. Terrain and human proximity move the mean by less than 0.001, inside the ±0.048 fold spread; dropping terrain nominally *improves* it. This is the Day-8 correlation finding under a much stronger test: static susceptibility layers describe where fires cluster over decades, not which day burns. Fire history does not leak: `fire_clim` is 2003–2015 MODIS; `days_since_fire` and `fires_Nyr` use detections through day *t* to predict *t*+1.

### C. Physiographic belts

**Table III.** Per-region LOYO (positives are pixel-days).

| Region | Positives | Base rate | PR-AUC | Climatology | Skill |
|---|---:|---:|---:|---:|---:|
| Terai | 323,038 | 1.29% | 0.1468 ± 0.0411 | 0.0655 | +136% |
| Chure | 699,612 | 1.58% | 0.2398 ± 0.0530 | 0.0604 | +341% |
| Middle Mountains | 521,499 | 0.59% | 0.0934 ± 0.0546 | 0.0172 | +469% |
| High Mountains | 37,798 | 0.11% | 0.0483 ± 0.0344 | 0.0022 | +2151% |

Raw PR-AUC falls with elevation, but so does the base rate, so **skill runs the other way**: the model adds most where fire is rarest. Chure is the best-served belt in absolute PR-AUC and the busiest.

The literature argues drivers differ substantially by belt. **The model only partly agrees.** SHAP family shares keep the same ranking everywhere—weather, fire history, vegetation, calendar, terrain, human last (~2.3% in every belt). What shifts is a monotone gradient: fire history 25.8% → 20.4% from Terai to High Mountains; terrain 8.5% → 11.5%; vegetation/thermal 17.4% → 21.1%. `lst_day` reaches ~10% only in the High Mountains. A real gradient, not a different-drivers-per-region story.

### D. SHAP (global, 2021)

Top shares: `doy_cos` 10.7%, `fire_clim` 9.2%, `days_since_fire` 6.8%, `doy_sin` 6.6%, `fires_5yr` 6.0%, `rh` 5.9%, `lst_day` 5.8%, `precip` 5.2%. Dependence plots are physically readable: `rh` crosses from fire-promoting to fire-suppressing near **60%**; `days_since_fire` spikes in the first ~50 days after a burn then decays (repeat ignition in the spring burning season); `fires_5yr` rises to ~8 prior fires and saturates; day-of-year traces the March–April peak. Gain importance on the 2021 fold is dominated by `days_since_fire` (25.7%) and `fire_clim` (14.1%), with pooled pair gain reported for collinear twins.

### E. Calibration and classes (held-out 2026)

**Table IV.** Raw booster versus isotonic, full grid.

| Horizon | Base rate | mean raw → cal. | ECE raw → cal. | Brier raw → cal. | PR-AUC |
|---|---:|---:|---:|---:|---:|
| *h*1 | 0.529% | 0.181 → 0.0052 | 0.1757 → **0.00018** | 0.0729 → 0.0047 | 0.1535 → 0.1538 |
| *h*7 | 2.603% | 0.179 → 0.0236 | 0.1531 → **0.00281** | 0.0709 → 0.0215 | 0.2880 → 0.2875 |

ECE improves ~**950×** (*h*1) and ~**55×** (*h*7). Calibrated mean probability lands within 2% of the observed base rate. PR-AUC is unchanged to ±0.0005: isotonic is monotone and cannot reorder. The reliability curve, plotted log–log because a linear diagram at this base rate is a dot in the corner, tracks the diagonal from 10⁻⁵ to 10⁻¹ after calibration.

**Table V.** Risk classes on 2026, *h*1.

| Class | % of grid | Observed rate | % of fires captured |
|---|---:|---:|---:|
| Low | 52.3% | 0.039% | 3.9% |
| Moderate | 28.6% | 0.215% | 11.7% |
| High | 12.6% | 0.809% | 19.3% |
| Very High | 3.4% | 2.046% | 13.3% |
| **Extreme** | **3.0%** | **9.038%** | **51.9%** |

Observed rate rises monotonically and spans a **232×** range from Low to Extreme. Extreme is 3% of the grid and holds **over half** of all fires.

### F. U-Net versus LightGBM

**Table VI.** Same ten holdouts, forest mask.

| Holdout | U-Net | LightGBM | Climatology | Winner |
|--------:|------:|---------:|------------:|---|
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

The U-Net still beats climatology on every fold (~+150% skill) and matches LightGBM on top-10% capture (0.706 vs 0.703). It does not win PR-AUC: LightGBM is ahead by ~**35% relative**. On this tabular geospatial problem that is the common outcome; reporting it is the point of the comparison.

### G. Operational verification

An *h*1 forecast for day *D* is scored against FIRMS on day *D*+1 on the forest mask, over the 2024–2026 backfill: **451** days, 442 with a valid next-day label. Mean daily PR-AUC **0.0808**, top-10% capture **0.557**, Brier 0.0083. Daily PR-AUC is noisier and lower than season-level LOYO because each row is a single day with a tiny base rate. Both numbers are shown on purpose; the verification page is not a replacement for Table I. Example date 2025-04-12: 121 fires, base rate 0.096%, next-day PR-AUC 0.024 on a quiet day.

---

## VII. System

The production path is **only** the frozen LightGBM bundle `v1` (calibrated *h*1 and *h*7). The U-Net remains under `runs/unet/`.

**Backend.** FastAPI over disk (no PostGIS at this scale): `GET /api/risk/tiles/{z}/{x}/{y}.png`, districts GeoJSON and timeseries, active fires, verification, `GET /api/explain?lat=&lon=` (calibrated chance, snapshot, grouped SHAP), `GET /api/whatif/schema` and `POST /api/whatif`, health and forecast catalogue. Out-of-bounds tiles return a 1 × 1 transparent PNG; empty fire windows return an empty FeatureCollection.

**Frontend.** Five routes: national map (COG tiles, 77 districts, date scrubber with play, *h*1/*h*7 toggle on the layer card and in the cell panel); what-if sandbox; district mean/max and season timeseries; FIRMS points with lookback; verification with base rate always visible. Design is a sharp dark theme with a paper light invert. Cell explanations lead with probability and a comparison chart; TreeSHAP is grouped into percent shares rather than slogan copy.

**Reproducibility.** `configs/base.yaml` is the only place years, months, grid, and feature lists may live. 100 tests; alignment of every raster is asserted. Loading a bundle by version is the only supported way to predict.

---

## VIII. Discussion

### A. What the numbers mean

A next-day PR-AUC of 0.15 against a ~0.8% base rate is a large lift over climatology, not a claim that the map is a crystal ball. Anyone advertising pixel-level next-day PR-AUC above ~0.5 on this problem is almost certainly leaking. ROC-AUC in the 0.8s is cheap here—climatology already has 0.81 on all-Nepal pixels—because negatives are easy. We therefore refuse to lead with ROC.

Top-10% capture (~70% in LOYO, ~55% in noisy daily verification) is the operational translation: concentrate attention on the purple slice.

### B. Why human proximity “fails”

It does not fail at the question the literature asked. Roads predict *where* fire lives over twenty years. They do not move from Monday to Tuesday. Once fire history and daily weather are in the model, distance-to-road is redundant for *when*. That is a feature of the task, not a bug in OSM.

### C. Why LightGBM beats the U-Net here

The informative structure is largely **per-cell time and history**, not a 128 × 128 spatial texture of ignition. Gradient-boosted trees on a well-aligned table, with SHAP and sub-second CPU inference, are the right default. Deep models remain interesting for spread and burned-area segmentation—out of scope.

### D. Limitations

Wind is derived from daily-**mean** ERA5 *u*/*v*, so direction-averaging understates gusts. Physiographic regions are elevation-band proxies. `built_frac` substitutes for population density (WorldPop was not downloaded). Rolling windows truncate in early January because December was not downloaded. TWI is slope-based, not full flow routing. 2016 is never a holdout. Raw booster scores must never be read as probabilities. The first `predict` of a season costs ~12 s. Live GEE re-export for “today” is not wired. Daily verification PR-AUC must not replace Table I. OSM district names mix scripts; `district_id` 1–77 is the key. Detections are not ground truth.

### E. Ethics and use

Prometheus is a preparedness overlay, not a dispatch system and not a substitute for a local officer on a specific hillside. No score is emitted off the forest mask. The model card records intended use and the limits above.

---

## IX. Conclusion

Prometheus is a daily 1 km wildfire **forecast** for Nepal’s pre-monsoon season. Leave-one-year-out evaluation gives LightGBM PR-AUC **0.1548 ± 0.0481** versus climatology **0.0425** (**+281% skill**, 10/10 folds), ~70% of fires in the top tenth of the map, and an Extreme class that is 3% of the grid and holds 52% of fires after calibration. Fire history, not roads, carries daily skill. A U-Net loses 10/10. The frozen bundle is served as maps, an API, and a verification ledger.

Future work that is actually next, rather than ornamental: wire a live GEE hop for “today”; replace the elevation-band physiography with official polygons; add November–December as a second, separately evaluated season; and, if a legal standard requires it, publish a side-by-side with a Nepal-tuned FWI rather than claiming to replace one.

---

## Acknowledgment

We thank the Department of Computer Science and Engineering, Kathmandu University, and our supervisor Dr. Rabindra Bista. This work depends on NASA FIRMS, ECMWF ERA5-Land via Google Earth Engine, NASA MODIS, ESA WorldCover, OpenStreetMap, and the open-source scientific Python and web stack. Any errors are ours.

---

## References

[1] K. Hamal, S. K. Ghimire, A. Khadka, B. Dawadi, and S. Sharma, “Interannual variability of spring fire in southern Nepal,” *Atmospheric Science Letters*, vol. 23, no. 9, 2022, doi: 10.1002/asl.1096.

[2] M. A. Matin, V. S. Chitale, M. S. R. Murthy, K. Uddin, B. Bajracharya, and S. Pradhan, “Understanding forest fire patterns and risk in Nepal using remote sensing, geographic information system and historical fire data,” *International Journal of Wildland Fire*, vol. 26, no. 4, pp. 276–286, 2017, doi: 10.1071/WF16056.

[3] B. Mishra, S. Panthi, S. Poudel, and B. R. Ghimire, “Forest fire pattern and vulnerability mapping using deep learning in Nepal,” *Fire Ecology*, vol. 19, no. 3, 2023, doi: 10.1186/s42408-022-00162-3.

[4] P. Jain, S. C. P. Coogan, S. G. Subramanian, M. Crowley, S. Taylor, and M. D. Flannigan, “A review of machine learning applications in wildfire science and management,” *Environmental Reviews*, vol. 28, no. 4, pp. 478–505, 2020, doi: 10.1139/er-2020-0019.

[5] I. Prapas, S. Kondylatos, and I. Papoutsis, “Deep learning methods for daily wildfire danger forecasting,” arXiv:2111.02736, 2021.

[6] S. Kondylatos, I. Prapas, M. Ronco, I. Papoutsis, G. Camps-Valls, M. Piles *et al.*, “Wildfire danger prediction and understanding with deep learning,” *Geophysical Research Letters*, vol. 49, e2022GL099368, 2022, doi: 10.1029/2022GL099368.

[7] G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye, and T.-Y. Liu, “LightGBM: A highly efficient gradient boosting decision tree,” in *Proc. NeurIPS*, 2017.

[8] S. M. Lundberg and S.-I. Lee, “A unified approach to interpreting model predictions,” in *Proc. NeurIPS*, 2017.

[9] A. Niculescu-Mizil and R. Caruana, “Predicting good probabilities with supervised learning,” in *Proc. ICML*, 2005, pp. 625–632.

[10] J. Muñoz-Sabater *et al.*, “ERA5-Land: A state-of-the-art global reanalysis dataset for land applications,” *Earth System Science Data*, vol. 13, pp. 4349–4383, 2021, doi: 10.5194/essd-13-4349-2021.

[11] L. Giglio, W. Schroeder, and C. O. Justice, “The collection 6 MODIS active fire detection algorithm and fire products,” *Remote Sensing of Environment*, vol. 178, pp. 31–41, 2016.

[12] W. Schroeder, P. Oliva, L. Giglio, and I. A. Csiszar, “The New VIIRS 375 m active fire detection data product: Algorithm description and initial assessment,” *Remote Sensing of Environment*, vol. 143, pp. 85–96, 2014.

[13] O. Ronneberger, P. Fischer, and T. Brox, “U-Net: Convolutional networks for biomedical image segmentation,” in *Proc. MICCAI*, 2015, pp. 234–241.

[14] D. Zanaga *et al.*, “ESA WorldCover 10 m 2021 v200,” 2022. [Online]. Available: https://doi.org/10.5281/zenodo.7254221

---

## Appendix A — Implementation inventory

| Artefact | Location | Notes |
|---|---|---|
| Fire labels | `data/cube/fire_daily.zarr` | 1664 × 465 × 912 |
| Feature cube | `data/cube/features_daily.zarr` | 5.69 GB, 17 variables + static |
| Training table | `data/cube/train_table.parquet` | 2,065,833 × 53 |
| Bundle `v1` | `data/models/bundles/v1/` | *h*1+*h*7, calibrators, model card |
| LOYO / ablations / SHAP | `runs/cv/`, `runs/shap/` | 10 folds |
| U-Net | `runs/unet/` | research comparison |
| Forecasts | `runs/forecasts/` | 454 days (2024–2026) |
| Tests | `tests/` | 100 passing |

Config, grid, and feature lists live only in `configs/base.yaml`. End-to-end commands are documented in the repository `README.md` and `PROGRESS_REPORT.md`.
