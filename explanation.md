# Prometheus, explained for a beginner

This note is the plain-language tour of the project: what it is, why it exists,
what was actually built, and how it compares to other fire maps.

If you only want to run the app, use [README.md](README.md).

---

## The one-sentence version

**Prometheus is a daily wildfire *forecast* for Nepal** — not a poster of
“where fires have happened over the last twenty years.”

It answers: *given the weather and vegetation we have today, how likely is each
1 km patch of forest to see a satellite fire detection tomorrow, and over the
next week?*

---

## Why it was made

Nepal’s fire season is brutally seasonal. About **nine-tenths of burned area
falls in March–May**; April alone is often half of all detections. District
officers, forest users, and journalists already know *that* spring is
dangerous. What they do not get from published Nepali fire ML is *which week*
and *which district* is about to light up.

Almost every Nepal fire-ML paper produces a **static susceptibility map**:

> “Over 10–20 years, fires clustered here. Therefore this slope / this road
> corridor is ‘high risk’ forever.”

That map is the same in January as in April, the same in a wet year as in a
drought. It cannot say “this Tuesday is worse than last Tuesday.”

Prometheus was built to fill that gap as a Year-III project: a **dynamic daily
forecast**, evaluated honestly, served as a map a non-specialist can open.

---

## What “a forecast” means here (and what it does not)

| It is | It is not |
|---|---|
| A **probability** that a 1 km forest cell will have a FIRMS fire detection in the next 1 or 7 days | A guarantee that a village will burn |
| Calibrated so 3% really means “about 3% of similar days burned” | The raw machine-learning score (those were ~30× too high) |
| Restricted to **forest / shrub / grass** below 4500 m | A score for cities, glaciers, or open water |
| Trained on **January–May, 2016–2026** | A monsoon or winter product |
| Compared against **next-day satellites** | Ground-truthed burned-area perimeters |

Satellites miss small or smoky fires and can geolocate them a pixel off. We
dilate labels by one pixel for that. Detections are still the best national
daily record we have.

---

## How the system works, without jargon

```
Satellites see fire (NASA FIRMS)
Weather (ERA5) + greenness (MODIS) + terrain + roads
        ↓
A table: one row per forest cell per day, 44 numbers describing the place
        ↓
A tree model (LightGBM) learns which combinations precede a fire
        ↓
A calibrator turns the score into a real probability
        ↓
A GeoTIFF “danger map” + district averages
        ↓
A website: paint the map, click a cell for the chance, check yesterday’s skill
```

**Tomorrow (H1)** = chance of a detection on the next calendar day.  
**Next 7 days (H7)** = chance over the coming week (a higher number, because
the window is longer).

Colors on the map run **pale yellow → orange → red → purple**. Purple is the
most dangerous slice of the country that day, not “the forest will definitely
burn.”

The rest of this section is the same story in order: what is downloaded, how it
is cleaned, how everything is forced onto one map, how the model is trained,
how LightGBM actually learns, and how a map is produced at inference time.

---

## The full pipeline: data → cube → model → map

Think of Prometheus as a factory with a single rule: **every number that
reaches the model must live on the same 1 km Nepal grid, for the same
calendar day, with no peeking at the future.** `configs/base.yaml` is the
only place years, months, grid size, and feature names are allowed to live.

### 0. One shared map (the canonical grid)

Before any download, the project fixes a raster:

- **465 × 912** cells, **EPSG:4326** (plain lon/lat)
- Pixel size ≈ **0.009°** (~1 km)
- Origin at the north-west corner of the Nepal ROI
  (about 80.01°E, 30.52°N)
- A binary **Nepal mask** (`nepal_mask_1km_roiAligned.tif`) saying which of
  those cells are actually in the country: **168,064** of them

Every later GeoTIFF is tested with `grid.assert_aligned`. If a file is
shifted by one pixel, the test fails. That is why weather, vegetation, fires,
and the forecast map can be stacked without “this fire is in the next
valley.”

A second mask, the **forest mask**, decides *who the model is allowed to
talk about*:

- ESA WorldCover tree + shrub + grass fraction **≥ 25%**
- Elevation **≤ 4500 m** (above Nepal’s treeline there is no continuous fuel)

That keeps **126,622** cells and **96%** of fire pixel-days. Cities, glaciers,
open water, and high alpine rock are not scored — a “no fire” there would be
a cheap, meaningless negative.

Only **January–May** is modelled (the pre-monsoon fire season). Years
**2016–2026** have full predictors. A cheaper MODIS fire archive from
**2003–2015** is kept only to build the day-of-year climatology feature, not
as weather.

---

### 1. What is downloaded, and how

Four independent sources. Nothing is scraped from the website; each has an
official export path.

#### A. Fire detections — NASA FIRMS (the labels)

- **What:** hot-spot points from three sensors: VIIRS S-NPP, VIIRS NOAA-20
  (from 2018), and MODIS Collection 6.1.
- **How:** the FIRMS **Area API**. A MAP_KEY (NASA Earthdata) is stored
  locally, never in git. The code walks the season in 1–5 day windows
  (the API’s maximum), for the Nepal bounding box, and caches each window
  as a CSV under `data/raw/firms/chunks/`.
- **Also:** MODIS 2003–2015 is downloaded the same way, but only to feed
  `fire_clim` (historical “this calendar day usually burns here”).
- **Scale:** ~1 million raw rows; after cleaning, **~821k seasonal
  detections** for 2016–2026.

This is the *answer key*. Everything else is a *clue*.

#### B. Weather — ERA5-Land via Google Earth Engine

- **What:** daily 2 m temperature (mean / min / max), dewpoint, precipitation,
  10 m wind *u*/*v*, soil water (top layer), surface pressure.
- **How:** `gee/era5_daily.js` runs in Google’s cloud. It exports **one
  multi-band GeoTIFF per month** (Jan–May × each year) to Google Drive,
  at ERA5’s **native ~9 km** (scale 11,132 m). You then copy the files into
  `data/raw/gee/era5/`.
- **Why not 1 km in GEE?** ERA5 does not *have* 1 km weather. Exporting it
  at 1 km would invent 121× more pixels of fake detail and a huge download.
  Honest downscaling happens later, on the laptop, using the real terrain.

#### C. Greenness and surface heat — MODIS via GEE

- **NDVI / EVI:** MOD13Q1, a **16-day** composite, ~1 km.
- **Land-surface temperature day/night:** MOD11A2, an **8-day** composite.
- **How:** `gee/ndvi_16day.js` and `gee/lst_8day.js`. One GeoTIFF per
  composite date, Drive → `data/raw/gee/ndvi/` and `.../lst/`.

These are not daily. Clouds punch holes. Both problems are repaired in
preprocessing, not by pretending the satellite saw every morning.

#### D. Static geography — GEE + OpenStreetMap

From GEE (`gee/static.js`), once:

- **SRTM** elevation, slope, aspect, a slope-based topographic wetness
  index (TWI)
- **ESA WorldCover v200** class fractions (tree, shrub, grass, crop, built, …)

From Geofabrik’s Nepal OSM extract, locally (`scripts/build_local_static.py`):

- Distance to the nearest **road**
- Distance to the nearest **settlement**
- A crude physiographic-region raster (Terai / Chure / Middle / High
  Mountains) as an elevation-band proxy

Static layers do not change with the calendar. They describe *the place*,
not *the day*.

**You only download this factory once.** The public website never talks to
FIRMS or Earth Engine.

---

### 2. Cleaning and preprocessing (still separate files)

Each source is cleaned in its own native shape. Alignment to 1 km comes
next.

#### Fires → a daily 0/1 cube

`scripts/build_fire_labels.py` / `src/prometheus/data/firms.py`:

1. Drop junk coordinates; clip to the Nepal bbox.
2. Keep only presumed vegetation fires (`type == 0`); drop gas flares and
   volcanoes.
3. **Confidence:** MODIS ≥ 50; VIIRS in {nominal, high}.
4. Deduplicate on rounded lat/lon + date + satellite.
5. Snap each point onto the 1 km grid. Cells outside the Nepal mask are
   ignored (zero fire pixels fall outside — tested).
6. **Dilate 1 pixel** (a 3 × 3 neighbourhood). Satellites can geolocate a
   fire one cell off; punishing the model for that is unfair.
7. Write `data/cube/fire_daily.zarr`: shape **(1664 days × 465 × 912)**,
   uint8, 1 = detection that calendar day.

Labels for training are *forward-looking*, built later:

- `label_h1` = any detection on **day t+1**
- `label_h7` = any detection in **days t+1 … t+7**

The last 1 (or 7) days of May are marked invalid, not labelled zero. That
stops “no fire because we ran out of season” from looking like a true
negative.

#### Weather → 1 km, with physics, not Photoshop

`src/prometheus/features/weather.py`, month by month:

1. Read the ~9 km ERA5 stack.
2. Warp each field onto the 1 km grid with bilinear resampling.
3. **Lapse-rate correction** so mountains are colder than the coarse cell
   that averaged them away:

   `T_1km = T_ERA5 + 0.0065 × (elev_ERA5 − elev_1km)`

   Dewpoint uses **2 K/km**, not 6.5. Using the dry-air rate for both
   would leave relative humidity unchanged and throw away the reason we
   downscaled.
4. Surface pressure is converted to **hPa** and hypsometrically adjusted
   (Pa would overflow float16).
5. **Derived on the fine grid:** relative humidity (Magnus–Tetens), vapour
   pressure deficit (VPD, kPa), wind speed = √(u² + v²).

Result: twelve daily weather fields that respect Nepal’s relief.

#### Vegetation / LST → daily

`src/prometheus/features/vegetation.py`:

1. Warp each composite to the canonical grid; throw away physically
   impossible values (NDVI outside −0.2…1, LST outside −60…70 °C).
2. Fill cloud holes **along time** (forward then backward) so
   interpolation never jumps across a NaN.
3. **Linear interpolation in time** onto every Jan–May calendar day.
   Dates outside the composite span hold the nearest composite; they do
   not extrapolate.
4. Pixels with no valid composite all season fall back to a multi-year
   per-pixel mean, then to the nearest neighbour in space.

`lst_diff` = day minus night temperature (a dryness / cloud-free-sky
cue). `ndvi_anomaly` is computed later, leave-one-year-out, so 2021’s
greenness is not compared to a climatology that already includes 2021.

#### Roads and settlements → distance rasters

OSM lines and points are burned onto the 1 km grid. A Euclidean distance
transform then writes, for every cell, “how far to the nearest road /
settlement.” Ablation later shows these barely move daily skill — they
are in the table so that claim is measured, not assumed.

---

### 3. Alignment: one feature cube

`scripts/build_feature_cube.py` writes `data/cube/features_daily.zarr`
(~5.7 GB, float16):

```
time (1664 season days) × y (465) × x (912)
```

- **Dynamic** (changes every day): the 12 weather fields, NDVI, EVI,
  LST day/night/diff.
- **Static** (copied once): elevation, slope, aspect sin/cos, TWI, land-
  cover fractions, road/settlement distance, forest mask, Nepal mask.

Outside Nepal is NaN. Inside the forest mask, NaN fraction is ~0% —
the cube is gap-free enough to train on.

This cube is still “raw-ish.” A second pass, shared by training and
inference so they cannot silently diverge
(`src/prometheus/features/table.py` → `iter_year_features`), adds
**derived** columns that need a run of days, not a single snapshot:

| Family | Examples | What they capture |
|---|---|---|
| Rolling dryness | `precip_7d`, `precip_30d`, `t2m_max_7d`, `rh_min_7d`, `wind_max_7d` | Fuel drying over a week / month |
| Dry-spell counters | `consecutive_dry_days`, `days_since_rain` | How long since ≥1 mm / ≥0.1 mm |
| Fire history | `days_since_fire`, `fires_1yr/3yr/5yr` | This cell’s recent burning (prior seasons + this season up to *today*) |
| Climatology | `fire_clim` | 2003–2015 MODIS rate for this cell and day-of-year |
| Calendar | `doy_sin`, `doy_cos` | Smooth seasonality (April ≠ January) without a hard month id |
| Greenness | `ndvi_anomaly` | This year’s NDVI vs other years, same calendar slot |

**Leakage rules baked in here:**

- History may use **today’s** detections when forecasting *tomorrow*
  (those pixels are already on the satellite), but never t+1.
- `fires_1yr` etc. are counts of **past seasons**, not a window that
  dangles into the undownloaded monsoon.
- Rolling windows **restart in early January** because December was
  never downloaded. That is a known limitation, not a silent bug.
- NDVI anomaly climatology is **leave-one-year-out**.

After this pass there are **44 predictors** in seven families: weather
(12), rolling dryness (7), vegetation/thermal (6), fire history (5),
terrain (5), land cover (4), human (3), plus day-of-year sin/cos.

---

### 4. From cube to a training table

The cube is ~126,622 forest cells × ~151 days × 11 years ≈ **210 million**
cell-days. Fire is ~0.8% of those. Training on every negative would drown
the signal and blow RAM.

`scripts/build_train_table.py` samples `train_table.parquet` (~2.1 million
rows × 53 columns):

1. Restrict to the forest mask.
2. Keep (almost) **every positive** cell-day, capped at 100k spread evenly
   across year × month so April cannot monopolise the sample.
3. Draw **20 negatives per positive** from the *same* year-month. That
   keeps the season’s shape (quiet January, violent April) instead of
   collapsing onto peak fire days.
4. Attach `label_h1` and `label_h7`, plus metadata (year, doy, row, col,
   lat, lon).

Capping positives thins **training only**. Evaluation later scores
**every** forest cell on the held-out year. Reported PR-AUC is not an
artefact of this 1:20 sample.

Normalisation statistics (means/stdevs) are computed **per fold, on
training years only**. LightGBM itself does not need scaling — trees
split on raw thresholds — but the stats exist for the U-Net comparison
and for documentation.

---

### 5. How the model is trained

Two heads share the same features and the same sampling: one booster for
**h1** (tomorrow), one for **h7** (next week). Production training
(`scripts/build_model_bundle.py`) uses a year split chosen so nothing
reported is anything the model fitted:

| Years | Role |
|---|---|
| 2016–2023 | Fit the trees |
| 2024 | Inner validation / early stopping (not the holdout year) |
| 2025 | Fit the **isotonic calibrator** on full-grid scores |
| 2026 | Report year — never used to pick trees or the calibrator |

Leave-one-year-out CV (ten folds, 2017–2026) is the science table.
**2016 is never a holdout:** there is no prior season to build fire
history from.

**Training one fold, in order:**

1. Split the parquet: all years except the held-out year, minus the most
   recent *remaining* season as inner validation. A random row split would
   leak — neighbouring cells on the same day are almost the same sample.
2. Compute `scale_pos_weight` = negatives / positives in the *fit* set.
   The table is already 1:20, so this only corrects residual imbalance,
   not the raw 1:100 base rate twice.
3. Hand LightGBM a matrix of 44 columns and a 0/1 label.
4. Grow up to 800 trees (learning rate 0.05 in the frozen bundle) with
   `num_leaves=127`, `min_data_in_leaf=200`, `feature_fraction=0.65`.
5. **Early stopping:** if inner-validation PR-AUC has not improved for 50
   rounds, stop. The saved model keeps only `best_iteration` trees.
6. Feature names are stored *inside* the booster so inference can rebuild
   columns in the same order.

A 24-config random search barely moved inner PR-AUC (0.37–0.38).
**Tuning is not the lever; the features and the year-wise protocol are.**

After the trees exist they are still a *ranker*, not a probability. Mean
raw score on 2026 was ~0.18 against a true rate of ~0.5% — overconfident
by ~30×. **Isotonic regression** is fitted on *full-grid* predictions for
2025 (never on the 1:20 table, which would learn the sampled prevalence).
It is stored as ~500 monotone breakpoints and applied with `np.interp`.
Ranking skill is almost unchanged; the numbers become honest percents.

Risk classes (Low → Extreme) are **relative quantiles** of that calibrated
distribution (50 / 75 / 90 / 95th). Extreme means “the most dangerous 5%
of forest cell-days,” not “P > 50%.” At a sub-1% base rate, 50% will
almost never happen.

The frozen artefact is `data/models/bundles/v1/` (~8 MB): two LightGBM
text models, two calibrators, feature lists, risk thresholds. That is
what inference loads. The 5 GB cube is needed to *build* a day’s 44
features, not to store the trees.

---

## How LightGBM works (and how it learns fire here)

### A decision tree, in one picture

A tree asks a sequence of yes/no questions about a row:

> Is VPD > 1.8 kPa?  
> &nbsp;&nbsp;Yes → has this cell burned in the last 3 seasons?  
> &nbsp;&nbsp;&nbsp;&nbsp;Yes → leaf A (many historical fires)  
> &nbsp;&nbsp;No → is day-of-year in late April?  
> &nbsp;&nbsp;&nbsp;&nbsp;… → leaf B (few fires)

Each **leaf** stores a number: “rows that landed here were this much more
(or less) likely to be a fire than the average.” A single tree is a crude
map of the feature space. It underfits.

### Gradient boosting: many trees, each correcting the last

**LightGBM** (Light Gradient Boosting Machine) grows an *ensemble*:

1. Start with a constant (roughly the log-odds of fire in the training
   table).
2. Compute how wrong that is on every row (**gradient** of the binary
   log-loss). Rows the model currently misses — especially rare positives
   up-weighted by `scale_pos_weight` — get more attention.
3. Fit a new tree to those residuals.
4. Add it, shrunk by the **learning rate** (0.05). Small steps; many
   trees; less overfitting than one huge tree.
5. Repeat until early stopping.

The final raw score for a cell-day is the **sum of all leaf values** the
row fell into. LightGBM’s `binary` objective means that sum is a
log-odds; `predict()` turns it into a number in (0, 1) — but after 1:20
sampling that number is **not** the real-world probability. That is why
the isotonic calibrator exists.

### Why LightGBM, specifically

- **Leaf-wise growth** (not level-wise): it expands the leaf that most
  reduces loss, so it fits interactions (hot × dry × recently burned)
  with fewer trees than a random forest.
- **Histogram splits:** features are bucketed, so 2 million rows train
  in about a minute on a laptop CPU. No GPU.
- **Native NaNs:** a missing value can go left or right as its own
  decision. We barely need this (the cube is full), but it is safe.
- **Feature fraction 0.65:** each tree sees a random 65% of columns, so
  collinear twins (`t2m` / `t2m_max`, `fires_3yr` / `fires_5yr`) cannot
  always steal the same split.
- **SHAP for free:** TreeSHAP explains a prediction as “VPD added +0.04,
  NDVI subtracted −0.01, …”. The map’s cell panel uses that calculation
  (grouped into heat / moisture / fire history / …) plus the **calibrated
  chance**, a comparison to a typical day, and readable condition figures.
  It does not dump slogan-like SHAP lines.

It does **not** look at neighbouring pixels as an image. Each forest cell
is a row. Spatial structure enters only as features we computed
(distance to road, fire climatology of *this* cell, 8-neighbour
persistence as a *baseline* — not as a model input). That is why a U-Net
that *does* see 128 × 128 patches still lost 10/10 folds: the useful
signal here is “this cell’s weather and history,” not a texture of
ignition.

### What “learning” means in this project

The booster never sees a satellite photo of a flame. It sees 44 numbers
and a 0/1 that will become true *tomorrow*. Over millions of rows it
discovers splits that separate the two:

- High **VPD**, low **RH**, a long **dry spell**, a high **fire_clim**,
  recent **fires_3yr** → leaves packed with positives.
- High **NDVI** (green, wet fuel), rain in the last 7 days, January
  day-of-year → leaves packed with zeros.

Cohen’s *d* on the table already hints at this: `fire_clim` +0.84, `vpd`
+0.82, `rh` −0.76, recent-fire counts +0.68 to +0.73; `dist_road` −0.06
(almost nothing). Ablation confirms it: drop fire history and PR-AUC
collapses (−56%); drop roads and settlements and nothing measurable
happens.

That is **association**, not a physics engine and not a statement that
roads “do not cause fire.” It says: *given weather and history, distance
to a road does not help pick which day burns.*

### How inference is made (the production path)

`RiskPredictor` in `src/prometheus/models/predict.py`:

1. Load bundle `v1` (two boosters + two calibrators).
2. **Warm the season** (~12 s, ~3 GB RAM): rebuild every derived feature
   for Jan–May of that year from the cube + fire cube. Rolling windows
   and `days_since_fire` depend on the season *up to that day*; there is
   no cheaper honest way to score 15 April without walking 1 January →
   15 April.
3. Slice one day: 126,622 rows × 44 columns, in the booster’s column
   order.
4. `booster.predict(...)` → raw scores (a few hundred milliseconds).
5. Calibrator interpolates raw → **calibrated probability**.
6. Write a 465 × 912 grid; NaN off the forest mask. Optionally
   `digitize` into Low…Extreme.
7. `scripts/forecast.py` saves a Cloud-Optimized GeoTIFF
   (`risk_YYYY-MM-DD_h1.tif`), a 77-district GeoJSON, and a verification
   row.

After warm-up, a national map is well under a second. The **website does
not re-run the model** when you move the date slider. It paints GeoTIFFs
that were written once. Live “today” would mean: download this morning’s
ERA5 + MODIS + FIRMS, append a day to the cube, warm, predict. That hop
is not wired.

---

## Map click panel (`GET /api/explain`)

Clicking a forest cell on the map (or a district, at the click point) opens
the right-hand panel. That is **not** a SHAP poem. The API returns:

- **calibrated probability** and risk class (same isotonic map as the COG)
- **base rate** and a comparison: this cell vs the district mean vs Nepal
  forest today vs a typical forest day, plus a nationwide percentile
- a **snapshot** of conditions with units (°C, %, mm, days, kPa)
- **drivers**: TreeSHAP grouped into heat / humidity / rain / fire history /
  … as percent shares (so `days_since_fire` and `fire_clim` do not repeat)

`Tomorrow` / `Next 7 days` in that panel is the same `setHorizon` as the
left layer card. First click of a season still costs ~12 s while the cube
warms; later clicks are fast. Off-forest clicks are 400.

---

## A “what-if” prediction page — live at `/predict`

The **What if** tab on the website is this sandbox. Click a forest cell,
then drag weather sliders. LightGBM is tabular, so one 44-feature row plus
the calibrator is enough. The page **must not** say “fire will / will not
occur”; it shows a **calibrated chance** (often a few percent even in
Extreme).

**What it can say:** *under these conditions, the calibrated chance of a
FIRMS detection in a forest cell tomorrow (or in the next 7 days) is P,
in risk class C.*

**What it must not say:** “a fire will / will not occur.” Even Extreme
is a few percent, not a guarantee. Satellites miss small fires; the
label is a detection, not a burned-area perimeter.

### Why a sandbox is different from the map

The map answers *“this real place, on this real date, with weather that
actually happened.”* A prediction page answers *“what if VPD were
higher?”* Those are different products. The second is for intuition and
for a viva (“show me that dryness moves the score”). The first is the
forecast.

### What the user should type — and what you must fill in for them

All 44 inputs are required. Most people should not set all 44 by hand.
Two layers:

**1. Pick a place (and optionally a real date).**  
Click the map or enter lat/lon. Copy the **static and history** features
from the cube for that cell: elevation, slope, aspect, TWI, tree/shrub/
grass/crop fractions, distance to road/settlement, `fire_clim`,
`days_since_fire`, `fires_1yr/3yr/5yr`, `doy_sin/cos`. Optionally start
weather from that date too, then let the user nudge it. This keeps
combinations physically possible (you cannot put Terai temperatures on
an 4000 m ridge without also changing elevation and pressure).

**2. Sliders only for the levers that mean something to a human**,
clamped to ranges actually seen in `train_table.parquet` (use the 1st–
99th percentile, not the absolute min/max, so one sensor glitch cannot
open a wild slider). Suggested groups:

| Slider (user-facing) | Feature(s) behind it | Sensible Nepal pre-monsoon window (order of magnitude) |
|---|---|---|
| Max / mean air temperature | `t2m_max`, `t2m`, `t2m_min` | roughly −10 … 42 °C (keep min ≤ mean ≤ max) |
| Relative humidity | `rh` | 10 … 100 % |
| Vapour pressure deficit | `vpd` | 0 … ~4 kPa — **recompute from T and RH**, do not let it contradict them |
| Today’s rain | `precip` | 0 … ~50 mm |
| Rain last week / month | `precip_7d`, `precip_30d` | 0 … a few hundred mm; 7-day ≤ 30-day |
| Consecutive dry days | `consecutive_dry_days`, `days_since_rain` | 0 … ~150 (season length) |
| Wind | `wind_speed`, `wind_max_7d` | 0 … ~15 m/s |
| Greenness | `ndvi`, `evi`, `ndvi_anomaly` | NDVI −0.2 … 1; anomaly about −0.4 … +0.4 |
| Surface temperature | `lst_day`, `lst_night`, `lst_diff` | keep night ≤ day |
| Day of year | `doy_sin`, `doy_cos` | a date picker, Jan–May only — never a raw angle |

Lock or auto-fill: `surface_pressure` (almost elevation), `soil_water_l1`,
`u10`/`v10` (or derive from wind speed + a default direction), land cover,
terrain, human distances. Independent sliders for `t2m` and `t2m_max`
(r = 0.99) or `fires_3yr` and `fires_5yr` (r = 0.96) will create nonsense
rows the trees never saw.

**Hard rules for the page:**

- Reject coordinates off the **forest mask** (“this is not a modelled
  cell — city / ice / water / above 4500 m”).
- Clamp every numeric input to the training percentile range; show a
  warning if the user hits a rail.
- Keep **physical couples** in sync: RH + temperature → VPD; min/mean/max
  temperature ordered; 7-day rain ≤ 30-day rain; `doy_sin/cos` from a
  date, not two free knobs.
- Always run the **calibrator**. Raw LightGBM output on this problem is
  ~30× too high.
- Show **probability + risk class + SHAP bar** (“what moved this
  score”), not a yes/no fire icon. Optionally print the season base rate
  next to it (“typical forest day is ~0.8%”).
- State the season: Jan–May only. A “what if monsoon” slider would be
  extrapolation.

### How it is wired

The **What if** tab (`/predict`) talks to:

1. `GET /api/whatif/schema` — slider list and training percentile rails.
2. `POST /api/whatif` with `{lat, lon, date, horizon, overrides: {rh: 25, t2m_max: 34, ...}}`.
3. The same warmed season matrix as `/api/explain` (shared `RiskPredictor`, so
   the first click of a season still costs ~12 s / ~3 GB, not twice that).
4. Overrides are clamped; VPD and dewpoint are recomputed from T and RH;
   min ≤ mean ≤ max temperature; 7-day rain cannot exceed 30-day rain.
5. Response: baseline vs scenario calibrated probability, risk class, SHAP.

No new training. Place first (terrain, land cover, fire history locked),
weather second. A fully synthetic mode with no map click is not offered.

### What this page is good for

- Teaching: drag RH down and watch the probability rise.
- The viva / report: a controllable demo that the model is weather- and
  history-driven, not a static road map.
- Sensitivity: “how dry does this ridge need to be to enter Extreme?”

It does **not** replace the daily forecast, and it does not tell a
village to evacuate. Used with ranges from the training table and the
calibrated output, it is a legitimate, cheap extra view — and it fits
the existing LightGBM bundle without retraining.

---


## What has actually been built

This is not a mock-up. The numbers below come from files under `data/` and
`runs/`.

### Data

- **Fire labels:** NASA FIRMS (VIIRS + MODIS), Jan–May 2016–2026, rasterised to
  a 1 km Nepal grid (465 × 912 cells). ~821k seasonal detections.
- **Weather:** ERA5-Land, honestly downscaled with a lapse rate rather than
  fake 1 km detail.
- **Vegetation / heat:** MODIS NDVI and land-surface temperature, interpolated
  to daily.
- **Static:** SRTM terrain, WorldCover, distance to roads and settlements.
- **Forest mask:** burnable cover ≥ 25% and elevation ≤ 4500 m → 126,622 cells,
  keeping 96% of fire pixel-days.

### Model

- **Primary model: LightGBM** (gradient-boosted trees), two heads (1-day and
  7-day).
- **Evaluation:** leave-one-year-out over ten seasons (2016 is never held out —
  there is no prior year to build fire history from).
- **Result:** mean PR-AUC **0.155 vs climatology 0.043** (~+280% skill). Beats
  both climatology and persistence in **all ten** folds.
- **Ablation:** drop fire history and skill collapses (−56%). Drop roads and
  settlements and nothing measurable happens. That is the thesis in a table:
  static human-proximity maps explain *where* fires live over decades, not
  *which day* burns.
- **CNN (U-Net):** trained as a fair comparison. LightGBM won **10/10** folds.
  The convolutional net is research-only; it is not on the website.
- **Calibration:** isotonic regression on a held-out year. Raw scores were
  overconfident by ~30×; after calibration, expected calibration error drops
  by orders of magnitude. Only the calibrated map is shown.
- **Speed:** after a ~12 s season warm-up, a national map is well under a
  second.

### Product

- Daily **Cloud-Optimized GeoTIFFs** and 77-district GeoJSON.
- **FastAPI** tiles, district time series, active fires, verification, and
  “why this cell?” explanations.
- **React map** (Leaflet): play the season, toggle tomorrow vs next week,
  tap a district, see satellite 🔥 detections, read an accuracy page.

---

## Years on the website (2024, 2025, and 2026)

| Layer | Years | Role |
|---|---|---|
| Training / evaluation cube | **2016–2026** | The model *learned* from these seasons. 2026 is a real holdout fold (PR-AUC 0.144). |
| Frozen bundle `v1` | fit 2016–2023, stop 2024, calibrate 2025, **report 2026** | 2026 was used to *score* the shipped model. |
| Maps on the website | **Jan–May 2024, 2025, and 2026** | Daily GeoTIFFs in `runs/forecasts/`. The map **opens on 12 April 2026**. |

The first backfill was 2024–2025 (~303 days). 2026 is the same product: weather
and fire labels were already in the cube; writing the extra ~151 GeoTIFFs is
what puts the year on the map. Rebuild with:

```bash
source .prometheus-venv/bin/activate
python scripts/forecast.py --backfill 2026
python scripts/forecast.py --verify 2026-01-01 2026-05-30
```

`/api/forecasts` lists whatever years exist on disk. Year tabs show **2026
first**. There is still no live “today” feed — the site plays **history from
the cube**, not a fresh Earth Engine download for this morning.

---

## How to read the numbers (so nobody is fooled)

Fire is rare. On a typical forest day, well under 1% of cells burn. That makes
ordinary accuracy (ROC-AUC, “percent correct”) look excellent while still
being useless.

Prometheus leads with:

- **PR-AUC** — ranking skill under rarity. Climatology is ~0.04; we are ~0.15.
- **Top-10% capture** — of all fires, what share sat in the reddest tenth of
  the map? About **70%** in leave-one-year-out; daily verification is noisier
  (~55%).
- **Base rate** — always printed next to a score. A quiet day is not a model
  failure.
- **Skill vs climatology** — the only shipping rule: beat the historical
  calendar, or do not ship.

Anyone advertising pixel-level next-day PR-AUC above ~0.5 on this problem is
almost certainly leaking the future into the past.

---

## Review: how this compares to similar work

### 1. Nepali academic susceptibility maps

Examples: Rasuwa Random Forest (often cited AUC ~0.90), Chure–Tarai–Madhesh
fuzzy-AHP + RF (AUC ~0.95), Terai Arc Landscape RF, Palpa weighted-index maps,
Matin et al. 2017, Mishra et al. 2023 (*Fire Ecology*).

| | Typical Nepal paper | Prometheus |
|---|---|---|
| Question | Where *can* fire occur? | Where is fire likely *tomorrow*? |
| Time | One map for a decade | A new map every day of the season |
| Labels | Often burned-area or pooled points | Daily FIRMS, dilated, forest-only |
| Test | Often one district / one split | Ten leave-one-year-out seasons |
| Output | “High / low susceptibility” | Calibrated probability + risk class |
| Human features | Usually top-ranked | Ablation: **no measurable daily skill** |
| Product | Figure in a PDF | API + playable map + verification |

Those papers are not wrong. They answer a different question, and they do it
well. Prometheus is the operational complement: weather-driven, day-specific,
and willing to say when the model adds nothing (terrain, roads).

### 2. Fire-weather indices (FWI, NFDRS, FFDI)

Canada’s Fire Weather Index, the US National Fire Danger Rating System, and
Australia’s FFDI are **hand-designed weather formulae**. They are operational,
trusted, and not trained on Nepal’s 1 km grid. They do not ingest FIRMS
history, MODIS greenness, or district polygons. Prometheus is closer to a
**learned, local FWI** with satellite labels and a public accuracy ledger.

### 3. Global / regional wildfire ML (GWIS, EFFIS, US “risk to communities”)

- **GWIS / EFFIS** — excellent operational viewers; danger is mostly FWI-style
  or burned-area monitoring, not a Nepal-specific calibrated 1 km next-day
  model.
- **US Wildfire Risk to Communities** — again mostly **static** risk to
  assets, not “tomorrow in Bajhang.”
- **Recent deep-learning wildfire papers** (ConvLSTM, U-Nets on patches) —
  we ran that comparison. On this tabular geospatial problem, **LightGBM
  won**. Reporting that is stronger than hiding it.

### 4. What Prometheus does *not* claim

- It is not a suppression-dispatch system.
- It is not better than a local officer’s eyes on a specific hillside.
- It does not replace FWI where FWI is already the legal standard.
- Daily verification PR-AUC (~0.085) is **noisier** than season-level LOYO
  (~0.155). Both are shown on purpose.

**Verdict:** among Nepal-focused ML, Prometheus is unusual for being a
*forecast* with error bars, calibration, an ablation that contradicts the
“roads cause fire maps” story, and a working map. Among global operational
systems, it is smaller and local, but more honest about probabilities.

---

## What a first-time visitor should do on the site

1. Open the **Map**. Yellow is quieter; purple is the most dangerous forest
   that day.
2. Hit **Play season** and watch April move.
3. Switch **Tomorrow** vs **Next 7 days** on the left layer card (or in the
   right-hand cell panel — they share `?horizon=`).
4. Click a forest cell. The panel shows the **chance (%)**, a bar chart vs
   typical / Nepal / the district, weather figures, and grouped “what moved
   this score” percentages. **See all of …** opens the district page.
5. Open **Fires** to see flame marks where satellites already saw fire (that is
   history, not the forecast).
6. Open **What if**, click a forest cell, and drag humidity or heat — the
   chance should move, and it still will not claim a fire “will happen.”
7. Open **Accuracy** to see whether yesterday’s map caught today’s
   detections.

---

## What’s in this repo (every folder, file, size, and phase)

Sizes below are from a **full laptop build** on this machine (about **74 GB**
on disk). Git only holds the small code and docs (~1 MB of project files,
plus git history). `data/` and `runs/` are gitignored on purpose.

Think of the project in three layers:

1. **Code** — `src/`, `scripts/`, `frontend/src/`, `tests/`, `gee/`, `configs/`
   (~1 MB). Lives in git. Always free to copy.
2. **Build artefacts** — `data/` (~69 GB) and most of `runs/`. Built once.
   Needed to *train* and to *write* maps. **Not** needed to host the map once
   forecasts exist.
3. **Serve artefacts** — `runs/forecasts/` (~0.9 GB) + the 8 MB model bundle.
   This is what the website actually reads.

### Which phase uses which folder

The 3-week plan in [BUILD_PLAN.md](BUILD_PLAN.md) is still the map. Folders
were not created all at once; each belongs to a phase.

| Phase | Plan days | What happens | Folders that matter |
|---|---|---|---|
| **0. Design** | before Day 1 | Question, grid, season, model choice | `BUILD_PLAN.md`, `configs/` |
| **1. Scaffold** | Day 1 | Package, grid, tests that pin constants | `src/prometheus/config.py`, `grid.py`, `pyproject.toml`, `tests/test_config.py` |
| **2. Labels** | Day 2 | Download and rasterise FIRMS | `src/prometheus/data/`, `scripts/build_fire_labels.py`, `data/raw/firms/`, `data/cube/fire_daily.zarr` |
| **3. Eval harness** | Day 3 | Metrics, climatology, persistence | `src/prometheus/eval/`, `scripts/run_baselines.py`, `runs/baselines/`, `data/cube/climatology_doy.npz` |
| **4. Download** | Days 4–5 | GEE weather/veg/static + OSM roads | `gee/`, `docs/DAY4_5_MANUAL.md`, `data/raw/gee/`, `data/raw/osm/`, `scripts/build_local_static.py`, `data/static/` |
| **5. Cube + table** | Days 6–8 | Align everything, sample training rows | `src/prometheus/features/`, `scripts/build_feature_cube.py`, `scripts/build_train_table.py`, `data/cube/` |
| **6. Model** | Days 9–12 | Train, CV, calibrate, compare U-Net | `src/prometheus/models/`, `cnn/`, `scripts/train_*.py`, `run_cv.py`, `build_model_bundle.py`, `data/models/`, `runs/cv|shap|unet|calibration|lightgbm/` |
| **7. Product** | Days 13–15 | Daily maps, API, React site | `src/prometheus/infer/`, `api/`, `scripts/forecast.py`, `run_api.py`, `runs/forecasts/`, `frontend/` |
| **8. Report / ops** | ongoing | Paper, this tour, $0 deploy notes | `docs/`, `explanation.md`, `PROGRESS_REPORT.md`, `tests/` (kept green all along) |

A test file is listed with the phase whose *rule* it protects (alignment,
no leakage, API contracts), even though tests were added throughout.

---

### Whole-disk snapshot

| Path | Size | In git? | Importance |
|---|---|---|---|
| `data/` | **~69 GB** | no | Factory. Train and backfill. |
| `runs/` | **~1.6 GB** | no | Experiments + the live map’s GeoTIFFs. |
| `.prometheus-venv/` | ~1.7 GB | no | Local Python 3.12 environment. |
| `.git/` | ~655 MB | — | History only. |
| `frontend/node_modules/` | ~230 MB | no | npm packages. Rebuild with `npm install`. |
| `frontend/` without node_modules | ~1 MB | yes (source) | The website. |
| `src/` | ~700 KB | yes | The Python library. |
| `tests/` | ~364 KB | yes | 13 files, 100 tests. |
| `scripts/` | ~112 KB | yes | One CLI per job. |
| `docs/` | ~52 KB | yes | Paper draft + GEE/2026 manuals. |
| `explanation.md` | ~48 KB | yes | This file. |
| `gee/` | ~20 KB | yes | Earth Engine export recipes. |
| `configs/` | ~8 KB | yes | The single source of truth. |
| Root markdown / Makefile / `pyproject.toml` | small | yes | How to run and what we intended. |

---

### Root (project brain)

Phase: **0–1 (design + scaffold)**, then used forever.

| Path | Size | Phase | Use |
|---|---|---|---|
| `README.md` | ~12 KB | 1, 7 | Install, run API + UI, troubleshooting. |
| `explanation.md` | ~48 KB | 8 | Beginner tour, pipeline, this catalogue. |
| `BUILD_PLAN.md` | ~28 KB | 0 | Original 3-week design (intent). |
| `PROGRESS_REPORT.md` | ~40 KB | 8 | Measured results day by day (what we actually got). |
| `Makefile` | ~4 KB | 7 | `make api`, `make ui`, `make forecast`, backfill, verify. |
| `pyproject.toml` | ~4 KB | 1 | Package name, Python deps (LightGBM, FastAPI, rasterio, …). |
| `.gitignore` | ~4 KB | 1 | Keeps `data/`, `runs/`, secrets, venv, rasters out of git. |
| `.python-version` | tiny | 1 | Pins Python 3.12 for local tooling. |

---

### `configs/` (~8 KB)

Phase: **1 — scaffold**. Loaded on every later command.

| Path | Use |
|---|---|
| `configs/base.yaml` | **Only place** years (2016–2026), Jan–May, grid, feature list, LightGBM knobs, CV years, risk-class quantiles may live. `src/prometheus/config.py` reads it. |

If a number disagrees with this file, the file is wrong.

---

### `gee/` (~20 KB)

Phase: **4 — download (Days 4–5)**. These are *recipes*, not data. They run
inside [Google Earth Engine](https://code.earthengine.google.com/). You paste,
click Run, later copy GeoTIFFs from Drive into `data/raw/gee/`.

| Path | What it exports | Native scale | Application |
|---|---|---|---|
| `00_roi.js` | Shared Nepal bounding box | — | Copied into the others so the ROI cannot drift. |
| `era5_daily.js` | Daily weather, one stack per month | **~9 km (11,132 m)** | Honest weather; 1 km would be fake. |
| `lst_8day.js` | MODIS land-surface temperature | 1 km | Day/night heat, interpolated later. |
| `ndvi_16day.js` | MODIS NDVI / EVI | 1 km | Greenness / fuel. |
| `static.js` | SRTM elev/slope/aspect/TWI + WorldCover fractions | 1 km | Terrain and land cover, once. |

Hands-on clicks: [docs/DAY4_5_MANUAL.md](docs/DAY4_5_MANUAL.md).

---

### `src/prometheus/` (~700 KB, ~40 `.py` files)

Phase: **1 through 7**. This is the library every `scripts/*.py` imports.
Nothing here is a GeoTIFF; it *reads and writes* those under `data/` and
`runs/`.

| Path | Size | Phase | Importance | What it does |
|---|---|---|---|---|
| `config.py` | ~8 KB | 1 | Load-bearing | Parses `base.yaml` into typed settings. |
| `grid.py` | ~4 KB | 1 | Load-bearing | 465×912, EPSG:4326, `assert_aligned`. Every raster must match. |
| `data/firms.py` | ~68 KB | 2 | Load-bearing | FIRMS API download, clean, rasterise, dilate, write `fire_daily.zarr`. |
| `features/warp.py` | | 5 | | Regrid any GeoTIFF onto the canonical grid. |
| `features/weather.py` | | 5 | | ERA5 9 km → 1 km with lapse-rate T / RH / VPD / wind. |
| `features/vegetation.py` | | 5 | | MODIS composites → daily NDVI/LST (time fill + interpolate). |
| `features/forest.py` | | 5 | Load-bearing | Forest mask: burnable ≥ 25% and elev ≤ 4500 m. |
| `features/derived.py` | | 5 | | Dry spells, rolling windows, fire history, NDVI anomaly, h1/h7 labels. |
| `features/cube.py` | | 5 | | Assemble `features_daily.zarr`. Strips Finder `.DS_Store` on open. |
| `features/table.py` | | 5–6 | Load-bearing | Sample `train_table.parquet`; also full-grid matrices for scoring. |
| `eval/metrics.py` | | 3 | Load-bearing | PR-AUC, Brier, top-10% capture. |
| `eval/baselines.py` | | 3 | Shipping rule | Climatology (MODIS 2003–2015) and 7-day persistence. |
| `eval/cv.py` | | 6 | | Leave-one-year-out + family ablations + SHAP summary. |
| `models/lgbm.py` | | 6 | Load-bearing | Train one fold, score a grid, search, save/load boosters. |
| `models/calibrate.py` | | 6 | Load-bearing | Isotonic map raw score → real probability; risk classes. |
| `models/bundle.py` | | 6–7 | | Versioned `v1` on disk (trees + calibrators + feature names). |
| `models/predict.py` | | 7 | | `RiskPredictor`: warm a season, return a 465×912 probability field. |
| `cnn/unet.py`, `cnn/data.py`, `cnn/stacks.py` | ~68 KB | 6 (Day 12) | Research only | Fair U-Net comparison. Lost 10/10. Not on the website. |
| `infer/forecast.py` | | 7 | | One date → COGs + district GeoJSON. Backfill a year. |
| `infer/io_cog.py` | | 7 | | Write/read Cloud-Optimized GeoTIFFs (`nodata = −1`). |
| `infer/districts.py` | | 7 | | 77 OSM districts, zonal mean/max risk. |
| `infer/verify.py` | | 7 | | Score yesterday’s map against next-day FIRMS → `verification.csv`. |
| `api/app.py` | | 7 | | FastAPI app; `PROMETHEUS_FORECASTS_ROOT`. |
| `api/runtime.py` | | 7 | | One cached `RiskPredictor` for explain and what-if. |
| `api/routes/meta.py` | | 7 | | `/health`, `/forecasts` (date catalogue; default **2026-04-12**). |
| `api/routes/risk_tiles.py` | | 7 | | XYZ PNG tiles from a day’s COG. |
| `api/routes/districts.py` | | 7 | | District GeoJSON + timeseries (cached `_district_ts.json`). |
| `api/routes/fires.py` | | 7 | | Active FIRMS points from `fire_daily.zarr`. |
| `api/routes/verification.py` | | 7 | | Accuracy table for the Accuracy page. |
| `api/routes/explain.py` | | 7 | Heaviest | Calibrated chance + snapshot + grouped SHAP. Cube + model in RAM (~3–4 GB first call). |
| `api/routes/whatif.py` | | 7 | | Weather sandbox. Shares the predictor with explain (`api/runtime.py`). |

---

### `scripts/` (~112 KB, 17 commands)

Phase: **the day in the table**. Each script is a thin CLI over the library.
You almost never import these; you run them.

| Script | Phase | Writes / reads | Application |
|---|---|---|---|
| `build_fire_labels.py` | 2 | → `data/cube/fire_daily.zarr` | Labels. |
| `build_local_static.py` | 4 | OSM → `data/static/dist_road.tif` etc. | Human proximity rasters. |
| `audit_gee_raw.py` | 4 | Inspects `data/raw/gee/` | Deduplicate Drive exports. |
| `build_feature_cube.py` | 5 | → `features_daily.zarr` | The big aligned cube. |
| `plot_cube_check.py` | 5 | figures under `runs/cube/` | Sanity plots. |
| `build_train_table.py` | 5 | → `train_table.parquet` | LightGBM’s table. |
| `plot_feature_diagnostics.py` | 5 | `runs/features/` | Cohen’s *d*, correlations. |
| `run_baselines.py` | 3 | `runs/baselines/` | Climatology / persistence scores. |
| `train_lightgbm.py` | 6 | `data/models/lgbm_holdout*.txt` | One fold or random search. |
| `run_cv.py` | 6 | `runs/cv/` | Ten LOYO folds + ablations. |
| `plot_shap.py` | 6 | `runs/shap/` | Beeswarm / dependence for the report. |
| `build_model_bundle.py` | 6 | `data/models/bundles/v1/` | Freeze production h1 + h7 + calibrators. |
| `plot_calibration.py` | 6 | `runs/calibration/` | Reliability curves. |
| `train_unet.py` | 6 | `runs/unet/*.pt` | Optional CNN. |
| `forecast.py` | 7 | `runs/forecasts/` | Daily maps, backfill, verify. |
| `run_api.py` | 7 | serves `:8000` | Website backend. |
| `check_ui_api.py` | 7 | read-only | Smoke-check UI routes vs API. |

---

### `tests/` (~364 KB, 11 files)

Phase: **written with the matching day**, run on every later change. They do
not train models; they pin rules so a refactor cannot silently leak the
future or misalign a raster.

| File | Phase it guards | What would break without it |
|---|---|---|
| `test_config.py` | 1 | Wrong years, grid, or mask cell count. |
| `test_firms.py` | 2 | Fire pixels outside Nepal; dirty labels. |
| `test_metrics.py` | 3 | A “better” metric that flatters rarity. |
| `test_static_alignment.py` | 4 | Roads/terrain shifted by one pixel. |
| `test_cube.py` | 5 | Time/space holes in the feature cube. |
| `test_table.py` | 5 | Train/infer feature mismatch; leakage in sampling. |
| `test_models.py` | 6 | Fold split leaking the holdout year. |
| `test_cv.py` | 6 | Ablation / SHAP plumbing. |
| `test_calibration.py` | 6 | Calibrator fitted on the 1:20 table (wrong base rate). |
| `test_forecast.py` | 7 | Misaligned COGs; non-idempotent backfill. |
| `test_api.py` | 7 | Tiles outside Nepal opaque; empty fires crashing; catalogue. |
| `test_explain.py` | 7 | Grouped drivers, snapshot units, comparison copy. |
| `test_whatif.py` | 7 | What-if rails, VPD coupling, off-mask 400. |

---

### `docs/` (~52 KB)

Phase: **4 and 8** (manuals during download; paper at the end).

| Path | Size | Phase | Use |
|---|---|---|---|
| `DAY4_5_MANUAL.md` | ~10 KB | 4 | Click-by-click GEE + OSM download. |
| `EXTEND_TO_2026.md` | ~3 KB | 5–6 | Why the train window grew to 2026. |
| `academic_report_draft.md` | ~33 KB | 8 | Conference-style paper draft. |

---

### `frontend/` (~231 MB with `node_modules`; **~1 MB** source)

Phase: **7 — product (Day 15)**. React 19 + Vite + Leaflet. Dev server
`:5173` proxies `/api` to FastAPI `:8000`.

| Path | Size | Use |
|---|---|---|
| `package.json` / `package-lock.json` | small / ~140 KB | React, Leaflet, Vite, lucide icons. |
| `vite.config.js` | tiny | Dev proxy `/api` → `127.0.0.1:8000`. |
| `index.html` | tiny | Shell; title and favicon. |
| `eslint.config.js` | tiny | Lint. |
| `Design.md` / `PLAN.md` | ~16 KB each | Visual system (Arcade Night) + original UI plan. |
| `README.md` | tiny | Frontend-only notes. |
| `public/favicon.svg`, `logo.svg` | tiny | Branding. |
| `public/nepal-border.geojson` | | Country outline on the Fires page. |
| `public/patch_grid.geojson` | | Leftover prototype grid; not the live risk layer. |
| `src/main.jsx` / `App.jsx` | | Boot + routes: Map `/`, What if `/predict`, District, Fires, Accuracy. |
| `src/pages/MapPage.jsx` | | National forecast map (opens on **2026** first). |
| `src/pages/PredictPage.jsx` | | What-if weather sandbox (`/predict`). |
| `src/pages/DistrictPage.jsx` | | One district’s season line. |
| `src/pages/FiresPage.jsx` | | Satellite 🔥 detections (history, not the forecast). |
| `src/pages/VerifyPage.jsx` | | Did yesterday’s map catch today’s fires? |
| `src/Components/map/` | | Leaflet: risk tiles, districts, scrubber, legend, explain drawer, stats. |
| `src/Components/chrome/` | | Header, theme toggle, “API down” banner. |
| `src/Components/charts/` | | District timeseries + verification sparkline. |
| `src/Components/ui/` | | Button, Card, Tabs. |
| `src/api/client.js` | | `getForecasts`, `getDistricts`, `getExplain`, `getWhatIfSchema`, `postWhatIf`, tile URLs. |
| `src/state/ForecastContext.jsx` | | Shared `?date=&horizon=`; year tabs newest-first. Horizon tabs on the map **and** in the cell panel both call `setHorizon`. |
| `src/lib/` | | Nepal bounds, default date `2026-04-12`, plain-language feature names, colours, flame mark. |
| `src/theme/` / `styles/tokens.css` | | Light / dark tokens. |
| `dist/` | ~944 KB | Production build (`npm run build`). What you deploy. |
| `node_modules/` | ~230 MB | Dependencies. Not source. |

The **What if** page (`/predict`, `PredictPage.jsx`) is the weather sandbox:
click a forest cell, then move sliders. API: `GET /api/whatif/schema`,
`POST /api/whatif`.

---

### `data/` — **~69 GB, gitignored, build-only**

Phase: **2, 4, 5, 6**. Needed to train and to *generate* maps. **Not** needed
on a host if `runs/forecasts/` already exists — except `/api/explain` and
live fires, which still want the fire/feature cubes.

#### `data/raw/` (~4.7 GB) — the downloads (phase 2 + 4)

| Path | Size | Phase | What it is |
|---|---|---|---|
| `data/raw/gee/era5/` | ~232 MB | 4 | Monthly ERA5 stacks at ~9 km. |
| `data/raw/gee/lst/` | ~539 MB | 4 | 8-day LST composites. |
| `data/raw/gee/ndvi/` | ~428 MB | 4 | 16-day NDVI/EVI composites. |
| `data/raw/gee/static/` | ~11 MB | 4 | SRTM + WorldCover from GEE. |
| `data/raw/firms/` | ~187 MB | 2 | FIRMS API chunk CSVs + archives. |
| `data/raw/osm/` | ~3.3 GB | 4 | Geofabrik Nepal extract (roads, places, admin). Large because it is the full dump, not just the three distance rasters. |

Once the cube exists you can archive `data/raw/` off the laptop. The model
does not read it again.

#### `data/static/` (~3.1 MB) — aligned 1 km layers (phase 4)

| File | Approx size | Use |
|---|---|---|
| `nepal_mask_1km_roiAligned.tif` | ~8 KB | Which cells are Nepal. |
| `elevation_static_srtm.tif` | ~687 KB | Height (lapse-rate + forest mask). |
| `slope_static_srtm.tif` | ~1.6 MB | Slope. |
| `dist_road.tif` / `dist_settlement.tif` | ~60–90 KB | Human proximity (ablation: little daily skill). |
| `physio_regions.tif` + `.json` | tiny | Terai / Chure / Middle / High Mountains proxy. |
| `districts_77.geojson` | ~696 KB | Cached 77-district polygons for zonal stats. |

WorldCover fractions also live as a GEE static GeoTIFF, warped in
`features/forest.py`.

#### `data/cube/` (~65 GB on this disk) — the aligned stack (phase 2 + 5)

| Path | Size | Phase | Importance |
|---|---|---|---|
| `fire_daily.zarr` | **~1.8 MB** | 2 | Daily 0/1 labels. Tiny because it is uint8 and sparse. |
| `features_daily.zarr` | **~5.3 GB** | 5 | Weather + veg + static, 1664 days × 465 × 912. **The cube the model trains and explains from.** |
| `train_table.parquet` | ~143 MB | 5 | ~2.1 million sampled rows for LightGBM. |
| `climatology_doy.npz` | ~13 MB | 3 | MODIS 2003–2015 day-of-year fire rate (`fire_clim` + baseline). |
| `features_daily_report.json` / `train_table_report.json` | tiny | 5 | Build logs. |
| `stacks/` | **~59 GB** | 5 leftover | Intermediate warps. **Safe to delete** once `features_daily.zarr` is written; they are not read at inference. |

#### `data/models/` (~77 MB) — trees (phase 6)

| Path | Size | Use |
|---|---|---|
| `bundles/v1/` | **~8 MB** | **Production.** `lgbm_h1.txt`, `lgbm_h7.txt`, `manifest.json`, `MODEL_CARD.md`. The only models the API should load. |
| `lgbm_holdoutYYYY.txt` | ~4–7 MB each | Per-fold CV artefacts. Report-only. |
| `norm_stats_v1.json` | ~76 KB | Per-fold means/stdevs (U-Net / docs; trees do not need scaling). |

---

### `runs/` — **~1.6 GB, gitignored**

Phase: **3, 6, 7**. Experiments stay here; the website only needs
`forecasts/`.

| Path | Size | Phase | Served to the website? | What it is |
|---|---|---|---|---|
| `runs/forecasts/` | **~913 MB** | 7 | **Yes** | 454 days × (`risk_*_h1.tif`, `risk_*_h7.tif`, `districts_*.geojson`) + `verification.csv`. Map tiles, district colours, Accuracy page. |
| `runs/unet/` | ~553 MB | 6 | no | Ten U-Net checkpoints + `per_fold_h1.csv`. Research. |
| `runs/calibration/` | ~206 MB | 6 | no | Reliability `.npz`/`.png`, risk-class tables. |
| `runs/shap/` | ~668 KB | 6 | no | SHAP plots for 2021. |
| `runs/features/` | ~332 KB | 5 | no | Feature diagnostic plots. |
| `runs/cube/` | ~284 KB | 5 | no | Cube-check figures. |
| `runs/cv/` | ~264 KB | 6 | no | `per_fold.csv`, `ablation.csv`, SHAP CSVs — the results tables. |
| `runs/lightgbm/` | ~12 KB | 6 | no | 2021 fold search + importance. |
| `runs/baselines/` | ~12 KB | 3 | no | Climatology / persistence metrics. |

`runs/forecasts/` also grows `_district_ts.json` the first time someone
opens a district page (a cache rebuilt if any `districts_*.geojson` is newer).

---

### Not source (ignore when reading the science)

| Path | Size | Why it exists |
|---|---|---|
| `.prometheus-venv/` | ~1.7 GB | `python3.12 -m venv`. Recreate from `pyproject.toml`. |
| `frontend/node_modules/` | ~230 MB | `npm install`. |
| `.pytest_cache/` / `.ruff_cache/` / `.mplconfig/` | small | Tool caches. |
| `.DS_Store` | — | macOS folder metadata. Cubes call `strip_finder_junk` on open so Zarr v3 does not warn. |

---

### One-line “what do I actually need?”

| If you want to… | Keep |
|---|---|
| Read the report | `docs/`, `PROGRESS_REPORT.md`, `runs/cv/` |
| Retrain LightGBM | `data/cube/train_table.parquet` + `configs/` + `src/` |
| Rebuild maps | cube + `data/models/bundles/v1/` + `scripts/forecast.py` |
| Run the website | `runs/forecasts/` + `frontend/` + FastAPI. Cube only for Explain / live Fires. |
| Copy the project to git | everything except `data/`, `runs/`, venv, `node_modules` |

---

## How the pieces talk to each other

```
configs/base.yaml
        │
        ▼
GEE + FIRMS ──► data/cube/*.zarr ──► train_table ──► LightGBM bundle (~8 MB)
                                                      │
                                                      ▼
                                            scripts/forecast.py
                                                      │
                                                      ▼
                                            runs/forecasts/*.tif + districts
                                                      │
                          ┌───────────────────────────┴───────────────────────────┐
                          ▼                                                       ▼
                   FastAPI (:8000)                                         React (:5173)
                   tiles / districts /                                   paints the map
                   fires / verify / explain
```

- **Map colours** → COG tiles under `runs/forecasts/` (no cube required at request time).
- **District borders / stats** → `districts_{date}.geojson`.
- **Accuracy page** → `verification.csv`.
- **Fires page** → currently reads `fire_daily.zarr` (cube). For a free demo,
  pre-export a small GeoJSON instead (below).
- **Why this spot?** → needs the model **and** the feature cube. Heaviest
  endpoint; disable or precompute a few cells for free hosting.

---

## Going live for $0 — without the big 5 GB files

### The idea

You do **not** deploy `data/cube/` or `data/raw/`. Those are factory tools.

You deploy a **slim demo pack** of pre-written maps + a static (or tiny)
backend. Training stays on your laptop forever.

### What actually has to be online

| Need for the public site | Keep? | Notes |
|---|---|---|
| Feature cube (~5+ GB) | **No** | Already baked into the GeoTIFFs |
| Raw GEE / FIRMS downloads | **No** | Build-only |
| Train table / CV / U-Net | **No** | Report-only |
| Full 454-day forecast folder (~0.9 GB) | Optional | Too fat for many free disks |
| **Peak-season subset** (e.g. 1–30 Apr 2025) | **Yes** | ~30 days × ~1.4 MB ≈ **40–50 MB** of COGs + GeoJSON |
| Model bundle `v1` (~8 MB) | Only if you keep live `/explain` | Otherwise skip |
| Pre-baked fire points GeoJSON | **Yes** (small) | Replaces the fire cube on the server |
| Frontend `dist/` | **Yes** | A few MB after `npm run build` |

### Recommended free stack (no credit card required)

| Piece | Free option | Role |
|---|---|---|
| Frontend | **Cloudflare Pages**, **Netlify**, or **GitHub Pages** | Hosts the React build + static JSON/tiles |
| Fat-ish files (optional) | **Hugging Face Datasets** (free) or **Cloudflare R2** free tier | Store the 40–50 MB April pack if Pages’ limit bites |
| API (optional) | **Render** free web service, **Fly.io** free allowance, or **Hugging Face Spaces** | Only if you still want FastAPI |
| Best $0 path | **Static-only** | No Python process at all |

GitHub’s soft file limit (~100 MB) and LFS quotas make dumping full COGs into
git a bad idea. Prefer Pages/Netlify for the UI and HF/R2 (or a zipped release
asset) for the demo rasters.

### Path A — fully static (simplest, truly free)

Do this once on your laptop (where the big files already exist):

1. **Pick a demo window** — e.g. April 2025 (peak fire month), ~30 days.
2. **Copy only those artefacts** into something like `demo/forecasts/`:
   - `risk_2025-04-XX_h1.tif` + `_h7.tif`
   - `districts_2025-04-XX.geojson`
   - a trimmed `verification.json` (summary + those days’ rows)
3. **Pre-render map tiles to PNG** (one-time), e.g. with `gdal2tiles` or a tiny
   script over the COGs → `demo/tiles/{date}/{h}/{z}/{x}/{y}.png`. PNG tiles
   are what Leaflet already expects; then the browser never needs GeoTIFF
   tooling on the server.
4. **Export fire points** for that window once to `demo/fires.geojson` so the
   Fires page does not open `fire_daily.zarr`.
5. **Turn off or stub** `/explain` (or precompute SHAP for 5 famous clicks into
   JSON). Explain is the only feature that still wants the cube + model in RAM.
6. **Build the frontend** against static URLs:
   ```bash
   cd frontend && npm run build
   ```
   Point `VITE_API_BASE` (or a small static client) at `/demo/...` files on the
   same origin instead of `http://127.0.0.1:8000`.
7. **Deploy `frontend/dist` + `demo/`** to Cloudflare Pages / Netlify / GitHub
   Pages.

Result: a playable map, district colours, scrubber, fires, and accuracy — **no
GPU, no 5 GB disk, no paid plan**. What you lose: live “today”, the full
2024–2026 scrub, and on-the-fly cell explanations.

### Path B — tiny FastAPI + slim forecast pack

If you want the real API with almost no changes:

1. Keep only the April (or two-week) COGs + district GeoJSON + `verification.csv`
   under `runs/forecasts/` on the host (~50 MB).
2. Ship `data/models/bundles/v1/` (~8 MB) **only** if explain must work — and
   still skip the cube; disable `/api/explain` and `/api/fires/active` or point
   fires at a static GeoJSON.
3. Run `scripts/run_api.py` on Render/Fly/HF Spaces with env
   `PROMETHEUS_FORECASTS_ROOT` aimed at that slim folder.
4. Host the React app on Pages; set the API URL to the free backend; widen CORS
   beyond localhost.

Watch free-tier sleep: Render’s free web service spins down after idle. The
first map load can take ~30–60 s to wake. For demos, Path A feels snappier.

### Path C — “looks live” with almost no backend

- Host the UI on Pages.
- Store one **GeoJSON of district risk per day** (already small) and skip raster
  tiles entirely for the demo — colour districts only. Instant, tiny, free.
- Keep one national PNG poster per day if you still want a continuous field.

Good for a supervisor demo when bandwidth is the bottleneck.

### What not to do on free tier

- Do **not** upload `features_daily.zarr` or `data/raw/`.
- Do **not** expect free hosts to run `predict()` for every date on the fly —
  season warm-up alone wants ~3–4 GB RAM.
- Do **not** commit COGs into git; use object storage or a release zip.
- Do **not** promise a free “today’s forecast from GEE” — Earth Engine + cold
  starts break the $0 story. Pre-bake days instead.

### Honest scope of a $0 deploy

| | Laptop (full) | Free public demo |
|---|---|---|
| Years on scrubber | 2024–2026 (opens on 2026) | One peak month is enough |
| Tile source | On-the-fly from COGs | Pre-tiled PNG or slim COGs |
| Fires | Live from zarr | Static GeoJSON |
| Explain | Live chance + grouped drivers | Off, or a handful of precomputed cells |
| New “today” | Needs GEE + cube | Manual bake when you care |
| Cost | Your electricity | $0 |

That is still a real product for a Year-III showcase: the science lived in the
cubes once; the public site only needs the **answers** you already wrote to
disk.

---

## Where to go next in the repo

| File | What it is |
|---|---|
| [README.md](README.md) | Install and run |
| [BUILD_PLAN.md](BUILD_PLAN.md) | Original 3-week design |
| [PROGRESS_REPORT.md](PROGRESS_REPORT.md) | Every measured result |
| [docs/academic_report_draft.md](docs/academic_report_draft.md) | Conference-style paper draft |
| [frontend/PLAN.md](frontend/PLAN.md) | How the website is wired |
| `configs/base.yaml` | Years, grid, features — the only place those should live |
| This file, **pipeline / LightGBM / what-if** | Download → cube → train → infer, how the trees learn, and the `/predict` sandbox |
| This file, **What’s in this repo** | Every folder and file: size, importance, use, and which build phase |
