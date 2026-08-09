# Extending training data through 2026

We cut at 2025 earlier because the build plan was drafted for a frozen 10-year window.  
**You are now past May 2026**, so the full pre-monsoon season (Jan–May 2026) should be available for FIRMS, MODIS, and ERA5-Land.

## What changed in the repo

| File | Change |
|---|---|
| `configs/base.yaml` | `train_end: 2026`, `years.all` and `cv.years` include 2026 |
| `src/prometheus/data/firms.py` | FIRMS windows follow config through `train_end` |
| `gee/*_*.js` | `END_YEAR = 2026` |
| Tests / baseline CLI | Expect 11 years |

Static layers (OSM, elev, WorldCover) do **not** need a redo for 2026.

## Do not restart from Day 1

| Already done | Action |
|---|---|
| Day 1 scaffold | None |
| Day 2 fire labels 2016–2025 | **Re-run** labels (adds 2026 + rebuilds cube) |
| Day 3 baselines | **Re-run** after new zarr (optional now, must before final numbers) |
| Day 4–5 GEE 2016–2025 in progress | **Add 2026 only** (set `START_YEAR = END_YEAR = 2026` in each JS) |
| OSM / dist static | None |

## Exact steps (do these)

### 1. Labels (FIRMS) — required

Cached chunks for 2016–2025 are reused automatically. Only ~2026 season downloads.

```bash
cd /Users/sammit/Desktop/Projects/Prometheus
source .prometheus-venv/bin/activate
python scripts/build_fire_labels.py
```

Expect:

- Detection table includes **2026** Jan–May columns/rows  
- Cube time length ≈ **11 × ~151** days (~1661, not exactly if leap days / calendar)  
- Outside-mask fires still **0**

If download is slow, leave it overnight; MAP_KEY same as before.

### 2. GEE predictors — required for 2026 season

**Do not re-export all years** if 2016–2025 already started/succeeded.

In each script (`era5_daily.js`, `lst_8day.js`, `ndvi_16day.js`), temporarily:

```javascript
var START_YEAR = 2026;
var END_YEAR = 2026;
```

Then Run → Start tasks only. For ERA5 that is **5** monthly files (Jan–May 2026).  
Static GEE: skip (one-time).

Caveats:

- **ERA5-Land** sometimes lags a few weeks; by Aug 2026, Jan–May is almost always complete. If a month task fails “no images”, wait a week or check the collection date range in the EE catalog.  
- **NDVI:** use the fixed script (no `clip`).

### 3. Baselines — after labels zarr exists

```bash
python scripts/run_baselines.py
```

Ship bar will be whatever the **new** climatology vs LOYO mean says (includes 2026 fold). Save fresh CSVs under `runs/baselines/`.

### 4. Day 6+ feature cube / LightGBM

When you build features, they must span the same years as the fire cube. If a feature pipeline hardcodes 2025, change it to `cfg.years` / `load_settings().years.train_end`. Prefer config — already the source of truth.

## What 2026 is good for

- Extra LOYO fold (more honest mean ± std)  
- More recent fuel/weather/fire patterns  
- Operational story: “trained through last fire season”

You can still **hold out 2026** for a final report table (“train 2016–2025, test 2026”) later without deleting the data — that is an eval *choice*, not a download choice. Default LOYO uses all `cv.years`.

## Checklist

- [ ] `cfg.years` ends at 2026 (`python -c "from prometheus.config import cfg; print(cfg.years)"`)  
- [ ] `build_fire_labels.py` finished; 2026 in detection table  
- [ ] ERA5 + LST + NDVI 2026 on Drive → `data/raw/gee/`  
- [ ] `run_baselines.py` re-run  
- [ ] Feature train years match cube when you reach Day 6  
