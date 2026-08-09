# Prometheus — Progress Report

Living log of what was built, cleaned, and verified.

| Field | Value |
|---|---|
| Project | Prometheus — daily wildfire risk for Nepal |
| Branch | `v2` |
| Last updated | 2026-08-09 |

---

## Repository layout

```
configs/  src/  scripts/  tests/  frontend/  docs/
data/          # gitignored (static, firms, cubes)
runs/          # gitignored experiment outputs
BUILD_PLAN.md  PROGRESS_REPORT.md  README.md  pyproject.toml
```

---

## Day 1 — Scaffold · Done

Package + `configs/base.yaml` + 1 km Nepal grid/mask.

---

## Day 2 — Fire labels · Done

FIRMS Area API (day_range **1–5** only; date = window start).

| Metric | Result |
|---|---|
| Download | 1271 chunks in ~78 min |
| Raw → cleaned | 967,983 → **803,025** |
| Jan–May 2016–2025 | **760,696** (>120k **PASS**) |
| Sensors | MODIS 47,920 · VIIRS SNPP 406,220 · VIIRS NOAA-20 306,556 |
| Cube | `(1513, 465, 912)` · fire pixels 1,768,527 · **0 outside mask** |
| April peak | Yes (e.g. 2024 Apr = 100,663 detections) |

Command: `python scripts/build_fire_labels.py`

---

## Day 3 — Evaluation harness + baselines · Done

### Implemented
- `src/prometheus/eval/metrics.py` — PR-AUC, ROC-AUC, Brier, skill vs clim, top-k capture, reliability, ECE  
- `src/prometheus/eval/baselines.py` — MODIS 2003–2015 doy climatology (±7 d temporal, σ=1 spatial); 7-day + 3×3 persistence  
- `src/prometheus/eval/cv.py` — per-year metrics 2016–2025, mean ± std  
- `scripts/run_baselines.py`

### Results (mean over 10 years, Nepal-mask pixels)

```
model            PR-AUC  ROC-AUC    Brier top10%-capture
climatology      0.0419   0.8091   0.0069         0.5553
persistence      0.0503   0.7975   0.0585         0.6803
```

| Model | Notes |
|---|---|
| **Climatology** | Soft maps from historical MODIS seasonality. Strong ROC (~0.81). Tops 10% risk areas catch **~56%** of fires. Lowest Brier (well-scaled probs). |
| **Persistence** | Higher **PR-AUC** and **top-10% capture (~68%)** — recent fire + neighbours is a strong short-term signal. Worse Brier (hard 0/1 scores). |

**Shipping rule:** a new model must beat **climatology PR-AUC (0.0419)** on this LOYO protocol (and preferably top-10% capture). Report mean ± std from `metrics_per_year.csv`.

### Outputs
| Path | |
|---|---|
| `runs/baselines/metrics_table.txt` | Display table |
| `runs/baselines/metrics_summary.csv` | Mean ± std |
| `runs/baselines/metrics_per_year.csv` | Per year |
| `data/cube/climatology_doy.npz` | Cached clim |

### Command

```bash
source .prometheus-venv/bin/activate
python scripts/run_baselines.py
```

---

## Alignment rule

Shape 465×912, EPSG:4326, zero positives outside Nepal mask.

---

## Train window extended to 2026

`configs/base.yaml` now covers **2016–2026** Jan–May (11 LOYO folds).  
Revisit: rebuild fire labels with FIRMS for 2026, export GEE for **2026 only** if older years already done, re-run baselines.

```bash
python scripts/build_fire_labels.py          # caches old chunks; adds 2026
python scripts/run_baselines.py              # after new fire_daily.zarr
```

---

## Day 4–5 — GEE + local static · In progress (manual)

Full step-by-step: **[docs/DAY4_5_MANUAL.md](docs/DAY4_5_MANUAL.md)**

| Piece | Location |
|---|---|
| GEE scripts | `gee/era5_daily.js`, `lst_8day.js`, `ndvi_16day.js`, `static.js` |
| Drive root | `Prometheus_GEE/{era5,lst,ndvi,static}` |
| Local OSM distances + physio | `scripts/build_local_static.py` |
| Alignment test | `tests/test_static_alignment.py` |
| Land copies | `data/raw/gee/…` after Drive download |

**You (manual):** start EE Tasks + Geofabrik download; agent cannot run EE or Drive for you.
