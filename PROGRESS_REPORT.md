# Prometheus — Progress Report

Living log of what was built, cleaned, and verified. Keep this short and current.

| Field | Value |
|---|---|
| Project | Prometheus — daily wildfire risk for Nepal |
| Branch | `v2` |
| Last updated | 2026-08-09 |

---

## Repository status (after cleanup)

**Clean layout only** — all v1 bulk data, legacy code, old reports, and unused FIRMS exports deleted.

```
configs/  src/  scripts/  tests/  frontend/  docs/
data/          # gitignored runtime data
BUILD_PLAN.md  PROGRESS_REPORT.md  README.md  pyproject.toml
```

### Removed (not needed for v2)
- `legacy/` (old source, GEE scripts, notebooks, model checkpoints, nested venv)
- `data_raw/`, `data_processed/`, `data_processed_normalized/` (16-day v1 rasters ~500 MB)
- `reports/`, `results/`, unused docs (`AUDIT_AND_ROADMAP.md`, commit-message scratch file)
- `frontend/node_modules/`
- Redundant cleaned FIRMS CSV variants

### Kept on disk under `data/` (gitignored)
| Path | Why |
|---|---|
| `data/static/nepal_mask_*.tif` | Canonical Nepal grid mask (168,064 valid 1 km cells) |
| `data/static/elevation_*.tif`, `slope_*.tif` | Terrain static layers |
| `data/raw/firms/archives/fire_archive_M-C61_*.csv` | MODIS seed for offline rebuilds |
| `data/raw/firms/.map_key` | FIRMS MAP_KEY (secret) |
| `data/cube/fire_daily.zarr` | Day-2 label cube |
| `data/raw/firms/firms_clean_points.csv` | Cleaned detections |

---

## Day 1 — Scaffold

**Done.** Package loads config; tests pass.

```text
python -c "from prometheus.config import cfg; print(cfg.years, cfg.season_months)"
→ [2016…2025] [1, 2, 3, 4, 5]
```

---

## Day 2 — Fire labels

**Pipeline done.** Uses FIRMS **Area API only**  
(`/api/area/csv/{MAP_KEY}/{SOURCE}/{west,south,east,north}/{day_range}/{date}`).

Per NASA status: **country / countries endpoints are not available**. We do not use them.

| Check | Result |
|---|---|
| Cube shape | (1513, 465, 912) Jan–May daily |
| Outside-mask fire pixels | **0** |
| April spike in year×month table | **Yes** |
| Season detections (MODIS seed only) | ~36.9k (VIIRS download still needed for >120k) |
| Tests | **9 passed** |

### Finish VIIRS download on your Mac (agent cannot reach NASA)

```bash
source .prometheus-venv/bin/activate
python scripts/build_fire_labels.py
```

Chunks cache under `data/raw/firms/chunks/`. Safe to re-run.

---

## Alignment rule (every future download)

1. CRS = EPSG:4326  
2. Shape = 465 × 912  
3. Affine = values in `configs/base.yaml`  
4. No positive labels outside Nepal mask  

Helpers: `grid.assert_aligned(path)`, `firms.assert_cube_alignment(cube)`.

---

## Commit suggestion (Days 1–2 + cleanup)

```
Scaffold clean v2 package and daily fire labels; remove v1 bulk.

Single config/grid layout, FIRMS area-API pipeline to fire_daily.zarr,
Nepal-mask alignment checks. Drop legacy code and unused 16-day rasters.
```

Stage code/docs only — **not** `data/` or `.map_key`.

---

## Next: Day 3

Evaluation harness + climatology / persistence baselines (local only, no large downloads).
