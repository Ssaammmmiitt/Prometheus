# Prometheus

Daily wildfire risk forecasting for Nepal (pre-monsoon: Jan–May).

## Repository layout

```
prometheus/
├── configs/base.yaml       # single source of truth (years, grid, features, CV)
├── src/prometheus/         # Python package
│   ├── config.py
│   ├── grid.py             # 1 km Nepal grid + mask
│   ├── data/               # FIRMS, GEE, cubes
│   ├── features/
│   ├── models/
│   └── eval/
├── scripts/                # thin CLIs only
├── tests/
├── frontend/               # React map app (no node_modules in git)
├── docs/                   # academic report draft
├── data/                   # runtime data (gitignored)
│   ├── static/             # mask, elevation, slope
│   ├── raw/firms/          # API key, chunks, archives, cleaned points
│   ├── raw/gee/            # future GEE downloads
│   └── cube/               # fire_daily.zarr, later feature cubes
├── BUILD_PLAN.md           # 3-week plan
├── PROGRESS_REPORT.md      # day-by-day progress log
└── pyproject.toml
```

## Setup

```bash
python3.12 -m venv .prometheus-venv
source .prometheus-venv/bin/activate
pip install -U pip hatchling
pip install -e ".[dev]"

python -c "from prometheus.config import cfg; print(cfg.years, cfg.season_months)"
pytest -q
```

## Day-2 fire labels

Uses the **FIRMS Area API** only (`/api/area/csv/...`).  
Country and countries endpoints are currently unavailable on NASA’s status board.

```bash
# MAP_KEY lives in data/raw/firms/.map_key (gitignored) or env FIRMS_MAP_KEY
python scripts/build_fire_labels.py
```

## Rules

- No hardcoded years/paths/channels outside `configs/base.yaml`.
- All rasters align to `grid.py` (465×912, EPSG:4326, Nepal mask).
- Do not commit `data/`, API keys, or model binaries.
