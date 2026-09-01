"""Catalogue endpoints the UI needs before it paints a map."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter

router_app = APIRouter()

# Peak-season landing dates, newest year first. The map opens on the first
# of these that actually exists on disk.
PREFERRED_DEFAULTS = ("2026-04-12", "2025-04-12", "2024-04-12")
DEFAULT_DATE = PREFERRED_DEFAULTS[0]


def _default(dates: list[str]) -> str | None:
    for preferred in PREFERRED_DEFAULTS:
        if preferred in dates:
            return preferred
    years = sorted({d[:4] for d in dates})
    if not years:
        return dates[-1] if dates else None
    year_dates = [d for d in dates if d.startswith(years[-1])]
    return year_dates[len(year_dates) // 2] if year_dates else dates[-1]


def available_dates(root: Path, horizon: int = 1) -> list[str]:
    # Try querying SQLite first
    try:
        from prometheus.db import get_connection
        conn = get_connection(root)
        try:
            cursor = conn.execute("SELECT forecast_date FROM forecasts ORDER BY forecast_date")
            dates = [row["forecast_date"] for row in cursor]
            if dates:
                return dates
        finally:
            conn.close()
    except Exception:
        pass

    # Fallback to scanning flat files
    dates = []
    for path in root.glob(f"risk_*_h{horizon}.tif"):
        # risk_YYYY-MM-DD_h1.tif
        stem = path.stem  # risk_2025-04-12_h1
        parts = stem.split("_")
        if len(parts) >= 3:
            dates.append(parts[1])
    return sorted(set(dates))


def router_factory(root_fn):
    @router_app.get("/health")
    def health() -> dict[str, Any]:
        root = Path(root_fn())
        dates = available_dates(root)
        return {
            "ok": True,
            "service": "prometheus",
            "n_forecasts": len(dates),
            "default_date": _default(dates),
        }

    @router_app.get("/forecasts")
    def forecasts() -> dict[str, Any]:
        root = Path(root_fn())
        dates = available_dates(root)
        years = sorted({d[:4] for d in dates})
        return {
            "dates": dates,
            "years": years,
            "default_date": _default(dates),
            "horizons": [1, 7],
        }

    return router_app


def router(root: Path | None = None) -> Any:
    if root is None:
        from prometheus.api.app import forecasts_root as root_fn  # type: ignore

        return router_factory(root_fn)
    return router_factory(lambda: root)


__all__ = ["DEFAULT_DATE", "PREFERRED_DEFAULTS", "available_dates", "router"]
