from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

router_app = APIRouter()


def router_factory(root_fn):
    @router_app.get("/verification")
    def verification(
        start: str | None = Query(None, description="YYYY-MM-DD"),
        end: str | None = Query(None, description="YYYY-MM-DD"),
    ) -> dict[str, Any]:
        root = Path(root_fn())
        df = None
        
        try:
            from prometheus.db import get_connection
            conn = get_connection(root)
            try:
                query = "SELECT * FROM verification_metrics"
                params = []
                conditions = []
                if start:
                    conditions.append("forecast_date >= ?")
                    params.append(start)
                if end:
                    conditions.append("forecast_date <= ?")
                    params.append(end)
                if conditions:
                    query += " WHERE " + " AND ".join(conditions)
                
                df = pd.read_sql_query(query, conn, params=params)
                # Ensure boolean type for valid column since SQLite doesn't have a native boolean
                if not df.empty and "valid" in df.columns:
                    df["valid"] = df["valid"].astype(bool)
            finally:
                conn.close()
        except Exception:
            pass

        if df is None or df.empty:
            # Fallback to CSV
            vpath = Path(root) / "verification.csv"
            if not vpath.is_file():
                raise HTTPException(status_code=404, detail="verification data missing")
            df = pd.read_csv(vpath)
        if start:
            df = df[df["forecast_date"] >= start]
        if end:
            df = df[df["forecast_date"] <= end]

        valid = df[df["valid"] == True]  # noqa: E712
        summary = {}
        if len(valid):
            summary = {
                "days": int(len(df)),
                "days_with_fires": int(len(valid)),
                "mean_pr_auc": float(valid["pr_auc"].mean()),
                "mean_top10_capture": float(valid["top10_capture"].mean()),
                "mean_brier": float(valid["brier"].mean()),
                "mean_fss": float(valid["fss"].mean()) if "fss" in valid else float("nan"),
                "mean_rev": float(valid["rev"].mean()) if "rev" in valid else float("nan"),
            }
        return {
            "range": {"start": start, "end": end},
            "summary": summary,
            "rows": df.to_dict(orient="records"),
        }

    return router_app


def router(root: Path | None = None) -> Any:
    if root is None:
        from prometheus.api.app import forecasts_root as root_fn  # type: ignore

        return router_factory(root_fn)
    return router_factory(lambda: root)


__all__ = ["router"]

