"""Daily forecast orchestration: predict → COG → district GeoJSON."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
import numpy as np

from prometheus.config import load_settings
from prometheus.infer import districts as dist
from prometheus.infer import io_cog
from prometheus.models.predict import RiskPredictor, _as_date

SEASON_START = (1, 1)
SEASON_END = (5, 31)


def season_dates(year: int) -> list[date]:
    start = date(year, *SEASON_START)
    end = date(year, *SEASON_END)
    days = []
    d = start
    while d <= end:
        days.append(d)
        d += timedelta(days=1)
    return days


def is_in_season(day: date) -> bool:
    return day.month in load_settings().season.months


@dataclass
class ForecastResult:
    day: date
    paths: dict[str, str] = field(default_factory=dict)
    skipped: bool = False
    seconds: float = 0.0


class ForecastPipeline:
    """
    Produce operational artefacts for one date from the frozen LightGBM bundle.

    "Both models" here are the calibrated h1 and h7 heads — LightGBM won the
    CNN comparison, so the convolutional net is not in the production path.
    Predictors come from the local feature cube (the same GEE ERA5 / MODIS
    stack the model was trained on). Live re-exports for "today" belong in a
    later ops step; day-13 backfill and `make forecast DATE=` use the cube.
    """

    def __init__(
        self,
        *,
        bundle: str | Path = "latest",
        out_dir: Path | None = None,
        horizons: list[int] | None = None,
    ):
        self.predictor = RiskPredictor(bundle)
        self.out_dir = out_dir or io_cog.forecasts_dir()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.horizons = horizons or list(self.predictor.horizons)

    def forecast(
        self,
        when: date | datetime | str,
        *,
        force: bool = False,
    ) -> ForecastResult:
        import time

        day = _as_date(when)
        if not is_in_season(day):
            raise ValueError(
                f"{day} is outside the modelled Jan–May season"
            )

        if not force and io_cog.is_complete(day, root=self.out_dir, horizons=self.horizons):
            return ForecastResult(
                day=day,
                paths={
                    **{f"h{h}": str(io_cog.risk_path(day, h, self.out_dir)) for h in self.horizons},
                    "districts": str(io_cog.districts_path(day, self.out_dir)),
                },
                skipped=True,
            )

        started = time.perf_counter()
        self.predictor.warm(day.year)

        risk_maps = {}
        paths: dict[str, str] = {}
        for h in self.horizons:
            risk = self.predictor.predict(day, horizon=h)
            tags = {"PROMETHEUS_BUNDLE_HASH": getattr(self.predictor.bundle, "bundle_hash", "")}
            path = io_cog.write_risk_cog(risk, io_cog.risk_path(day, h, self.out_dir), tags=tags)
            risk_maps[h] = risk
            paths[f"h{h}"] = str(path)

        thresholds = self.predictor.bundle.horizons[self.horizons[0]].risk_thresholds
        gdf = dist.zonal_risk(
            risk_maps,
            bundle_class_names=self.predictor.class_names(),
            thresholds_h1=thresholds,
        )
        gdf["date"] = day.isoformat()
        dpath = dist.write_districts(gdf, io_cog.districts_path(day, self.out_dir))
        paths["districts"] = str(dpath)

        # Ingest into SQLite database
        from prometheus.db import get_connection, init_db
        init_db(self.out_dir)
        conn = get_connection(self.out_dir)
        try:
            # 1. Forecast Metadata
            bundle_hash = getattr(self.predictor.bundle, "bundle_hash", "")
            features_hash = "" # Future: implement feature hashing
            conn.execute(
                "INSERT OR IGNORE INTO forecasts (forecast_date, bundle_hash, features_hash) VALUES (?, ?, ?)",
                (day.isoformat(), bundle_hash, features_hash)
            )

            # 2. District Stats
            for _, row in gdf.iterrows():
                did = str(row["district_id"])
                for h in self.horizons:
                    mean_h = row.get(f"mean_h{h}")
                    max_h = row.get(f"max_h{h}")
                    if mean_h is not None and not np.isnan(mean_h):
                        conn.execute(
                            """
                            INSERT OR REPLACE INTO district_stats (district_id, forecast_date, horizon, mean_prob, max_prob)
                            VALUES (?, ?, ?, ?, ?)
                            """,
                            (did, day.isoformat(), h, float(mean_h), float(max_h) if max_h is not None and not np.isnan(max_h) else None)
                        )
            conn.commit()
        except Exception as e:
            print(f"Error writing to db: {e}")
        finally:
            conn.close()

        return ForecastResult(
            day=day,
            paths=paths,
            skipped=False,
            seconds=time.perf_counter() - started,
        )

    def backfill(
        self,
        years: list[int],
        *,
        force: bool = False,
        verbose: bool = True,
    ) -> list[ForecastResult]:
        results = []
        for year in years:
            candidates = season_dates(year)
            # Features only exist for days present in the cube (leap quirks etc.).
            try:
                available = set(self.predictor._year_features(year)["dates"])
            except Exception as exc:
                if verbose:
                    print(f"  skip {year}: {exc}")
                continue
            days = [d for d in candidates if d in available]
            if verbose:
                print(f"  {year}: {len(days)} season days", flush=True)
            for i, day in enumerate(days, start=1):
                result = self.forecast(day, force=force)
                results.append(result)
                if verbose and (i == 1 or i % 25 == 0 or i == len(days) or not result.skipped):
                    tag = "skip" if result.skipped else f"{result.seconds:.1f}s"
                    print(f"    [{i}/{len(days)}] {day} {tag}", flush=True)
        return results


__all__ = ["ForecastPipeline", "ForecastResult", "is_in_season", "season_dates"]
