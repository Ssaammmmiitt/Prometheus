"""
Leave-one-year-out cross-validation, family ablations, and per-region scoring.

Everything here is scored on the full forest grid, never on the sampled training
table, and every baseline is recomputed on the same pixels as the model.
"""

from __future__ import annotations

import gc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from prometheus.config import load_settings
from prometheus.eval import baselines, metrics
from prometheus.features import derived, forest
from prometheus.features import table as ftable
from prometheus.models import lgbm

#: 2016 is the first season with fire labels, so nothing before it exists to
#: build fire history from. It trains and seeds history but is never held out —
#: scoring it would measure a model whose history features are blank by
#: construction, not a model that failed.
HISTORY_WARMUP_YEAR = 2016

REGION_NAMES = {1: "Terai", 2: "Chure", 3: "MiddleMountains", 4: "HighMountains"}


def feature_families() -> dict[str, list[str]]:
    """
    Feature families for ablation.

    Rolling and dryness aggregates are counted as weather even though the config
    files them under `temporal`: dropping weather while keeping `precip_30d`
    would not be an honest ablation. Day-of-year encoding stays, since it is
    calendar position rather than an observation.
    """
    f = load_settings().features
    weather_derived = [
        "consecutive_dry_days",
        "days_since_rain",
        "precip_7d",
        "precip_30d",
        "t2m_max_7d",
        "rh_min_7d",
        "wind_max_7d",
    ]
    return {
        "weather": list(f.weather_daily) + weather_derived,
        "vegetation": list(f.vegetation) + list(f.thermal),
        "terrain": list(f.terrain_static) + list(f.landcover_static),
        "human": list(f.human_static),
        "fire_history": list(f.history),
    }


def ablation_features(all_features: list[str], drop: str | None) -> list[str]:
    if drop is None:
        return list(all_features)
    families = feature_families()
    if drop not in families:
        raise KeyError(f"unknown family {drop!r}; have {sorted(families)}")
    removed = set(families[drop])
    kept = [f for f in all_features if f not in removed]
    if len(kept) == len(all_features):
        raise ValueError(f"family {drop!r} removed nothing")
    return kept


@dataclass
class Variant:
    name: str
    features: list[str]
    booster: Any = None
    train_seconds: float = 0.0
    best_iteration: int = 0


def region_codes(rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    codes = forest.human_static()["physio_region"][rows, cols]
    return np.nan_to_num(codes, nan=0.0).astype(np.int8)


def _score_block(y_true, y_pred, y_clim, y_persist, *, full: bool = True) -> dict[str, float]:
    out = {
        "n": int(y_true.size),
        "n_pos": int(y_true.sum()),
        "base_rate": float(y_true.mean()) if y_true.size else float("nan"),
        "pr_auc": metrics.pr_auc(y_true, y_pred),
        "top10_capture": metrics.top_k_capture(y_true, y_pred, 0.10),
    }
    if full:
        # Ablations are compared on PR-AUC alone, so the extra sorts over 19M
        # rows are only worth paying for on the model we actually report.
        out["roc_auc"] = metrics.roc_auc(y_true, y_pred)
        out["brier"] = metrics.brier(y_true, y_pred)
        out["top5_capture"] = metrics.top_k_capture(y_true, y_pred, 0.05)
    if y_clim is not None:
        out["clim_pr_auc"] = metrics.pr_auc(y_true, y_clim)
        out["clim_top10_capture"] = metrics.top_k_capture(y_true, y_clim, 0.10)
        out["skill_vs_clim"] = (out["pr_auc"] - out["clim_pr_auc"]) / max(
            out["clim_pr_auc"], 1e-9
        )
    if y_persist is not None:
        out["persistence_pr_auc"] = metrics.pr_auc(y_true, y_persist)
    return out


def shap_summary(
    booster,
    table: pd.DataFrame,
    year: int,
    features: list[str],
    *,
    sample: int = 20_000,
    seed: int = 0,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Mean |SHAP| per feature on a sample of the held-out season.

    Uses LightGBM's own path-dependent TreeSHAP. With correlated features that
    attributes along the tree's actual splits rather than assuming independence,
    which is the right choice here — but it still divides credit between the
    collinear twins, so read those at the pair level.
    """
    rows = table[table["year"] == year]
    if len(rows) > sample:
        rows = rows.sample(sample, random_state=seed)
    x = rows[features].to_numpy(np.float32)
    contrib = booster.predict(x, pred_contrib=True, **lgbm._predict_kwargs(booster))
    values = contrib[:, :-1]  # last column is the expected-value baseline
    mean_abs = pd.Series(np.abs(values).mean(axis=0), index=features).sort_values(
        ascending=False
    )
    frame = pd.DataFrame(values, columns=features, index=rows.index)
    return mean_abs, frame


def run_fold(
    table: pd.DataFrame,
    year: int,
    *,
    all_features: list[str],
    drops: list[str | None],
    cube,
    fire_ds,
    anomaly,
    fit_years: list[int],
    horizon: int = 1,
    label: str = lgbm.PRIMARY_LABEL,
    shap_sample: int = 20_000,
    verbose: bool = True,
) -> dict:
    """Train every variant for one held-out season and score them all on its grid."""
    fold_table = table[table["year"].isin(fit_years + [year])]

    variants: list[Variant] = []
    for drop in drops:
        name = "full" if drop is None else f"drop_{drop}"
        feats = ablation_features(all_features, drop)
        sub = fold_table[feats + [label, "year"]]
        booster, result = lgbm.train_fold(sub, year, label=label)
        variants.append(
            Variant(name, feats, booster, result.train_seconds, result.best_iteration)
        )
        if verbose:
            print(
                f"    {name:<20} {len(feats):>3} feats · {result.train_seconds:5.1f}s "
                f"· {result.best_iteration:>3} trees · inner PR-AUC {result.valid_pr_auc:.4f}",
                flush=True,
            )

    if verbose:
        print("    building full-grid feature matrix …", flush=True)
    bundle = ftable.year_grid_features(
        year, cube=cube, fire_ds=fire_ds, anomaly=anomaly, features=all_features
    )
    rows, cols = bundle["rows"], bundle["cols"]

    labels_all = derived.horizon_labels(bundle["fire"], [horizon])
    usable = labels_all[f"valid_h{horizon}"]
    y = labels_all[f"label_h{horizon}"][:, rows, cols][usable]

    clim_rates = baselines.load_or_build_climatology()
    doy = np.array(
        [min(d.timetuple().tm_yday, clim_rates.shape[0] - 1) for d in bundle["dates"]]
    )
    y_clim = clim_rates[doy][:, rows, cols][usable]
    y_persist = baselines.persistence_scores(
        bundle["fire"], pd.DatetimeIndex(bundle["dates"])
    )[:, rows, cols][usable]

    codes = np.broadcast_to(region_codes(rows, cols), (bundle["n_days"], len(rows)))[usable]

    out: dict[str, Any] = {"year": year, "horizon": horizon, "variants": {}, "regions": {}}
    for v in variants:
        is_full = v.name == "full"
        scores = lgbm.score_matrix(v.booster, bundle)[usable]
        block = _score_block(
            y,
            scores,
            y_clim if is_full else None,
            y_persist if is_full else None,
            full=is_full,
        )
        block.update(n_features=len(v.features), train_seconds=round(v.train_seconds, 1),
                     best_iteration=v.best_iteration)
        out["variants"][v.name] = block
        if verbose:
            print(f"    {v.name:<20} grid PR-AUC {block['pr_auc']:.4f}", flush=True)

        if is_full:
            for code, region in REGION_NAMES.items():
                sel = codes == code
                if not sel.any():
                    continue
                out["regions"][region] = _score_block(
                    y[sel], scores[sel], y_clim[sel], y_persist[sel]
                )
            lgbm.save_model(
                v.booster,
                lgbm.FoldResult(
                    year=year, params=lgbm.default_params(),
                    best_iteration=v.best_iteration, train_seconds=v.train_seconds,
                    n_train=len(fold_table), n_valid=0,
                    scale_pos_weight=float("nan"), valid_pr_auc=float("nan"),
                    grid_metrics=block,
                ),
            )
            mean_abs, _ = shap_summary(
                v.booster, fold_table, year, v.features, sample=shap_sample
            )
            out["shap_mean_abs"] = mean_abs.to_dict()
            out["region_shap"] = _region_shap(
                v.booster, fold_table, year, v.features, shap_sample
            )
        del scores

    del bundle
    gc.collect()
    return out


def _region_shap(booster, table, year, features, sample) -> dict[str, dict[str, float]]:
    """Mean |SHAP| per feature within each physiographic region."""
    rows = table[table["year"] == year]
    if not {"row", "col"}.issubset(rows.columns):
        return {}
    codes = region_codes(rows["row"].to_numpy(), rows["col"].to_numpy())
    out: dict[str, dict[str, float]] = {}
    for code, region in REGION_NAMES.items():
        sub = rows[codes == code]
        if len(sub) < 500:
            continue
        if len(sub) > sample:
            sub = sub.sample(sample, random_state=0)
        contrib = booster.predict(
            sub[features].to_numpy(np.float32),
            pred_contrib=True,
            **lgbm._predict_kwargs(booster),
        )
        mean_abs = np.abs(contrib[:, :-1]).mean(axis=0)
        share = 100 * mean_abs / max(mean_abs.sum(), 1e-12)
        out[region] = dict(zip(features, share.round(3)))
    return out


def run_cv(
    table: pd.DataFrame | None = None,
    *,
    years: list[int] | None = None,
    drops: list[str | None] | None = None,
    horizon: int = 1,
    shap_sample: int = 20_000,
    out_dir: Path | None = None,
    verbose: bool = True,
) -> dict:
    """Full leave-one-year-out sweep with ablations and per-region breakdown."""
    settings = load_settings()
    table = table if table is not None else ftable.load_train_table()
    all_features = ftable.present_features(table)
    all_years = sorted(settings.years.all)
    years = years or [y for y in all_years if y != HISTORY_WARMUP_YEAR]
    drops = drops if drops is not None else [None, *feature_families()]
    fit_years = all_years

    cube = ftable.open_cube()
    fire_ds = ftable._fire_cube()
    anomaly = ftable.build_anomaly(cube, all_years)

    out_dir = out_dir or (Path(settings.root) / "runs" / "cv")
    out_dir.mkdir(parents=True, exist_ok=True)

    folds = []
    for i, year in enumerate(years, start=1):
        if verbose:
            print(f"\n[{i}/{len(years)}] holdout {year}", flush=True)
        fold = run_fold(
            table, year, all_features=all_features, drops=drops, cube=cube,
            fire_ds=fire_ds, anomaly=anomaly, fit_years=fit_years, horizon=horizon,
            shap_sample=shap_sample, verbose=verbose,
        )
        folds.append(fold)
        (out_dir / f"fold_{year}.json").write_text(
            json.dumps(fold, indent=2), encoding="utf-8"
        )

    summary = summarise(folds)
    for name, frame in summary.items():
        frame.to_csv(out_dir / f"{name}.csv", index=False)
    (out_dir / "folds.json").write_text(json.dumps(folds, indent=2), encoding="utf-8")
    return {"folds": folds, "summary": summary}


def summarise(folds: list[dict]) -> dict[str, pd.DataFrame]:
    """Per-fold, ablation, region, and SHAP tables with mean ± std across folds."""
    per_fold = pd.DataFrame(
        [{"year": f["year"], **f["variants"]["full"]} for f in folds]
    )

    rows = []
    for f in folds:
        base = f["variants"]["full"]["pr_auc"]
        for name, block in f["variants"].items():
            rows.append(
                {"year": f["year"], "variant": name, "pr_auc": block["pr_auc"],
                 "top10_capture": block["top10_capture"],
                 "n_features": block["n_features"],
                 "delta_pr_auc": block["pr_auc"] - base,
                 "delta_pct": 100 * (block["pr_auc"] - base) / max(base, 1e-9)}
            )
    long = pd.DataFrame(rows)
    ablation = (
        long.groupby("variant")
        .agg(n_features=("n_features", "first"),
             pr_auc_mean=("pr_auc", "mean"), pr_auc_std=("pr_auc", "std"),
             delta_mean=("delta_pr_auc", "mean"), delta_std=("delta_pr_auc", "std"),
             delta_pct_mean=("delta_pct", "mean"))
        .reset_index()
        .sort_values("pr_auc_mean", ascending=False)
    )

    region_rows = [
        {"region": region, "year": f["year"], **block}
        for f in folds
        for region, block in f["regions"].items()
    ]
    regions = pd.DataFrame(region_rows)
    if not regions.empty:
        regions = (
            regions.groupby("region")
            .agg(n_pos=("n_pos", "sum"), base_rate=("base_rate", "mean"),
                 pr_auc_mean=("pr_auc", "mean"), pr_auc_std=("pr_auc", "std"),
                 clim_pr_auc=("clim_pr_auc", "mean"),
                 top10_mean=("top10_capture", "mean"),
                 skill_mean=("skill_vs_clim", "mean"))
            .reset_index()
        )
        order = {n: i for i, n in enumerate(REGION_NAMES.values())}
        regions = regions.sort_values("region", key=lambda s: s.map(order))

    shap_frame = pd.DataFrame([f.get("shap_mean_abs", {}) for f in folds])
    shap_tbl = pd.DataFrame(
        {"feature": shap_frame.columns,
         "mean_abs_shap": shap_frame.mean().to_numpy(),
         "std_across_folds": shap_frame.std().to_numpy()}
    ).sort_values("mean_abs_shap", ascending=False)
    shap_tbl["share_pct"] = (
        100 * shap_tbl["mean_abs_shap"] / max(shap_tbl["mean_abs_shap"].sum(), 1e-12)
    )

    region_shap_rows = []
    for f in folds:
        for region, shares in f.get("region_shap", {}).items():
            for feature, share in shares.items():
                region_shap_rows.append(
                    {"region": region, "feature": feature, "share_pct": share}
                )
    region_shap = pd.DataFrame(region_shap_rows)
    region_family = pd.DataFrame()
    if not region_shap.empty:
        region_shap = (
            region_shap.groupby(["region", "feature"])["share_pct"].mean().reset_index()
        )
        # Individual features are noisy and split credit between collinear twins;
        # rolled up to families the regional comparison is actually readable.
        lookup = {f: fam for fam, feats in feature_families().items() for f in feats}
        region_shap["family"] = region_shap["feature"].map(lookup).fillna("calendar")
        region_family = (
            region_shap.pivot_table(
                index="region", columns="family", values="share_pct", aggfunc="sum"
            )
            .reindex([r for r in REGION_NAMES.values()])
            .dropna(how="all")
            .reset_index()
        )

    return {
        "per_fold": per_fold,
        "ablation": ablation,
        "ablation_per_fold": long,
        "regions": regions,
        "shap_global": shap_tbl,
        "region_shap": region_shap,
        "region_family_shap": region_family,
    }


__all__ = [
    "HISTORY_WARMUP_YEAR",
    "REGION_NAMES",
    "ablation_features",
    "feature_families",
    "region_codes",
    "run_cv",
    "run_fold",
    "shap_summary",
    "summarise",
]
