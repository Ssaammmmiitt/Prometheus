"""LightGBM fire-risk model: fold training, full-grid scoring, light search."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from prometheus.config import load_settings
from prometheus.features import table as ftable

PRIMARY_LABEL = "label_h1"


def default_params() -> dict[str, Any]:
    m = load_settings().modeling
    return {
        "objective": "binary",
        "metric": ["average_precision", "auc"],
        "learning_rate": m.learning_rate,
        "num_leaves": m.num_leaves,
        "min_data_in_leaf": m.min_data_in_leaf,
        "feature_fraction": m.feature_fraction,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "seed": m.random_seed,
        "num_threads": 0,
        "verbosity": -1,
    }


def models_dir() -> Path:
    path = load_settings().paths.resolve("models")
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass
class FoldResult:
    year: int
    params: dict[str, Any]
    best_iteration: int
    train_seconds: float
    n_train: int
    n_valid: int
    scale_pos_weight: float
    valid_pr_auc: float
    grid_metrics: dict[str, float] = field(default_factory=dict)
    model_path: str | None = None

    def to_dict(self) -> dict:
        return {
            "year": self.year,
            "params": self.params,
            "best_iteration": self.best_iteration,
            "train_seconds": round(self.train_seconds, 2),
            "n_train": self.n_train,
            "n_valid": self.n_valid,
            "scale_pos_weight": round(self.scale_pos_weight, 3),
            "valid_pr_auc": self.valid_pr_auc,
            "grid_metrics": self.grid_metrics,
            "model_path": self.model_path,
        }


def split_fold(
    table: pd.DataFrame, held_out_year: int, label: str = PRIMARY_LABEL
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Training rows and an inner validation season for early stopping.

    The stopping signal must never come from the held-out year, so the most
    recent remaining season is set aside instead. Using a random row split would
    leak: neighbouring cells on the same day are almost the same sample.
    """
    train_years = [y for y in sorted(table["year"].unique()) if y != held_out_year]
    if len(train_years) < 2:
        raise ValueError("need at least two training years")
    inner = train_years[-1] if train_years[-1] != held_out_year else train_years[-2]
    fit = table[(table["year"] != held_out_year) & (table["year"] != inner)]
    valid = table[table["year"] == inner]
    return fit, valid


def train_fold(
    table: pd.DataFrame,
    held_out_year: int,
    *,
    params: dict[str, Any] | None = None,
    label: str = PRIMARY_LABEL,
    num_boost_round: int | None = None,
    verbose: bool = False,
) -> tuple[Any, FoldResult]:
    """Fit one leave-one-year-out fold with early stopping."""
    import lightgbm as lgb

    settings = load_settings()
    params = {**default_params(), **(params or {})}
    num_boost_round = num_boost_round or settings.modeling.n_estimators

    features = ftable.present_features(table)
    fit, valid = split_fold(table, held_out_year, label)

    n_pos = int(fit[label].sum())
    n_neg = len(fit) - n_pos
    # The table is already sampled at 1:20, so this only corrects the residual
    # imbalance rather than re-weighting the raw 1:100 base rate twice.
    scale_pos_weight = n_neg / max(n_pos, 1)
    params = {**params, "scale_pos_weight": scale_pos_weight}

    # Names must travel with the model: full-grid scoring rebuilds the matrix
    # from scratch and relies on them to get the column order right.
    dtrain = lgb.Dataset(
        fit[features].to_numpy(np.float32),
        label=fit[label].to_numpy(),
        feature_name=features,
    )
    dvalid = lgb.Dataset(
        valid[features].to_numpy(np.float32),
        label=valid[label].to_numpy(),
        reference=dtrain,
        feature_name=features,
    )

    started = datetime.now()
    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        valid_sets=[dvalid],
        valid_names=["inner_valid"],
        callbacks=[
            lgb.early_stopping(settings.modeling.early_stopping_rounds, verbose=verbose),
            *([lgb.log_evaluation(50)] if verbose else []),
        ],
    )
    elapsed = (datetime.now() - started).total_seconds()

    from prometheus.eval.metrics import pr_auc

    valid_scores = booster.predict(
        valid[features].to_numpy(np.float32), num_iteration=booster.best_iteration
    )
    result = FoldResult(
        year=held_out_year,
        params=params,
        best_iteration=int(booster.best_iteration or num_boost_round),
        train_seconds=elapsed,
        n_train=len(fit),
        n_valid=len(valid),
        scale_pos_weight=scale_pos_weight,
        valid_pr_auc=pr_auc(valid[label].to_numpy(), valid_scores),
    )
    return booster, result


def predict_grid(
    booster,
    year: int,
    *,
    cube=None,
    fire_ds=None,
    anomaly=None,
    all_years: list[int] | None = None,
    features: list[str] | None = None,
    chunk_rows: int = 2_000_000,
) -> dict:
    """Score every forest cell for every day of a season."""
    bundle = ftable.year_grid_features(
        year,
        cube=cube,
        fire_ds=fire_ds,
        anomaly=anomaly,
        all_years=all_years,
        features=features or list(booster.feature_name()),
    )
    matrix = bundle["matrix"]
    scores = np.empty(matrix.shape[0], dtype=np.float32)
    for start in range(0, matrix.shape[0], chunk_rows):
        stop = min(start + chunk_rows, matrix.shape[0])
        scores[start:stop] = booster.predict(
            matrix[start:stop], num_iteration=booster.best_iteration
        )
    bundle["scores"] = scores.reshape(bundle["n_days"], bundle["n_cells"])
    del bundle["matrix"]
    return bundle


def evaluate_grid(
    booster,
    year: int,
    *,
    horizon: int = 1,
    cube=None,
    fire_ds=None,
    anomaly=None,
    all_years: list[int] | None = None,
) -> dict:
    """
    Score the model and both baselines on one identical pixel population.

    Comparability is the whole point: the baselines in `runs/baselines` cover
    all Nepal cells, while the model only speaks for the forest mask. Recomputing
    them here on the model's own cells removes that mismatch.
    """
    from prometheus.eval import baselines, metrics
    from prometheus.features import derived

    bundle = predict_grid(
        booster, year, cube=cube, fire_ds=fire_ds, anomaly=anomaly, all_years=all_years
    )
    rows, cols = bundle["rows"], bundle["cols"]
    labels_all = derived.horizon_labels(bundle["fire"], [horizon])
    y = labels_all[f"label_h{horizon}"][:, rows, cols]
    usable = labels_all[f"valid_h{horizon}"]

    y_true = y[usable].ravel()
    y_pred = bundle["scores"][usable].ravel()

    clim_rates = baselines.load_or_build_climatology()
    doy = np.array([min(d.timetuple().tm_yday, clim_rates.shape[0] - 1) for d in bundle["dates"]])
    y_clim = clim_rates[doy][:, rows, cols][usable].ravel()

    times = pd.DatetimeIndex(bundle["dates"])
    persistence = baselines.persistence_scores(bundle["fire"], times)[:, rows, cols][
        usable
    ].ravel()

    out = {
        "year": year,
        "horizon": horizon,
        "n": int(y_true.size),
        "n_pos": int(y_true.sum()),
        "base_rate": float(y_true.mean()),
        "model_pr_auc": metrics.pr_auc(y_true, y_pred),
        "model_roc_auc": metrics.roc_auc(y_true, y_pred),
        "model_brier": metrics.brier(y_true, y_pred),
        "model_top10_capture": metrics.top_k_capture(y_true, y_pred, 0.10),
        "clim_pr_auc": metrics.pr_auc(y_true, y_clim),
        "clim_top10_capture": metrics.top_k_capture(y_true, y_clim, 0.10),
        "persistence_pr_auc": metrics.pr_auc(y_true, persistence),
        "persistence_top10_capture": metrics.top_k_capture(y_true, persistence, 0.10),
    }
    out["skill_vs_clim"] = (
        (out["model_pr_auc"] - out["clim_pr_auc"]) / max(out["clim_pr_auc"], 1e-9)
    )
    out["beats_climatology"] = bool(out["model_pr_auc"] > out["clim_pr_auc"])
    return out


def sample_param_grid(n: int, seed: int = 42) -> list[dict[str, Any]]:
    """Random draws over the four knobs that actually move this model."""
    rng = np.random.default_rng(seed)
    space = {
        "num_leaves": [31, 63, 127, 255],
        "min_data_in_leaf": [20, 50, 100, 200, 400],
        "learning_rate": [0.02, 0.05, 0.08, 0.12],
        "feature_fraction": [0.5, 0.65, 0.8, 1.0],
    }
    seen: set[tuple] = set()
    configs: list[dict[str, Any]] = []
    while len(configs) < n:
        pick = {k: v[int(rng.integers(len(v)))] for k, v in space.items()}
        key = tuple(sorted(pick.items()))
        if key in seen:
            continue
        seen.add(key)
        configs.append(pick)
    return configs


def search(
    table: pd.DataFrame,
    held_out_year: int,
    *,
    n_configs: int = 24,
    label: str = PRIMARY_LABEL,
    verbose: bool = True,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Random search scored on the inner validation season."""
    rows = []
    best: tuple[float, dict[str, Any]] | None = None
    for i, config in enumerate(sample_param_grid(n_configs), start=1):
        _, result = train_fold(table, held_out_year, params=config, label=label)
        rows.append({**config, "valid_pr_auc": result.valid_pr_auc,
                     "best_iteration": result.best_iteration,
                     "train_seconds": round(result.train_seconds, 1)})
        if best is None or result.valid_pr_auc > best[0]:
            best = (result.valid_pr_auc, config)
        if verbose:
            print(
                f"  [{i:>2}/{n_configs}] PR-AUC {result.valid_pr_auc:.4f} "
                f"({result.train_seconds:5.1f}s, {result.best_iteration:>3} trees) {config}",
                flush=True,
            )
    frame = pd.DataFrame(rows).sort_values("valid_pr_auc", ascending=False)
    return (best[1] if best else {}), frame


def save_model(booster, result: FoldResult, tag: str = "lgbm") -> Path:
    path = models_dir() / f"{tag}_holdout{result.year}.txt"
    booster.save_model(str(path), num_iteration=booster.best_iteration)
    result.model_path = str(path)
    (models_dir() / f"{tag}_holdout{result.year}.json").write_text(
        json.dumps(result.to_dict(), indent=2), encoding="utf-8"
    )
    return path


def load_model(path: Path):
    import lightgbm as lgb

    return lgb.Booster(model_file=str(path))


__all__ = [
    "PRIMARY_LABEL",
    "FoldResult",
    "default_params",
    "evaluate_grid",
    "load_model",
    "models_dir",
    "predict_grid",
    "sample_param_grid",
    "save_model",
    "search",
    "split_fold",
    "train_fold",
]
