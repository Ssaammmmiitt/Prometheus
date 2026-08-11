#!/usr/bin/env python3
"""Day 10: SHAP beeswarm plus dependence plots for the top features.

  python scripts/plot_shap.py --year 2021
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from prometheus.config import load_settings  # noqa: E402
from prometheus.eval import cv  # noqa: E402
from prometheus.features.table import load_train_table, present_features  # noqa: E402
from prometheus.models import lgbm  # noqa: E402

#: Twins split their credit, so a dependence plot for one is only half the story.
TWIN = {a: b for a, b, _ in lgbm.COLLINEAR_TWINS}
TWIN.update({b: a for a, b, _ in lgbm.COLLINEAR_TWINS})


def display_values(x: pd.DataFrame) -> pd.DataFrame:
    """
    Hide the never-burned sentinel from the plots.

    `days_since_fire` is 9999 wherever a cell has no recorded fire. Left in, that
    single value owns the whole axis and colour scale, and the informative
    0-to-a-few-hundred-day range collapses into a stripe. The model still uses
    the sentinel; only the picture drops it.
    """
    from prometheus.features.derived import NO_FIRE_SENTINEL

    out = x.copy()
    if "days_since_fire" in out:
        out.loc[out["days_since_fire"] >= NO_FIRE_SENTINEL, "days_since_fire"] = np.nan
    return out


def beeswarm(shap_values: pd.DataFrame, x: pd.DataFrame, order: list[str], path: Path):
    """Beeswarm: contribution spread per feature, coloured by feature value."""
    import shap

    explanation = shap.Explanation(
        values=shap_values[order].to_numpy(),
        data=x[order].to_numpy(),
        feature_names=order,
    )
    plt.figure(figsize=(9, 6))
    shap.plots.beeswarm(explanation, max_display=len(order), show=False)
    plt.title("SHAP contributions to next-day fire probability", fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def dependence_grid(
    shap_values: pd.DataFrame, x: pd.DataFrame, features: list[str], path: Path
):
    """One dependence panel per feature: value on x, its SHAP contribution on y."""
    n = len(features)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(4.2 * ((n + 1) // 2), 7.5))
    for ax, feature in zip(np.ravel(axes), features):
        xv = x[feature].to_numpy()
        yv = shap_values[feature].to_numpy()
        finite = np.isfinite(xv)
        lo, hi = np.nanpercentile(xv[finite], [0.5, 99.5]) if finite.any() else (0, 1)
        keep = finite & (xv >= lo) & (xv <= hi)
        ax.scatter(xv[keep], yv[keep], s=2, alpha=0.15, c="#c0392b", edgecolors="none")
        ax.axhline(0, color="0.6", lw=0.8, ls="--")

        if keep.sum() > 100:
            bins = np.linspace(lo, hi, 25)
            idx = np.clip(np.digitize(xv[keep], bins) - 1, 0, len(bins) - 2)
            med = pd.Series(yv[keep]).groupby(idx).median()
            centres = (bins[:-1] + bins[1:]) / 2
            ax.plot(centres[med.index], med.to_numpy(), color="#1a1a1a", lw=1.8)

        title = feature
        if feature in TWIN:
            title += f"  (credit shared with {TWIN[feature]})"
        hidden = int((~finite).sum())
        if hidden:
            title += f"\n{hidden / len(xv):.0%} never-burned excluded from view"
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(feature, fontsize=8)
        ax.set_ylabel("SHAP", fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in np.ravel(axes)[n:]:
        ax.axis("off")
    fig.suptitle("SHAP dependence — how each driver moves the prediction", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    settings = load_settings()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--year", type=int, default=2021, help="Held-out season to explain")
    ap.add_argument("--sample", type=int, default=20_000)
    ap.add_argument("--top", type=int, default=6)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args(argv)

    model_path = lgbm.models_dir() / f"lgbm_holdout{args.year}.txt"
    if not model_path.is_file():
        print(f"no model for {args.year} — run scripts/run_cv.py first")
        return 1

    out_dir = args.out_dir or (Path(settings.root) / "runs" / "shap")
    out_dir.mkdir(parents=True, exist_ok=True)

    booster = lgbm.load_model(model_path)
    table = load_train_table()
    features = [f for f in present_features(table) if f in set(booster.feature_name())]

    mean_abs, shap_values = cv.shap_summary(
        booster, table, args.year, features, sample=args.sample
    )
    x = display_values(table.loc[shap_values.index, features])

    order = list(mean_abs.head(12).index)
    top = list(mean_abs.head(args.top).index)

    beeswarm(shap_values, x, order, out_dir / f"beeswarm_{args.year}.png")
    dependence_grid(shap_values, x, top, out_dir / f"dependence_{args.year}.png")
    mean_abs.rename("mean_abs_shap").to_csv(out_dir / f"global_{args.year}.csv")

    share = 100 * mean_abs / mean_abs.sum()
    print(f"SHAP on {len(x):,} held-out rows from {args.year}\n")
    for feature in order:
        twin = f"   [credit shared with {TWIN[feature]}]" if feature in TWIN else ""
        print(f"  {feature:<22}{share[feature]:>6.1f}%{twin}")
    print(f"\nplots → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
