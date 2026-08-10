#!/usr/bin/env python3
"""Day 8 diagnostics: feature correlations and fire / no-fire separation.

  python scripts/plot_feature_diagnostics.py
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
from prometheus.features.table import (  # noqa: E402
    load_train_table,
    present_features,
    train_table_path,
)

GROUP_ORDER = (
    "weather_daily",
    "temporal",
    "vegetation",
    "thermal",
    "history",
    "terrain_static",
    "landcover_static",
    "human_static",
)


def ordered_features(table: pd.DataFrame) -> list[str]:
    """Group features by family so the correlation matrix has visible blocks."""
    cfg = load_settings().features
    available = set(present_features(table))
    out: list[str] = []
    for group in GROUP_ORDER:
        for name in getattr(cfg, group, []):
            if name in available and name not in out:
                out.append(name)
    return out


def separation(table: pd.DataFrame, features: list[str], label: str) -> pd.DataFrame:
    """Standardised mean difference between fire and no-fire rows."""
    pos = table[table[label] == 1]
    neg = table[table[label] == 0]
    rows = []
    for name in features:
        sd = table[name].std()
        d = (pos[name].mean() - neg[name].mean()) / sd if sd > 0 else 0.0
        rows.append({"feature": name, "fire": pos[name].mean(),
                     "no_fire": neg[name].mean(), "cohens_d": d})
    return pd.DataFrame(rows).sort_values("cohens_d", key=np.abs, ascending=False)


def plot_correlation(table: pd.DataFrame, features: list[str], out: Path) -> None:
    corr = table[features].corr()
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(features)))
    ax.set_xticklabels(features, rotation=90, fontsize=7)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features, fontsize=7)
    ax.set_title("Feature correlation matrix (grouped by family)", fontsize=13)
    fig.colorbar(im, ax=ax, shrink=0.7, label="Pearson r")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_separation(sep: pd.DataFrame, out: Path, label: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 11))
    colours = ["#c0392b" if v > 0 else "#2980b9" for v in sep["cohens_d"]]
    ax.barh(sep["feature"], sep["cohens_d"], color=colours)
    ax.invert_yaxis()
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Standardised mean difference (fire − no fire)")
    ax.set_title(f"Feature separation for {label}", fontsize=12)
    ax.grid(axis="x", alpha=0.3)
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_distributions(
    table: pd.DataFrame, features: list[str], out: Path, label: str, n: int = 12
) -> None:
    fig, axes = plt.subplots(4, 3, figsize=(14, 12))
    pos = table[table[label] == 1]
    neg = table[table[label] == 0]
    for ax, name in zip(axes.ravel(), features[:n]):
        lo, hi = np.percentile(table[name], [0.5, 99.5])
        if not np.isfinite([lo, hi]).all() or lo == hi:
            lo, hi = table[name].min(), table[name].max() + 1e-6
        bins = np.linspace(lo, hi, 50)
        ax.hist(neg[name], bins=bins, density=True, alpha=0.55,
                color="#2980b9", label="no fire")
        ax.hist(pos[name], bins=bins, density=True, alpha=0.55,
                color="#c0392b", label="fire")
        ax.set_title(name, fontsize=10)
        ax.tick_params(labelsize=7)
    axes.ravel()[0].legend(fontsize=8)
    fig.suptitle(f"Top {n} discriminating features — {label}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", default="label_h1")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    if not train_table_path().exists():
        print(f"missing {train_table_path()} — run scripts/build_train_table.py")
        return 1

    table = load_train_table()
    features = ordered_features(table)
    out_dir = args.out_dir or (Path(load_settings().root) / "runs" / "features")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{len(table):,} rows · {len(features)} features · label {args.label}")

    plot_correlation(table, features, out_dir / "correlation_matrix.png")
    sep = separation(table, features, args.label)
    plot_separation(sep, out_dir / "feature_separation.png", args.label)
    plot_distributions(
        table, list(sep["feature"]), out_dir / "feature_distributions.png", args.label
    )
    sep.to_csv(out_dir / "feature_separation.csv", index=False)

    corr = table[features].corr().abs()
    pairs = (
        corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        .stack()
        .sort_values(ascending=False)
    )

    print("\nTop 15 by separation:")
    print(sep.head(15).to_string(index=False, float_format=lambda v: f"{v:10.3f}"))
    print("\nMost collinear pairs (|r| > 0.9):")
    strong = pairs[pairs > 0.9]
    if len(strong):
        for (a, b), v in strong.items():
            print(f"  {a:<22} {b:<22} r={v:.3f}")
    else:
        print("  none")
    print(f"\nwrote plots to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
