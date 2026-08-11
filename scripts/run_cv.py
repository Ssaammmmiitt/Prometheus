#!/usr/bin/env python3
"""Day 10: leave-one-year-out CV with family ablations and a per-region breakdown.

  python scripts/run_cv.py                   # all 10 folds, all ablations
  python scripts/run_cv.py --years 2021 2024 # quick subset
  python scripts/run_cv.py --no-ablations    # full model only
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from prometheus.config import load_settings
from prometheus.eval import cv
from prometheus.features.table import load_train_table, train_table_path


def _fmt(mean: float, std: float, digits: int = 4) -> str:
    if pd.isna(std):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def report(summary: dict[str, pd.DataFrame]) -> None:
    per_fold, ablation = summary["per_fold"], summary["ablation"]
    regions, shap_tbl = summary["regions"], summary["shap_global"]
    full = ablation[ablation["variant"] == "full"].iloc[0]

    print("\n" + "=" * 72)
    print("RESULTS — leave-one-year-out, scored on the full forest grid")
    print("=" * 72)
    print(f"\n{'year':>6}{'PR-AUC':>10}{'clim':>10}{'persist':>10}"
          f"{'top10%':>9}{'base rate':>11}")
    for _, r in per_fold.iterrows():
        print(f"{int(r.year):>6}{r.pr_auc:>10.4f}{r.clim_pr_auc:>10.4f}"
              f"{r.persistence_pr_auc:>10.4f}{r.top10_capture:>9.3f}"
              f"{r.base_rate:>10.3%}")
    print("-" * 56)
    print(f"{'mean':>6}{per_fold.pr_auc.mean():>10.4f}"
          f"{per_fold.clim_pr_auc.mean():>10.4f}"
          f"{per_fold.persistence_pr_auc.mean():>10.4f}"
          f"{per_fold.top10_capture.mean():>9.3f}"
          f"{per_fold.base_rate.mean():>10.3%}")
    print(f"{'std':>6}{per_fold.pr_auc.std():>10.4f}"
          f"{per_fold.clim_pr_auc.std():>10.4f}"
          f"{per_fold.persistence_pr_auc.std():>10.4f}"
          f"{per_fold.top10_capture.std():>9.3f}")
    print(f"\nLightGBM  PR-AUC {_fmt(full.pr_auc_mean, full.pr_auc_std)} "
          f"vs climatology {per_fold.clim_pr_auc.mean():.4f} "
          f"({per_fold.skill_vs_clim.mean():+.0%} skill)")

    if len(ablation) > 1:
        print("\n" + "=" * 72)
        print("ABLATIONS — mean ΔPR-AUC across folds when a family is removed")
        print("=" * 72)
        print(f"\n{'variant':<20}{'feats':>7}{'PR-AUC':>20}{'Δ':>10}{'Δ%':>9}")
        for _, r in ablation.iterrows():
            delta = (
                ""
                if r.variant == "full"
                else f"{r.delta_mean:>+10.4f}{r.delta_pct_mean:>+8.1f}%"
            )
            print(f"{r.variant:<20}{int(r.n_features):>7}"
                  f"{_fmt(r.pr_auc_mean, r.pr_auc_std):>20}{delta}")

    if not regions.empty:
        print("\n" + "=" * 72)
        print("PER-REGION — physiographic breakdown, full model")
        print("=" * 72)
        print(f"\n{'region':<18}{'positives':>11}{'base rate':>11}"
              f"{'PR-AUC':>20}{'clim':>9}{'skill':>9}")
        for _, r in regions.iterrows():
            print(f"{r.region:<18}{int(r.n_pos):>11,}{r.base_rate:>10.2%}"
                  f"{_fmt(r.pr_auc_mean, r.pr_auc_std):>20}"
                  f"{r.clim_pr_auc:>9.4f}{r.skill_mean:>+8.0%}")

    print("\n" + "=" * 72)
    print("SHAP — global mean |contribution|, averaged over folds")
    print("=" * 72)
    for _, r in shap_tbl.head(12).iterrows():
        print(f"  {r.feature:<22}{r.share_pct:>6.1f}%   (±{r.std_across_folds:.4f} across folds)")

    region_shap = summary.get("region_shap")
    if region_shap is not None and not region_shap.empty:
        print("\ntop drivers by region (SHAP share):")
        for region in cv.REGION_NAMES.values():
            sub = region_shap[region_shap["region"] == region]
            if sub.empty:
                continue
            top = sub.nlargest(4, "share_pct")
            joined = ", ".join(f"{r.feature} {r.share_pct:.0f}%" for _, r in top.iterrows())
            print(f"  {region:<18}{joined}")

    family = summary.get("region_family_shap")
    if family is not None and not family.empty:
        cols = [c for c in family.columns if c != "region"]
        print("\nSHAP share by family and region (%):")
        print(f"  {'region':<18}" + "".join(f"{c:>14}" for c in cols))
        for _, r in family.iterrows():
            print(f"  {r.region:<18}" + "".join(f"{r[c]:>14.1f}" for c in cols))


def main(argv: list[str] | None = None) -> int:
    settings = load_settings()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--years", type=int, nargs="*", default=None)
    ap.add_argument("--no-ablations", action="store_true")
    ap.add_argument("--shap-sample", type=int, default=20_000)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args(argv)

    if not train_table_path().exists():
        print("train_table.parquet missing — run scripts/build_train_table.py")
        return 1

    table = load_train_table()
    drops = [None] if args.no_ablations else [None, *cv.feature_families()]
    years = args.years or [
        y for y in sorted(settings.years.all) if y != cv.HISTORY_WARMUP_YEAR
    ]

    print(f"{len(table):,} rows · {len(years)} folds · {len(drops)} variants per fold")
    print(f"holdout years: {years}")
    print(f"{cv.HISTORY_WARMUP_YEAR} is history warm-up only — trains, never held out")

    out_dir = args.out_dir or (Path(settings.root) / "runs" / "cv")
    result = cv.run_cv(
        table, years=years, drops=drops, shap_sample=args.shap_sample, out_dir=out_dir
    )
    report(result["summary"])
    print(f"\ntables written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
