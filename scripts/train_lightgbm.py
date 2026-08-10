#!/usr/bin/env python3
"""Day 9: train LightGBM and score it on the full grid.

  python scripts/train_lightgbm.py --year 2021              # one fold
  python scripts/train_lightgbm.py --year 2021 --search 24  # + random search
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prometheus.config import load_settings
from prometheus.features.table import load_train_table, present_features, train_table_path
from prometheus.models import lgbm


def main(argv: list[str] | None = None) -> int:
    settings = load_settings()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--year", type=int, default=settings.years.all[-1], help="Held-out season")
    ap.add_argument("--search", type=int, default=0, help="Random-search configs (0 = skip)")
    ap.add_argument("--label", default=lgbm.PRIMARY_LABEL)
    ap.add_argument("--no-grid-eval", action="store_true", help="Skip full-grid scoring")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args(argv)

    if not train_table_path().exists():
        print("train_table.parquet missing — run scripts/build_train_table.py")
        return 1

    out_dir = args.out_dir or (Path(settings.root) / "runs" / "lightgbm")
    out_dir.mkdir(parents=True, exist_ok=True)

    table = load_train_table()
    features = present_features(table)
    print(f"{len(table):,} rows · {len(features)} features · holdout {args.year}")

    params = None
    if args.search:
        print(f"\nRandom search over {args.search} configs (inner validation season):")
        params, frame = lgbm.search(table, args.year, n_configs=args.search, label=args.label)
        frame.to_csv(out_dir / f"search_holdout{args.year}.csv", index=False)
        print(f"\nbest: {params}")

    print("\nTraining final fold …")
    booster, result = lgbm.train_fold(table, args.year, params=params, label=args.label)
    model_path = lgbm.save_model(booster, result)

    print(f"  trained in {result.train_seconds:.1f}s · {result.best_iteration} trees")
    print(f"  inner-validation PR-AUC {result.valid_pr_auc:.4f}")
    print(f"  model → {model_path}")

    if args.no_grid_eval:
        return 0

    print("\nScoring every forest cell of the held-out season …")
    horizon = int(args.label.removeprefix("label_h"))
    grid_metrics = lgbm.evaluate_grid(booster, args.year, horizon=horizon)
    result.grid_metrics = grid_metrics
    lgbm.save_model(booster, result)

    (out_dir / f"grid_metrics_holdout{args.year}.json").write_text(
        json.dumps(grid_metrics, indent=2), encoding="utf-8"
    )

    print("\n=== Day 9 report — held-out season {} ===".format(args.year))
    print(f"pixels scored:  {grid_metrics['n']:,}  positives {grid_metrics['n_pos']:,} "
          f"(base rate {grid_metrics['base_rate']:.4%})")
    print(f"\n{'model':<14}{'PR-AUC':>9}{'top10%':>9}")
    print(f"{'LightGBM':<14}{grid_metrics['model_pr_auc']:>9.4f}"
          f"{grid_metrics['model_top10_capture']:>9.4f}")
    print(f"{'climatology':<14}{grid_metrics['clim_pr_auc']:>9.4f}"
          f"{grid_metrics['clim_top10_capture']:>9.4f}")
    print(f"{'persistence':<14}{grid_metrics['persistence_pr_auc']:>9.4f}"
          f"{grid_metrics['persistence_top10_capture']:>9.4f}")
    print(f"\nskill vs climatology: {grid_metrics['skill_vs_clim']:+.1%}")

    ok = grid_metrics["beats_climatology"] and result.train_seconds < 120
    print(f"trains < 2 min:       {'PASS' if result.train_seconds < 120 else 'FAIL'} "
          f"({result.train_seconds:.1f}s)")
    print(f"beats climatology:    {'PASS' if grid_metrics['beats_climatology'] else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
