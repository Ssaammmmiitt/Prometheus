#!/usr/bin/env python3
"""Day 12: U-Net comparison, scored by the same harness as LightGBM.

  python scripts/train_unet.py --years 2024            # one fold
  python scripts/train_unet.py                         # all 10 folds
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from prometheus.cnn import stacks, unet
from prometheus.config import load_settings
from prometheus.eval import baselines, cv, metrics


def score_fold(result: dict, horizon: int) -> dict:
    """Metrics on exactly the population the LightGBM row uses."""
    y = result["labels"].ravel()
    p = result["scores"].ravel()

    clim_rates = baselines.load_or_build_climatology()
    dates = [pd.Timestamp(d) for d in result["dates"]]
    doy = np.array(
        [min(d.dayofyear, clim_rates.shape[0] - 1) for d in dates]
    )[result["valid_days"]]
    y_clim = clim_rates[doy][:, result["rows"], result["cols"]].ravel()

    out = {
        "n": int(y.size),
        "n_pos": int(y.sum()),
        "base_rate": float(y.mean()),
        "pr_auc": metrics.pr_auc(y, p),
        "roc_auc": metrics.roc_auc(y, p),
        "top10_capture": metrics.top_k_capture(y, p, 0.10),
        "clim_pr_auc": metrics.pr_auc(y, y_clim),
    }
    out["skill_vs_clim"] = (out["pr_auc"] - out["clim_pr_auc"]) / max(
        out["clim_pr_auc"], 1e-9
    )
    return out


def lightgbm_reference(out_root: Path, horizon: int) -> pd.DataFrame | None:
    path = out_root / "cv" / "per_fold.csv"
    return pd.read_csv(path) if path.is_file() else None


def main(argv: list[str] | None = None) -> int:
    settings = load_settings()
    all_years = sorted(settings.years.all)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--years", type=int, nargs="*", default=None, help="Holdout years")
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batches-per-epoch", type=int, default=250)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args(argv)

    years = args.years or [y for y in all_years if y != cv.HISTORY_WARMUP_YEAR]
    missing = [y for y in all_years if not stacks.is_cached(y, [args.horizon])]
    if missing:
        print(f"season stacks missing for {missing} — build them first:")
        print("  python -c 'from prometheus.cnn import stacks; stacks.build_all()'")
        return 1

    out_dir = args.out_dir or (Path(settings.root) / "runs" / "unet")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"device {unet.device()} · {len(years)} folds · h{args.horizon} · "
          f"{args.epochs} epochs x {args.batches_per_epoch} batches")

    folds = []
    for i, year in enumerate(years, start=1):
        print(f"\n[{i}/{len(years)}] holdout {year}")
        net, train_result = unet.train_fold(
            year, horizon=args.horizon, epochs=args.epochs,
            batches_per_epoch=args.batches_per_epoch, batch_size=args.batch_size,
            lr=args.lr, pretrained=not args.no_pretrained,
        )
        path = unet.save_net(net, out_dir / f"unet_h{args.horizon}_holdout{year}.pt")
        train_result.model_path = str(path)

        print("    scoring held-out season on the full grid …", flush=True)
        prediction = unet.predict_season(net, year, year, horizon=args.horizon)
        block = score_fold(prediction, args.horizon)
        block.update(
            year=year,
            train_seconds=round(train_result.train_seconds, 1),
            final_loss=train_result.losses[-1] if train_result.losses else None,
        )
        folds.append(block)
        print(f"    U-Net PR-AUC {block['pr_auc']:.4f} · climatology "
              f"{block['clim_pr_auc']:.4f} · top10% {block['top10_capture']:.3f} "
              f"· {train_result.train_seconds / 60:.1f} min")
        (out_dir / f"fold_{year}.json").write_text(
            json.dumps({**block, "losses": train_result.losses}, indent=2, default=float),
            encoding="utf-8",
        )

    frame = pd.DataFrame(folds).sort_values("year")
    frame.to_csv(out_dir / f"per_fold_h{args.horizon}.csv", index=False)

    print("\n" + "=" * 72)
    print("RESULTS — U-Net vs LightGBM, identical folds and pixel population")
    print("=" * 72)

    reference = lightgbm_reference(Path(settings.root) / "runs", args.horizon)
    ref = reference.set_index("year")["pr_auc"].to_dict() if reference is not None else {}

    print(f"\n{'year':>6}{'U-Net':>10}{'LightGBM':>11}{'clim':>9}{'winner':>11}")
    for _, r in frame.iterrows():
        lgb = ref.get(int(r.year))
        winner = "—" if lgb is None else ("U-Net" if r.pr_auc > lgb else "LightGBM")
        shown = f"{lgb:.4f}" if lgb is not None else "—"
        print(f"{int(r.year):>6}{r.pr_auc:>10.4f}{shown:>11}"
              f"{r.clim_pr_auc:>9.4f}{winner:>11}")
    print("-" * 47)
    shared = [int(y) for y in frame.year if int(y) in ref]
    lgb_mean = float(np.mean([ref[y] for y in shared])) if shared else None
    tail = f"{lgb_mean:>11.4f}" if lgb_mean is not None else f"{'—':>11}"
    print(f"{'mean':>6}{frame.pr_auc.mean():>10.4f}{tail}")
    print(f"{'std':>6}{frame.pr_auc.std():>10.4f}")

    if lgb_mean is not None:
        wins = sum(1 for _, r in frame.iterrows() if r.pr_auc > ref.get(int(r.year), np.inf))
        print(f"\nU-Net {frame.pr_auc.mean():.4f} ± {frame.pr_auc.std():.4f} vs "
              f"LightGBM {lgb_mean:.4f} · U-Net wins {wins}/{len(frame)} folds")
        print(f"verdict: {'U-Net' if frame.pr_auc.mean() > lgb_mean else 'LightGBM'} wins")
    print(f"\ntables written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
