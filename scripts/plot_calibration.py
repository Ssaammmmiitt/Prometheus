#!/usr/bin/env python3
"""Day 11: reliability diagrams before and after isotonic calibration.

  python scripts/plot_calibration.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from prometheus.config import load_settings  # noqa: E402
from prometheus.models import calibrate as cal  # noqa: E402


def panel(ax, y, scores, title, *, n_bins: int, color: str):
    curve = cal.reliability_curve(y, scores, n_bins=n_bins, strategy="quantile")
    ece = cal.expected_calibration_error(y, scores, n_bins=n_bins, strategy="quantile")

    hi = max(curve["mean_predicted"].max(), curve["observed_frequency"].max())
    lo = max(min(curve["mean_predicted"].min(), curve["observed_frequency"].min()), 1e-6)
    ax.plot([lo, hi], [lo, hi], ls="--", lw=1, color="0.5", label="perfect")
    ax.plot(
        curve["mean_predicted"], curve["observed_frequency"],
        "o-", ms=4, lw=1.4, color=color, label="model",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed frequency")
    ax.set_title(f"{title}\nECE = {ece:.5f}", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.25, which="both", lw=0.4)


def main(argv: list[str] | None = None) -> int:
    settings = load_settings()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 7])
    ap.add_argument("--bins", type=int, default=15)
    ap.add_argument("--in-dir", type=Path, default=None)
    args = ap.parse_args(argv)

    in_dir = args.in_dir or (Path(settings.root) / "runs" / "calibration")
    for horizon in args.horizons:
        path = in_dir / f"reliability_h{horizon}.npz"
        if not path.is_file():
            print(f"missing {path} — run scripts/build_model_bundle.py")
            return 1
        data = np.load(path)
        y = data["y"]

        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
        panel(axes[0], y, data["raw"], f"h{horizon} raw booster score",
              n_bins=args.bins, color="#c0392b")
        panel(axes[1], y, data["calibrated"], f"h{horizon} after isotonic calibration",
              n_bins=args.bins, color="#1f7a4d")
        # Log-log axes: at a sub-1 % base rate a linear diagram is a dot in the
        # corner, and the interesting behaviour spans four orders of magnitude.
        fig.suptitle(
            f"Reliability, held-out season · base rate {y.mean():.3%} "
            f"· equal-count bins",
            fontsize=11,
        )
        fig.tight_layout()
        out = in_dir / f"reliability_h{horizon}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
