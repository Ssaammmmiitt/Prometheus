"""Patch sampling and leak-free normalisation for the convolutional model."""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from prometheus.cnn.stacks import SeasonStack
from prometheus.config import load_settings
from prometheus.features import forest

PATCH = 128


def load_norm_stats(holdout_year: int) -> dict[str, dict[str, float]]:
    """
    Per-fold channel statistics, reused from the tabular pipeline.

    These were computed from training years only, so the held-out season never
    influences its own normalisation — the same guarantee the LightGBM path has.
    """
    path = load_settings().paths.resolve("models") / "norm_stats_v1.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["folds"][str(holdout_year)]


def normaliser(features: list[str], holdout_year: int) -> tuple[np.ndarray, np.ndarray]:
    stats = load_norm_stats(holdout_year)
    mean = np.array([stats[f]["mean"] for f in features], dtype=np.float32)
    std = np.array([max(stats[f]["std"], 1e-6) for f in features], dtype=np.float32)
    return mean.reshape(-1, 1, 1), std.reshape(-1, 1, 1)


def patch_positions(min_forest: float = 0.05, stride: int = 64) -> list[tuple[int, int]]:
    """Top-left corners of patches with enough forest to be worth training on."""
    mask = forest.forest_mask()
    h, w = mask.shape
    out = []
    for r in range(0, h - PATCH + 1, stride):
        for c in range(0, w - PATCH + 1, stride):
            if mask[r : r + PATCH, c : c + PATCH].mean() >= min_forest:
                out.append((r, c))
    return out


@dataclass
class Batch:
    x: np.ndarray  # (B, C, PATCH, PATCH) normalised
    y: np.ndarray  # (B, 1, PATCH, PATCH) labels
    mask: np.ndarray  # (B, 1, PATCH, PATCH) forest validity


class PatchSampler:
    """
    Draws training patches season by season, one day-plane at a time.

    Reading a day-plane is a single contiguous 37 MB read, so several patch
    positions are taken from each plane rather than sampling positions
    independently — random access into a 5.7 GB memmap would amplify reads by
    more than an order of magnitude.

    Days containing fire are oversampled. At a 0.03 % pixel rate an unweighted
    sample is almost all empty rasters, and the model would learn the prior
    rather than the signal. This biases the score scale, not the ranking, and
    PR-AUC is rank-based.
    """

    def __init__(
        self,
        years: list[int],
        holdout_year: int,
        *,
        horizon: int = 1,
        batch_size: int = 32,
        positions_per_plane: int = 16,
        fire_day_weight: float = 8.0,
        fire_patch_fraction: float = 0.5,
        seed: int = 0,
    ):
        self.years = list(years)
        self.horizon = horizon
        self.batch_size = batch_size
        self.positions_per_plane = positions_per_plane
        self.fire_patch_fraction = fire_patch_fraction
        self.rng = np.random.default_rng(seed)

        self.stacks = {y: SeasonStack(y, horizon) for y in self.years}
        any_stack = next(iter(self.stacks.values()))
        self.features = any_stack.features
        self.mean, self.std = normaliser(self.features, holdout_year)
        self.positions = patch_positions()
        self.forest = forest.forest_mask().astype(np.float32)

        self.plane_pool: list[tuple[int, int]] = []
        self.plane_weights: list[float] = []
        for year, stack in self.stacks.items():
            fires_per_day = np.asarray(stack.labels).reshape(stack.n_days, -1).sum(axis=1)
            for t in range(stack.n_days):
                if not stack.valid_days[t]:
                    continue
                self.plane_pool.append((year, t))
                self.plane_weights.append(fire_day_weight if fires_per_day[t] > 0 else 1.0)
        weights = np.asarray(self.plane_weights, dtype=np.float64)
        self.plane_probs = weights / weights.sum()

    def _positions_for(self, label_plane: np.ndarray) -> list[tuple[int, int]]:
        """Prefer patches that actually contain fire, then fill with random ones."""
        want = self.positions_per_plane
        with_fire = [
            (r, c) for r, c in self.positions
            if label_plane[r : r + PATCH, c : c + PATCH].any()
        ]
        n_fire = min(len(with_fire), int(round(want * self.fire_patch_fraction)))
        chosen: list[tuple[int, int]] = []
        if n_fire:
            idx = self.rng.choice(len(with_fire), size=n_fire, replace=False)
            chosen += [with_fire[i] for i in idx]
        remaining = want - len(chosen)
        if remaining > 0:
            idx = self.rng.choice(
                len(self.positions), size=remaining, replace=remaining > len(self.positions)
            )
            chosen += [self.positions[i] for i in idx]
        return chosen

    def batches(self, n_batches: int):
        """Yield `n_batches` batches; patches are drawn plane by plane."""
        per_plane = self.positions_per_plane
        planes_per_batch = max(1, self.batch_size // per_plane)
        for _ in range(n_batches):
            xs, ys, ms = [], [], []
            picks = self.rng.choice(
                len(self.plane_pool), size=planes_per_batch, p=self.plane_probs
            )
            for pick in np.atleast_1d(picks):
                year, t = self.plane_pool[int(pick)]
                stack = self.stacks[year]
                plane = stack.day(t)
                label_plane = stack.label_day(t)
                plane = (plane - self.mean) / self.std
                for r, c in self._positions_for(label_plane):
                    xs.append(plane[:, r : r + PATCH, c : c + PATCH])
                    ys.append(label_plane[None, r : r + PATCH, c : c + PATCH])
                    ms.append(self.forest[None, r : r + PATCH, c : c + PATCH])
            yield Batch(
                x=np.stack(xs).astype(np.float32),
                y=np.stack(ys).astype(np.float32),
                mask=np.stack(ms).astype(np.float32),
            )


def normalise_plane(plane: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((plane - mean) / std).astype(np.float32)


__all__ = [
    "PATCH",
    "Batch",
    "PatchSampler",
    "load_norm_stats",
    "normalise_plane",
    "normaliser",
    "patch_positions",
]
