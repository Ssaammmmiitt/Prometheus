"""U-Net training and full-grid inference, scored by the same harness as LightGBM."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from prometheus.cnn.data import PATCH, PatchSampler, normaliser
from prometheus.cnn.stacks import SeasonStack
from prometheus.config import load_settings
from prometheus.features import forest

#: The encoder downsamples by 32, so the raster is padded up to a multiple of it.
ALIGN = 32


def device():
    """
    Prefer Apple Metal, then CUDA, then CPU.

    On macOS 26 + Torch 2.13, MPS works in a normal shell but
    `torch.backends.mps.is_available()` falsely returns False inside the Cursor
    / Codex sandboxed seatbelt. Training and inference should be launched outside
    that sandbox (or with full permissions) so this path sees the real GPU.
    """
    import torch

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_unet(in_channels: int, encoder: str = "resnet18", pretrained: bool = True):
    import segmentation_models_pytorch as smp

    return smp.Unet(
        encoder_name=encoder,
        encoder_weights="imagenet" if pretrained else None,
        in_channels=in_channels,
        classes=1,
    )


class MaskedFocalTversky:
    """
    Focal + Tversky (β = 0.7), evaluated only on forest pixels.

    Tversky with β > α punishes false negatives harder than false positives,
    which is the right asymmetry for a rare hazard. Both terms are masked so the
    ~60 % of every raster that lies outside the country contributes no gradient.
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.7, gamma: float = 2.0,
                 focal_weight: float = 1.0, tversky_weight: float = 1.0):
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.focal_weight, self.tversky_weight = focal_weight, tversky_weight

    def __call__(self, logits, target, mask):
        import torch
        import torch.nn.functional as F

        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * target + (1 - p) * (1 - target)
        focal = ((1 - p_t) ** self.gamma * bce * mask).sum() / mask.sum().clamp(min=1)

        p_m, t_m = p * mask, target * mask
        tp = (p_m * t_m).sum()
        fn = ((1 - p_m) * t_m).sum()
        fp = (p_m * (1 - t_m)).sum()
        tversky = 1 - (tp + 1.0) / (tp + self.alpha * fp + self.beta * fn + 1.0)
        return self.focal_weight * focal + self.tversky_weight * tversky


@dataclass
class TrainResult:
    holdout_year: int
    epochs: int
    steps: int
    train_seconds: float
    losses: list[float] = field(default_factory=list)
    model_path: str | None = None


def train_fold(
    holdout_year: int,
    *,
    years: list[int] | None = None,
    horizon: int = 1,
    epochs: int = 20,
    batches_per_epoch: int = 250,
    batch_size: int = 32,
    lr: float = 3e-4,
    pretrained: bool = True,
    seed: int = 0,
    verbose: bool = True,
) -> tuple[object, TrainResult]:
    import torch

    settings = load_settings()
    all_years = sorted(settings.years.all)
    years = years or [y for y in all_years if y != holdout_year]

    torch.manual_seed(seed)
    sampler = PatchSampler(
        years, holdout_year, horizon=horizon, batch_size=batch_size, seed=seed
    )
    dev = device()
    net = build_unet(len(sampler.features), pretrained=pretrained).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    criterion = MaskedFocalTversky()

    started = time.perf_counter()
    losses: list[float] = []
    net.train()
    for epoch in range(epochs):
        running, seen = 0.0, 0
        for batch in sampler.batches(batches_per_epoch):
            x = torch.from_numpy(batch.x).to(dev)
            y = torch.from_numpy(batch.y).to(dev)
            m = torch.from_numpy(batch.mask).to(dev)
            opt.zero_grad()
            loss = criterion(net(x), y, m)
            loss.backward()
            opt.step()
            running += float(loss.detach().cpu())
            seen += 1
        schedule.step()
        losses.append(running / max(seen, 1))
        if verbose:
            print(f"    epoch {epoch + 1:>2}/{epochs}  loss {losses[-1]:.4f}"
                  f"  ({time.perf_counter() - started:.0f}s)", flush=True)

    return net, TrainResult(
        holdout_year=holdout_year,
        epochs=epochs,
        steps=epochs * batches_per_epoch,
        train_seconds=time.perf_counter() - started,
        losses=losses,
    )


def predict_season(
    net, year: int, holdout_year: int, *, horizon: int = 1, batch_days: int = 4
) -> dict:
    """
    Score a whole season, returning (n_valid_days, n_forest_cells) like LightGBM.

    Full rasters go through the network rather than tiles, so there are no patch
    seams, and the output is reduced to the same forest cells the tabular model
    predicts on — that is what makes the two rows comparable.
    """
    import torch
    import torch.nn.functional as F

    stack = SeasonStack(year, horizon)
    mean, std = normaliser(stack.features, holdout_year)
    mask = forest.forest_mask()
    rows, cols = np.where(mask)
    dev = device()

    h, w = mask.shape
    pad_h = (ALIGN - h % ALIGN) % ALIGN
    pad_w = (ALIGN - w % ALIGN) % ALIGN

    valid_idx = np.where(stack.valid_days)[0]
    scores = np.empty((valid_idx.size, rows.size), dtype=np.float32)

    net.eval()
    with torch.no_grad():
        for start in range(0, valid_idx.size, batch_days):
            chunk = valid_idx[start : start + batch_days]
            planes = np.stack([(stack.day(int(t)) - mean) / std for t in chunk])
            x = torch.from_numpy(planes.astype(np.float32)).to(dev)
            if pad_h or pad_w:
                x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
            logits = net(x)[:, :, :h, :w]
            probability = torch.sigmoid(logits)[:, 0].cpu().numpy()
            scores[start : start + len(chunk)] = probability[:, rows, cols]

    labels = np.asarray(stack.labels)[stack.valid_days][:, rows, cols]
    return {
        "year": year,
        "scores": scores,
        "labels": labels.astype(np.uint8),
        "rows": rows,
        "cols": cols,
        "valid_days": stack.valid_days,
        "dates": stack.dates,
    }


def save_net(net, path: Path) -> Path:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), path)
    return path


def load_net(path: Path, in_channels: int, *, encoder: str = "resnet18"):
    import torch

    net = build_unet(in_channels, encoder=encoder, pretrained=False)
    net.load_state_dict(torch.load(path, map_location="cpu"))
    return net.to(device())


__all__ = [
    "ALIGN",
    "MaskedFocalTversky",
    "PATCH",
    "TrainResult",
    "build_unet",
    "device",
    "load_net",
    "predict_season",
    "save_net",
    "train_fold",
]
