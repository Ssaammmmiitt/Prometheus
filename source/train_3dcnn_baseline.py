import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, WeightedRandomSampler

from dataset_loader import FireConvLSTMDataset

DATA_ROOT = "/Users/sammit/Desktop/Projects/Prometheus/data_processed_normalized"
FIRE_ROOT = "/Users/sammit/Desktop/Projects/Prometheus/data_processed/fire16"

TRAIN_CSV = "/Users/sammit/Desktop/Projects/Prometheus/reports/dataset/dataset_index_train_vr50.csv"
VAL_CSV   = "/Users/sammit/Desktop/Projects/Prometheus/reports/dataset/dataset_index_val.csv"
TEST_CSV  = "/Users/sammit/Desktop/Projects/Prometheus/reports/dataset/dataset_index_test.csv"

VARS = ["ndvi16","temp16","precip16","rh16","vpd16","elevation","slope"]

OUT_DIR = Path("/Users/sammit/Desktop/Projects/Prometheus/reports/models/3dcnn_baseline")
OUT_DIR.mkdir(parents=True, exist_ok=True)

class Simple3DCNN(nn.Module):
    """
    Input:  (B, T, C, H, W)
    Output: logits (B, 1, H, W)
    """
    def __init__(self, in_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=(3,3,3), padding=(1,1,1)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

            nn.Conv3d(32, 64, kernel_size=(3,3,3), padding=(1,1,1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

            nn.Conv3d(64, 64, kernel_size=(3,3,3), padding=(1,1,1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.time_pool = nn.AdaptiveAvgPool3d((1, None, None))
        self.head = nn.Conv3d(64, 1, kernel_size=1)

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (B,C,T,H,W)
        x = self.net(x)
        x = self.time_pool(x)                      # (B,64,1,H,W)
        x = self.head(x)                           # (B,1,1,H,W)
        return x.squeeze(2)                        # (B,1,H,W)

def dice_iou_from_logits(logits, targets, thr=0.5, eps=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs >= thr).float()

    if targets.ndim == 3:
        targets = targets.unsqueeze(1)
    targets = targets.float()

    inter = (preds * targets).sum(dim=(1,2,3))
    union = (preds + targets - preds*targets).sum(dim=(1,2,3))

    dice = (2*inter + eps) / (preds.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3)) + eps)
    iou  = (inter + eps) / (union + eps)

    return dice.mean().item(), iou.mean().item()

def precision_recall_from_logits(logits, targets, thr=0.5, eps=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs >= thr).float()

    if targets.ndim == 3:
        targets = targets.unsqueeze(1)
    targets = targets.float()

    tp = (preds * targets).sum(dim=(1,2,3))
    fp = (preds * (1 - targets)).sum(dim=(1,2,3))
    fn = ((1 - preds) * targets).sum(dim=(1,2,3))

    precision = (tp + eps) / (tp + fp + eps)
    recall    = (tp + eps) / (tp + fn + eps)

    return precision.mean().item(), recall.mean().item()

@torch.no_grad()
def estimate_pos_weight(train_loader, device, max_batches=200):
    pos = 0.0
    neg = 0.0
    batches = 0

    for _, y in train_loader:
        y = y.to(device)
        pos += y.sum().item()
        neg += (y.numel() - y.sum().item())
        batches += 1
        if batches >= max_batches:
            break

    if pos < 1:
        return 1.0

    w = neg / pos
    return float(np.clip(w, 1.0, 200.0))

def make_train_loader(batch_size=8):
    train_ds = FireConvLSTMDataset(TRAIN_CSV, DATA_ROOT, FIRE_ROOT, VARS, patch_size=32)

    y_patch = train_ds.df["has_fire"].astype(int).to_numpy()
    counts = np.bincount(y_patch, minlength=2)

    w0 = 1.0 / max(counts[0], 1)
    w1 = 1.0 / max(counts[1], 1)
    weights = np.where(y_patch == 1, w1, w0).astype(np.float32)

    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(weights),
        num_samples=len(weights),
        replacement=True
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True)
    return train_ds, train_loader, counts

def make_eval_loader(csv_path, batch_size=8):
    ds = FireConvLSTMDataset(csv_path, DATA_ROOT, FIRE_ROOT, VARS, patch_size=32)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return ds, loader

def run_epoch(model, loader, optimizer, criterion, device, train=True, thr=0.5):
    model.train() if train else model.eval()

    total_loss = 0.0
    n_batches = 0

    dice_vals, iou_vals, p_vals, r_vals = [], [], [], []

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)

        if train:
            optimizer.zero_grad()

        logits = model(X)                       # (B,1,H,W)
        loss = criterion(logits, y.unsqueeze(1))

        if train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        d, j = dice_iou_from_logits(logits.detach(), y.detach(), thr=thr)
        p, r = precision_recall_from_logits(logits.detach(), y.detach(), thr=thr)

        dice_vals.append(d)
        iou_vals.append(j)
        p_vals.append(p)
        r_vals.append(r)

    return {
        "loss": total_loss / max(n_batches, 1),
        "dice": float(np.mean(dice_vals)),
        "iou": float(np.mean(iou_vals)),
        "precision": float(np.mean(p_vals)),
        "recall": float(np.mean(r_vals)),
    }

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Device:", device)

    batch_size = 8
    epochs = 6
    lr = 1e-3
    thr = 0.5

    train_ds, train_loader, counts = make_train_loader(batch_size=batch_size)
    _, val_loader = make_eval_loader(VAL_CSV, batch_size=batch_size)

    model = Simple3DCNN(in_channels=len(VARS)).to(device)

    pos_weight = estimate_pos_weight(train_loader, device, max_batches=200)
    print("Patch counts [0,1]:", counts, "patch fire ratio:", counts[1]/counts.sum())
    print("Estimated pixel pos_weight:", pos_weight)

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_dice = -1.0
    best_path = OUT_DIR / "best.pt"

    for epoch in range(1, epochs + 1):
        tr = run_epoch(model, train_loader, optimizer, criterion, device, train=True, thr=thr)
        va = run_epoch(model, val_loader, optimizer=None, criterion=criterion, device=device, train=False, thr=thr)

        print(
            f"Epoch {epoch:02d} | "
            f"train loss {tr['loss']:.4f} dice {tr['dice']:.4f} iou {tr['iou']:.4f} P {tr['precision']:.4f} R {tr['recall']:.4f} | "
            f"val loss {va['loss']:.4f} dice {va['dice']:.4f} iou {va['iou']:.4f} P {va['precision']:.4f} R {va['recall']:.4f}"
        )

        if va["dice"] > best_val_dice:
            best_val_dice = va["dice"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "pos_weight": pos_weight,
                    "thr": thr,
                    "vars": VARS
                },
                best_path
            )
            print("Saved best checkpoint:", best_path)

    print("Training complete. Best val dice:", best_val_dice)

if __name__ == "__main__":
    main()
