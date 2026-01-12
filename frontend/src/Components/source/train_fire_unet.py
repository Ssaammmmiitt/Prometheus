import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", DEVICE)

DATA_DIR = Path("/Users/sammit/Desktop/Projects/Prometheus/data_processed/normalized/patches_p32_s32")

TRAIN_FILE_A = "Xy_in0102_to_03_p32_s32.npz"
TRAIN_FILE_B = "Xy_in0203_to_04_p32_s32.npz"
TEST_FILE = "Xy_in0304_to_05_p32_s32.npz"

BATCH_SIZE = 32
EPOCHS = 60
LR = 1e-3
GRAD_CLIP_NORM = 1.0

VAL_FRACTION_OF_FILE_B = 0.25
SEED = 42

DICE_WEIGHT = 0.30
POS_WEIGHT_MIN = 50.0
POS_WEIGHT_MAX = 200.0

torch.manual_seed(SEED)
np.random.seed(SEED)

def load_npz(fname: str):
    d = np.load(DATA_DIR / fname)
    X = d["X"].astype(np.float32)            # N, C, H, W
    y = d["y"].astype(np.float32)            # N, H, W
    y = y[:, None, :, :]                     # N, 1, H, W
    return X, y

class PatchDataset(Dataset):
    def __init__(self, X, y, indices=None, augment=False):
        self.X = X
        self.y = y
        self.indices = indices if indices is not None else np.arange(len(X))
        self.augment = augment

    def set_indices(self, indices):
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        x = torch.from_numpy(self.X[idx])     # C, H, W
        y = torch.from_numpy(self.y[idx])     # 1, H, W

        if self.augment:
            if np.random.rand() > 0.5:
                k = np.random.randint(4)
                x = torch.rot90(x, k, dims=[1, 2])
                y = torch.rot90(y, k, dims=[1, 2])

                if np.random.rand() > 0.5:
                    x = torch.flip(x, dims=[2])
                    y = torch.flip(y, dims=[2])

        return x, y

class SmallUNet(nn.Module):
    def __init__(self, in_ch=7, base=32):
        super().__init__()

        def conv_block(c_in, c_out):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, padding=1),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.Conv2d(c_out, c_out, 3, padding=1),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
            )

        self.enc1 = conv_block(in_ch, base)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = conv_block(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = conv_block(base * 2, base * 4)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = conv_block(base * 4, base * 2)

        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = conv_block(base * 2, base)

        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))

        d2 = self.up2(b)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.out(d1)

class DiceBCELoss(nn.Module):
    def __init__(self, pos_weight: float, dice_weight: float, device: str):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        bce = self.bce(logits, targets)

        probs = torch.sigmoid(logits)
        smooth = 1.0
        dims = (0, 2, 3)
        inter = (probs * targets).sum(dims)
        denom = probs.sum(dims) + targets.sum(dims)
        dice = (2.0 * inter + smooth) / (denom + smooth)
        dice_loss = 1.0 - dice.mean()

        return (1.0 - self.dice_weight) * bce + self.dice_weight * dice_loss

@torch.no_grad()
def compute_metrics_from_logits(logits, y_true, threshold):
    probs = torch.sigmoid(logits)
    y_pred = (probs >= threshold).float()

    tp = (y_pred * y_true).sum().item()
    fp = (y_pred * (1 - y_true)).sum().item()
    fn = ((1 - y_pred) * y_true).sum().item()
    tn = ((1 - y_pred) * (1 - y_true)).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    return acc, precision, recall, f1

@torch.no_grad()
def collect_logits_and_targets(model, loader):
    model.eval()
    all_logits = []
    all_y = []
    for Xb, yb in loader:
        Xb = Xb.to(DEVICE)
        yb = yb.to(DEVICE)
        logits = model(Xb)
        all_logits.append(logits.cpu())
        all_y.append(yb.cpu())
    return torch.cat(all_logits, dim=0), torch.cat(all_y, dim=0)

@torch.no_grad()
def find_best_threshold_f1(model, loader):
    logits, y_true = collect_logits_and_targets(model, loader)
    best_t = 0.5
    best_f1 = -1.0

    for t in np.linspace(0.01, 0.99, 50):
        _, _, _, f1 = compute_metrics_from_logits(logits, y_true, float(t))
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)

    return best_t, best_f1

@torch.no_grad()
def evaluate(model, loader, threshold):
    logits, y_true = collect_logits_and_targets(model, loader)
    return compute_metrics_from_logits(logits, y_true, threshold)

def main():
    X_a, y_a = load_npz(TRAIN_FILE_A)
    X_b, y_b = load_npz(TRAIN_FILE_B)
    X_test, y_test = load_npz(TEST_FILE)

    rng = np.random.default_rng(SEED)

    idx_b = np.arange(len(X_b))
    rng.shuffle(idx_b)
    val_size_b = int(len(idx_b) * VAL_FRACTION_OF_FILE_B)
    val_idx_b = idx_b[:val_size_b]
    train_idx_b = idx_b[val_size_b:]

    X_train = np.concatenate([X_a, X_b[train_idx_b]], axis=0)
    y_train = np.concatenate([y_a, y_b[train_idx_b]], axis=0)
    X_val = X_b[val_idx_b]
    y_val = y_b[val_idx_b]

    print("Train patches:", len(X_train))
    print("Val patches:", len(X_val))
    print("Test patches:", len(X_test))

    patch_fire = y_train.sum(axis=(1, 2, 3))
    pos_idx = np.where(patch_fire > 0)[0]
    neg_idx = np.where(patch_fire == 0)[0]
    print("Train positive patches:", len(pos_idx), "Train negative patches:", len(neg_idx))

    train_pos_pixels = float(y_train.sum())
    train_total_pixels = float(y_train.size)
    train_neg_pixels = train_total_pixels - train_pos_pixels
    raw_pos_weight = train_neg_pixels / max(train_pos_pixels, 1.0)

    pos_weight_value = float(np.clip(raw_pos_weight, POS_WEIGHT_MIN, POS_WEIGHT_MAX))
    print("Train fire pixel rate:", train_pos_pixels / train_total_pixels)
    print("Raw pos_weight:", raw_pos_weight)
    print("Clipped pos_weight:", pos_weight_value)

    train_ds = PatchDataset(X_train, y_train, indices=np.arange(len(X_train)), augment=True)
    val_ds = PatchDataset(X_val, y_val, indices=np.arange(len(X_val)), augment=False)
    test_ds = PatchDataset(X_test, y_test, indices=np.arange(len(X_test)), augment=False)

    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = SmallUNet(in_ch=X_train.shape[1], base=32).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    loss_fn = DiceBCELoss(pos_weight=pos_weight_value, dice_weight=DICE_WEIGHT, device=DEVICE)

    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    best_val_f1 = -1.0
    best_state = None
    best_thr = 0.5
    best_epoch = 0
    bad_epochs = 0
    patience = 10

    for epoch in range(1, EPOCHS + 1):
        model.train()

        n_pos = len(pos_idx)
        neg_sample = rng.choice(
            neg_idx,
            size=n_pos,
            replace=False if len(neg_idx) >= n_pos else True
        )
        epoch_indices = np.concatenate([pos_idx, neg_sample])
        rng.shuffle(epoch_indices)
        train_ds.set_indices(epoch_indices)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

        total_loss = 0.0
        for Xb, yb in train_loader:
            Xb = Xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            logits = model(Xb)
            loss = loss_fn(logits, yb)
            loss.backward()

            if GRAD_CLIP_NORM and GRAD_CLIP_NORM > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)

            optimizer.step()
            total_loss += loss.item() * Xb.size(0)

        avg_loss = total_loss / len(train_ds)

        thr, val_f1 = find_best_threshold_f1(model, val_loader)
        val_acc, val_p, val_r, val_f1_eval = evaluate(model, val_loader, thr)
        scheduler.step(val_f1_eval)

        print(
            f"Epoch {epoch:02d} loss {avg_loss:.4f} "
            f"val_thr {thr:.2f} acc {val_acc:.4f} "
            f"P {val_p:.4f} R {val_r:.4f} F1 {val_f1_eval:.4f} "
            f"lr {optimizer.param_groups[0]['lr']:.2e}"
        )

        if val_f1_eval > best_val_f1 + 1e-4:
            best_val_f1 = val_f1_eval
            best_thr = thr
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print("Early stopping")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        print("Loaded best model from epoch:", best_epoch, "val F1:", best_val_f1, "thr:", best_thr)

    test_acc, test_p, test_r, test_f1 = evaluate(model, test_loader, best_thr)
    print(f"Test at thr {best_thr:.2f} acc {test_acc:.4f} P {test_p:.4f} R {test_r:.4f} F1 {test_f1:.4f}")

    out_path = DATA_DIR / "fire_unet_p32_s32_best.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "best_threshold": float(best_thr),
            "val_f1": float(best_val_f1),
            "epoch": int(best_epoch),
            "pos_weight": float(pos_weight_value),
            "raw_pos_weight": float(raw_pos_weight),
            "dice_weight": float(DICE_WEIGHT),
        },
        out_path,
    )
    print("Saved checkpoint to", out_path)

if __name__ == "__main__":
    main()
