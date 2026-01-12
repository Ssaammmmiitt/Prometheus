import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# 1) POINT THIS TO YOUR RUN
# -------------------------
RUN_DIR = Path("/Users/sammit/Desktop/Projects/Prometheus/source/runs")  # change this
CKPT_BEST = RUN_DIR / "best.pt"

OUT_DIR = RUN_DIR / "inference_report"
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "val_examples").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "test_examples").mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
print("RUN_DIR:", RUN_DIR)

# -------------------------
# 2) REDECLARE MODEL CLASSES
# Must match training code
# -------------------------
class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_channels = int(hidden_channels)
        self.conv = nn.Conv2d(
            in_channels=int(input_channels) + int(hidden_channels),
            out_channels=4 * int(hidden_channels),
            kernel_size=int(kernel_size),
            padding=int(padding)
        )

    def forward(self, x, state):
        h, c = state
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i); f = torch.sigmoid(f); o = torch.sigmoid(o); g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_state(self, batch, spatial, device, dtype):
        H, W = spatial
        h = torch.zeros(batch, self.hidden_channels, H, W, device=device, dtype=dtype)
        c = torch.zeros(batch, self.hidden_channels, H, W, device=device, dtype=dtype)
        return h, c

class ConvLSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels, num_layers=1, kernel_size=3):
        super().__init__()
        self.num_layers = int(num_layers)
        if isinstance(hidden_channels, (list, tuple)):
            hidden_list = list(hidden_channels)
        else:
            hidden_list = [int(hidden_channels)] * self.num_layers
        self.cells = nn.ModuleList([])
        for i in range(self.num_layers):
            in_ch = int(input_channels) if i == 0 else int(hidden_list[i - 1])
            self.cells.append(ConvLSTMCell(in_ch, int(hidden_list[i]), kernel_size=kernel_size))

    def forward(self, x):
        B, T, C, H, W = x.shape
        layer_input = x
        for cell in self.cells:
            h, c = cell.init_state(B, (H, W), x.device, x.dtype)
            outputs = []
            for t in range(T):
                h, c = cell(layer_input[:, t], (h, c))
                outputs.append(h)
            layer_input = torch.stack(outputs, dim=1)
        return layer_input

class FirePatchConvLSTM(nn.Module):
    def __init__(self, in_channels, hidden=64, lstm_layers=1, kernel=3, dropout=0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.convlstm = ConvLSTM(
            input_channels=32,
            hidden_channels=[hidden] * int(lstm_layers),
            num_layers=int(lstm_layers),
            kernel_size=int(kernel)
        )
        self.drop = nn.Dropout(float(dropout))
        self.head = nn.Sequential(
            nn.Conv2d(hidden, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x2 = x.reshape(B * T, C, H, W)
        f2 = self.encoder(x2)
        feats = f2.reshape(B, T, 32, H, W)
        seq = self.convlstm(feats)
        h_last = seq[:, -1]
        h_last = self.drop(h_last)
        pooled = self.head(h_last).reshape(B, 64)
        logits = self.fc(pooled)
        return logits

# -------------------------
# 3) LOAD CHECKPOINT + CONFIG
# -------------------------
ckpt = torch.load(CKPT_BEST, map_location=device, weights_only=False)
cfg = ckpt.get("config", {})
best_payload = ckpt.get("best_payload", {})
print("Best epoch:", ckpt.get("epoch"))
print("Stored val best thr:", best_payload.get("val_best_thr"))

# Infer in_channels from config
vars_list = cfg.get("VARS", [])
add_mask = cfg.get("ADD_MISSINGNESS_MASK", True)
in_channels = len(vars_list) + (1 if add_mask else 0)

model = FirePatchConvLSTM(
    in_channels=in_channels,
    hidden=64,
    lstm_layers=1,
    kernel=3,
    dropout=0.2
).to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()

# -------------------------
# 4) REUSE YOUR DATASET / LOADER
# Import from your training cell if it still exists.
# If not, re-run the cell that defines FirePatchSeqDataset and make_loader.
# -------------------------
# Assumes these objects exist in the notebook already:
# - FirePatchSeqDataset
# - make_loader
#
# If you restarted the notebook, you must re-run the training definitions cell first.

# Pull paths from cfg
TRAIN_CSV = cfg.get("TRAIN_CSV")
VAL_CSV   = cfg.get("VAL_CSV")
TEST_CSV  = cfg.get("TEST_CSV")

# Make loaders (train=False, no sampler)
_, val_loader = make_loader(VAL_CSV, train=False)
_, test_loader = make_loader(TEST_CSV, train=False)

# -------------------------
# 5) HELPERS: RUN PREDICTIONS + METRICS
# -------------------------
@torch.no_grad()
def predict_loader(loader):
    probs_all = []
    y_all = []
    for X, y in loader:
        X = X.to(device, non_blocking=True)
        logits = model(X)
        probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
        probs_all.append(probs)
        y_all.append(y.detach().cpu().numpy().reshape(-1))
    probs_all = np.concatenate(probs_all, axis=0)
    y_all = np.concatenate(y_all, axis=0).astype(int)
    return probs_all, y_all

def threshold_sweep(probs, y_true, thresholds):
    rows = []
    eps = 1e-9
    for thr in thresholds:
        pred = (probs >= thr).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        tn = int(((pred == 0) & (y_true == 0)).sum())
        p = (tp + eps) / (tp + fp + eps)
        r = (tp + eps) / (tp + fn + eps)
        f1 = (2 * p * r) / (p + r + eps)
        rows.append(dict(thr=float(thr), P=float(p), R=float(r), F1=float(f1), TP=tp, FP=fp, FN=fn, TN=tn))
    df = pd.DataFrame(rows).sort_values("thr")
    best = df.iloc[df["F1"].values.argmax()].to_dict()
    return df, best

def plot_prob_hist(probs, y_true, out_png):
    pos = probs[y_true == 1]
    neg = probs[y_true == 0]
    plt.figure()
    plt.hist(neg, bins=50, alpha=0.7, label="No fire")
    plt.hist(pos, bins=50, alpha=0.7, label="Fire")
    plt.xlabel("Predicted probability")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

# -------------------------
# 6) RUN VAL + TEST
# -------------------------
THR_SWEEP = cfg.get("THR_SWEEP", [round(x,2) for x in np.arange(0.10, 0.91, 0.05)])
val_probs, val_y = predict_loader(val_loader)
test_probs, test_y = predict_loader(test_loader)

pd.DataFrame({"prob": val_probs, "y": val_y}).to_csv(OUT_DIR / "val_predictions.csv", index=False)
pd.DataFrame({"prob": test_probs, "y": test_y}).to_csv(OUT_DIR / "test_predictions.csv", index=False)

val_thr_df, val_best = threshold_sweep(val_probs, val_y, THR_SWEEP)
test_thr_df, test_best = threshold_sweep(test_probs, test_y, THR_SWEEP)

val_thr_df.to_csv(OUT_DIR / "val_threshold_sweep.csv", index=False)
test_thr_df.to_csv(OUT_DIR / "test_threshold_sweep.csv", index=False)

# Use the threshold stored in best payload if present
val_best_thr = best_payload.get("val_best_thr", val_best["thr"])
val_best_thr = float(val_best_thr)

# Test at that val threshold
test_at_val = threshold_sweep(test_probs, test_y, [val_best_thr])[0].iloc[0].to_dict()

plot_prob_hist(val_probs, val_y, OUT_DIR / "val_prob_hist.png")
plot_prob_hist(test_probs, test_y, OUT_DIR / "test_prob_hist.png")

print("VAL best by sweep:", val_best)
print("VAL best thr from ckpt:", val_best_thr)
print("TEST at val best thr:", test_at_val)

# -------------------------
# 7) SAVE EXAMPLE PATCH VISUALS
# Uses the first timestep NDVI (or channel 0) just to show something interpretable
# -------------------------
def save_examples(loader, probs, y_true, out_folder, thr):
    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)

    # pick indices for TP, FP, FN, TN under threshold
    pred = (probs >= thr).astype(int)

    def pick_idx(cond, k=4):
        idx = np.where(cond)[0]
        if len(idx) == 0:
            return []
        if len(idx) <= k:
            return idx.tolist()
        return np.random.choice(idx, size=k, replace=False).tolist()

    tp_idx = pick_idx((pred == 1) & (y_true == 1))
    fp_idx = pick_idx((pred == 1) & (y_true == 0))
    fn_idx = pick_idx((pred == 0) & (y_true == 1))
    tn_idx = pick_idx((pred == 0) & (y_true == 0))

    want = [("TP", tp_idx), ("FP", fp_idx), ("FN", fn_idx), ("TN", tn_idx)]
    want_map = {}
    for name, ids in want:
        for i in ids:
            want_map[i] = name

    # Walk loader again to retrieve those patches
    cur = 0
    for X, y in loader:
        B = X.shape[0]
        for b in range(B):
            gi = cur + b
            if gi in want_map:
                Xb = X[b].numpy()  # (T,C,H,W)
                # choose first timestep, first feature channel
                img = Xb[0, 0]     # (H,W)
                plt.figure()
                plt.imshow(img)
                plt.title(f"{want_map[gi]} y={int(y[b].item())} prob={probs[gi]:.3f} thr={thr:.2f}")
                plt.axis("off")
                plt.tight_layout()
                plt.savefig(out_folder / f"{want_map[gi]}_{gi}_p{probs[gi]:.3f}.png", dpi=160)
                plt.close()
        cur += B

save_examples(val_loader, val_probs, val_y, OUT_DIR / "val_examples", val_best_thr)
save_examples(test_loader, test_probs, test_y, OUT_DIR / "test_examples", val_best_thr)

# -------------------------
# 8) WRITE A CLEAN REPORT FILE
# -------------------------
report = []
report.append(f"# Patch ConvLSTM Fire Classifier Inference Report\n")
report.append(f"Run directory: {RUN_DIR}\n")
report.append(f"Checkpoint: {CKPT_BEST}\n")
report.append(f"Best epoch: {ckpt.get('epoch')}\n\n")

report.append("## Validation\n")
report.append(f"Best F1 on validation (sweep): F1={val_best['F1']:.3f}, P={val_best['P']:.3f}, R={val_best['R']:.3f}, thr={val_best['thr']:.2f}\n")
report.append(f"Checkpoint stored validation threshold: thr={val_best_thr:.2f}\n\n")

report.append("## Test\n")
report.append(f"Metrics on test at validation threshold: F1={test_at_val['F1']:.3f}, P={test_at_val['P']:.3f}, R={test_at_val['R']:.3f}\n")
report.append(f"Confusion at thr={val_best_thr:.2f}: TP={test_at_val['TP']}, FP={test_at_val['FP']}, FN={test_at_val['FN']}, TN={test_at_val['TN']}\n\n")

report.append("## Outputs written\n")
report.append("- val_predictions.csv, test_predictions.csv\n")
report.append("- val_threshold_sweep.csv, test_threshold_sweep.csv\n")
report.append("- val_prob_hist.png, test_prob_hist.png\n")
report.append("- val_examples/, test_examples/\n")

(OUT_DIR / "report.md").write_text("".join(report))
print("Wrote:", OUT_DIR / "report.md")
print("All outputs in:", OUT_DIR)
