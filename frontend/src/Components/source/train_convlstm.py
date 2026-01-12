import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from time import time
from tqdm import tqdm



print("Torch version:", torch.__version__)
print("MPS available:", torch.backends.mps.is_available())
print("MPS built:", torch.backends.mps.is_built())


from torch_dataset_loader import (
    FirePatchDataset, Paths, Config,
    load_index, split_by_year,
    TRAIN_YEARS, VAL_YEARS,
    make_weighted_sampler, compute_pos_weight_pixelwise
)

# -----------------------------
# Device (Apple Silicon)
# -----------------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# -----------------------------
# ConvLSTM Cell
# -----------------------------
class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch, hid_ch, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_ch + hid_ch,
            4 * hid_ch,
            kernel_size,
            padding=padding
        )
        self.hid_ch = hid_ch

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)

        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

# -----------------------------
# ConvLSTM Model
# -----------------------------
class ConvLSTM(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.cell1 = ConvLSTMCell(in_ch, 32)
        self.cell2 = ConvLSTMCell(32, 16)
        self.out_conv = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape

        h1 = torch.zeros(B, 32, H, W, device=x.device)
        c1 = torch.zeros_like(h1)
        h2 = torch.zeros(B, 16, H, W, device=x.device)
        c2 = torch.zeros_like(h2)

        for t in range(T):
            h1, c1 = self.cell1(x[:, t], h1, c1)
            h2, c2 = self.cell2(h1, h2, c2)

        logits = self.out_conv(h2).squeeze(1)
        return logits  # (B, H, W)

# -----------------------------
# Training setup
# -----------------------------
paths = Paths(
    root_norm=Path("/Users/sammit/Desktop/Projects/Prometheus/data_processed_normalized"),
    root_fire=Path("/Users/sammit/Desktop/Projects/Prometheus/data_processed/fire16"),
    index_csv=Path("/Users/sammit/Desktop/Projects/Prometheus/reports/dataset/dataset_index_p32_s16.csv"),
    static_elev=Path("/Users/sammit/Desktop/Projects/Prometheus/data_processed_normalized/static/elevation_static_srtm.tif"),
    static_slope=Path("/Users/sammit/Desktop/Projects/Prometheus/data_processed_normalized/static/slope_static_srtm.tif"),
)

cfg = Config(patch_size=32)

df = load_index(paths)
df_train = split_by_year(df, TRAIN_YEARS)
df_val = split_by_year(df, VAL_YEARS)

train_ds = FirePatchDataset(df_train, paths, cfg)
val_ds = FirePatchDataset(df_val, paths, cfg)

sampler = make_weighted_sampler(df_train)
train_loader = DataLoader(train_ds, batch_size=8, sampler=sampler)
val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

pos_weight = compute_pos_weight_pixelwise(paths, df_train, cfg).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

model = ConvLSTM(in_ch=6).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

print("Running MPS warm-up...")

model.eval()
with torch.no_grad():
    X_warm, y_warm = next(iter(train_loader))
    X_warm = X_warm.to(device)
    y_warm = y_warm.to(device)
    _ = model(X_warm)

print("Warm-up complete.")


# -----------------------------
# Training loop (minimal)
# -----------------------------
EPOCHS = 10

for epoch in range(EPOCHS):
    print(f"\n===== Epoch {epoch+1}/{EPOCHS} =====")
    model.train()
    running_loss = 0.0

    for X, y in tqdm(train_loader, desc=f"Train epoch {epoch+1}"):
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        running_loss += loss.item()

    print(f"Train loss: {running_loss / len(train_loader):.4f}")

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X, y in val_loader:
            X = X.to(device)
            y = y.to(device)
            logits = model(X)
            loss = criterion(logits, y)
            val_loss += loss.item()

    print(f"Val loss: {val_loss / len(val_loader):.4f}")

\
print("Training complete.")
