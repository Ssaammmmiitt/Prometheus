import torch
import numpy as np
from pathlib import Path
import rasterio
from rasterio.windows import Window
from datetime import datetime, timedelta


def predict_from_manual_input(model, data_json, config, device):
    """
    Handles a request from the frontend.
    data_json example:
    {
        "year": 2018,
        "t_steps": ["20180117", "20180202", "20180218"],
        "patch_row": 48,
        "patch_col": 64
    }
    """
    model.eval()
    patch_size = config["PATCH_SIZE"]
    vars_list = config["VARS"]
    data_root = Path(config["DATA_ROOT"])

    frames = []

    # 1. Loop through the 3 time steps provided by frontend
    for t in data_json["t_steps"]:
        channels = []
        masks = []

        for var in vars_list:
            # Build path to the specific raster
            if var in ["elevation", "slope"]:
                path = data_root / "static" / f"{var}_static_srtm.tif"
            else:
                path = (
                    data_root
                    / var
                    / str(data_json["year"])
                    / f"{var}_{data_json['year']}_{t}.tif"
                )

            # Open the large raster and crop just the 32x32 area
            with rasterio.open(path) as src:
                window = Window(
                    col_off=data_json["patch_col"],
                    row_off=data_json["patch_row"],
                    width=patch_size,
                    height=patch_size,
                )

                # Read data; boundless=True handles patches near the edge of the map
                arr = src.read(
                    1, window=window, boundless=True, fill_value=config["FILL_VALUE"]
                )

                # Create mask (1 for valid data, 0 for missing/nodata)
                mask = (arr > -1000).astype(np.float32)
                arr = np.where(arr < -1000, config["FILL_VALUE"], arr).astype(
                    np.float32
                )

                channels.append(arr)
                masks.append(mask)

        # Stack variables into (C, H, W)
        frame = np.stack(channels, axis=0)

        # Add the Missingness Mask channel if required
        if config["ADD_MISSINGNESS_MASK"]:
            combined_mask = (np.stack(masks).min(axis=0) > 0.5).astype(np.float32)
            frame = np.concatenate([frame, combined_mask[None, ...]], axis=0)

        frames.append(frame)

    # 2. Convert to Tensor: (Batch=1, Time=3, Channels=8, H=32, W=32)
    input_tensor = torch.from_numpy(np.stack(frames)).unsqueeze(0).to(device)

    # 3. Model Inference
    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda" if "cuda" in str(device) else "cpu"):
            logits = model(input_tensor)
            probability = torch.sigmoid(logits).item()

    return {
        "probability": round(probability, 4),
        "prediction": 1 if probability >= 0.5 else 0,
        "coords": [data_json["patch_row"], data_json["patch_col"]],
    }


def get_anchor_dates(input_date_str):
    # Parse input date
    input_date = datetime.strptime(str(input_date_str), "%Y%m%d")
    year = int(input_date.year)

    if year < 2018 or year > 2025:
        raise ValueError("Year must be between 2018 and 2025")

    # Start of the year
    start_of_year = datetime(year, 1, 1)

    # Generate all 16-day anchor dates for the year
    anchors = []
    current = start_of_year
    while current.year == year:
        anchors.append(current)
        current += timedelta(days=16)

    # Find nearest anchor date
    nearest = min(anchors, key=lambda d: abs((d - input_date).days))
    idx = anchors.index(nearest)

    # We need 3 preceding dates + itself → index must be >= 3
    if idx < 3:
        return [20180101, 20180117, 20180202]

    # Collect preceding three + itself
    result = anchors[idx - 3 : idx + 1]

    return [int(d.strftime("%Y%m%d")) for d in result][:3]

