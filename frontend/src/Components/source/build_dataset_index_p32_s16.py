# from pathlib import Path
# import rasterio
# import pandas as pd
# import numpy as np
# import json
# from datetime import datetime

# # =========================
# # CONFIG
# # =========================
# ROOT = Path("/Users/sammit/Desktop/Projects/Prometheus").resolve()

# DATA_NORM = ROOT / "data_processed_normalized"
# DATA_FIRE = ROOT / "data_processed" / "fire16"

# REPORT_DIR = ROOT / "reports" / "dataset"
# REPORT_DIR.mkdir(parents=True, exist_ok=True)

# PATCH_SIZE = 32
# STRIDE = 16
# INPUT_STEPS = 3

# TARGET_MONTHS = {3, 4, 5}  # March–May

# YEARS = list(range(2018, 2026))

# INDEX_CSV = REPORT_DIR / "dataset_index_p32_s16.csv"
# SUMMARY_JSON = REPORT_DIR / "dataset_summary_p32_s16.json"
# LOG_FILE = REPORT_DIR / "dataset_log_p32_s16.txt"

# # =========================
# # HELPERS
# # =========================
# def log(msg):
#     print(msg)
#     with open(LOG_FILE, "a", encoding="utf-8") as f:
#         f.write(msg + "\n")

# def parse_date_from_name(name):
#     # expects *_YYYYMMDD.tif
#     return name.split("_")[-1].replace(".tif", "")

# def list_sorted_dates(var, year):
#     folder = DATA_NORM / var / str(year)
#     if not folder.exists():
#         return []
#     files = sorted(folder.glob("*.tif"))
#     return [(parse_date_from_name(f.name), f) for f in files]

# def month_from_yyyymmdd(s):
#     return int(s[4:6])

# def raster_shape(sample_path):
#     with rasterio.open(sample_path) as src:
#         return src.height, src.width

# # =========================
# # MAIN
# # =========================
# def main():
#     log("Dataset index creation started")
#     log(f"Patch size: {PATCH_SIZE}, Stride: {STRIDE}")
#     log(f"Input timesteps: {INPUT_STEPS}")
#     log(f"Target months (t4): {sorted(TARGET_MONTHS)}")
#     log(f"Years scanned: {YEARS}")

#     index_rows = []

#     total_samples = 0
#     fire_patches = 0
#     non_fire_patches = 0
#     per_year_counts = {}

#     for year in YEARS:
#         ndvi_dates = list_sorted_dates("ndvi16", year)
#         if len(ndvi_dates) < INPUT_STEPS + 1:
#             continue

#         per_year_counts[year] = 0

#         # get grid shape once
#         _, sample_ndvi = ndvi_dates[0]
#         H, W = raster_shape(sample_ndvi)

#         # valid patch start positions
#         rows = list(range(0, H - PATCH_SIZE + 1, STRIDE))
#         cols = list(range(0, W - PATCH_SIZE + 1, STRIDE))

#         log(f"Year {year}: {len(ndvi_dates)} timesteps, grid {H}×{W}, patches {len(rows)*len(cols)} per window")

#         for i in range(INPUT_STEPS, len(ndvi_dates)):
#             t1 = ndvi_dates[i - 3][0]
#             t2 = ndvi_dates[i - 2][0]
#             t3 = ndvi_dates[i - 1][0]
#             t4 = ndvi_dates[i][0]

#             if month_from_yyyymmdd(t4) not in TARGET_MONTHS:
#                 continue

#             fire_path = DATA_FIRE / str(year) / f"fire16_{year}_{t4}.tif"
#             if not fire_path.exists():
#                 continue

#             with rasterio.open(fire_path) as fire_src:
#                 fire_arr = fire_src.read(1)

#             for r in rows:
#                 for c in cols:
#                     patch = fire_arr[r:r+PATCH_SIZE, c:c+PATCH_SIZE]
#                     has_fire = int(np.any(patch == 1))

#                     index_rows.append({
#                         "year": year,
#                         "t1": t1,
#                         "t2": t2,
#                         "t3": t3,
#                         "t4": t4,
#                         "patch_row": r,
#                         "patch_col": c,
#                         "has_fire": has_fire
#                     })

#                     total_samples += 1
#                     per_year_counts[year] += 1
#                     if has_fire:
#                         fire_patches += 1
#                     else:
#                         non_fire_patches += 1

#     # =========================
#     # SAVE OUTPUTS
#     # =========================
#     df = pd.DataFrame(index_rows)
#     df.to_csv(INDEX_CSV, index=False)

#     summary = {
#         "patch_size": PATCH_SIZE,
#         "stride": STRIDE,
#         "input_timesteps": INPUT_STEPS,
#         "target_months": sorted(TARGET_MONTHS),
#         "years_included": YEARS,
#         "total_samples": total_samples,
#         "fire_patches": fire_patches,
#         "non_fire_patches": non_fire_patches,
#         "fire_ratio": fire_patches / max(total_samples, 1),
#         "samples_per_year": per_year_counts
#     }

#     with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
#         json.dump(summary, f, indent=2)

#     log("Dataset index creation completed")
#     log(f"Total samples: {total_samples}")
#     log(f"Fire patches: {fire_patches}")
#     log(f"Non-fire patches: {non_fire_patches}")
#     log(f"Fire ratio: {summary['fire_ratio']:.6f}")
#     log(f"Index CSV written to: {INDEX_CSV}")
#     log(f"Summary JSON written to: {SUMMARY_JSON}")

#     print("\nFINAL SUMMARY")
#     print("-------------")
#     print(f"Total samples: {total_samples}")
#     print(f"Fire patches: {fire_patches}")
#     print(f"Non-fire patches: {non_fire_patches}")
#     print(f"Fire ratio: {summary['fire_ratio']:.6f}")
#     print(f"Index file: {INDEX_CSV}")

# if __name__ == "__main__":
#     main()



from pathlib import Path
import rasterio
import pandas as pd
import numpy as np
import json

# =========================
# CONFIG
# =========================
ROOT = Path('/Users/b_karki/Desktop/Prometheus').resolve()

DATA_NORM = ROOT / "data_processed_normalized"
DATA_FIRE = ROOT / "data_processed" / "fire16"

REPORT_DIR = ROOT / "reports" / "dataset"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

PATCH_SIZE = 32
STRIDE = 32
INPUT_STEPS = 3

TARGET_MONTHS = {3, 4, 5}  # March–May
YEARS = list(range(2018, 2026))

INDEX_CSV = REPORT_DIR / "dataset_index_p32_s16.csv"
SUMMARY_JSON = REPORT_DIR / "dataset_summary_p32_s16.json"
LOG_FILE = REPORT_DIR / "dataset_log_p32_s16.txt"

FEATURE_VARS = ["ndvi16", "temp16", "precip16", "rh16", "vpd16"]
VALID_RATIO_THRESHOLD = 0.5


# =========================
# HELPERS
# =========================
def log(msg):
    print(msg)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

def parse_date_from_name(name: str) -> str:
    # expects *_YYYYMMDD.tif or *_YYYYMMDD.tiff
    stem = Path(name).stem  # removes last suffix
    return stem.split("_")[-1]

def list_sorted_dates(var, year):
    folder = DATA_NORM / var / str(year)
    if not folder.exists():
        return []
    files = sorted([p for p in folder.glob("*.tif*") if p.is_file()])
    return [(parse_date_from_name(f.name), f) for f in files]

def month_from_yyyymmdd(s):
    return int(s[4:6])

def raster_shape(sample_path):
    with rasterio.open(sample_path) as src:
        return src.height, src.width

def feature_paths_exist(year: int, date_strs):
    """
    Ensure all feature rasters exist for each required date.
    date_strs: iterable of YYYYMMDD (t1,t2,t3)
    """
    for var in FEATURE_VARS:
        for d in date_strs:
            p = DATA_NORM / var / str(year) / f"{var}_{year}_{d}.tif"
            if not p.exists():
                return False, str(p)
    return True, ""

# =========================
# MAIN
# =========================
def main():
    # reset log file each run for clarity
    LOG_FILE.write_text("", encoding="utf-8")

    log("Dataset index creation started")
    log(f"Patch size: {PATCH_SIZE}, Stride: {STRIDE}")
    log(f"Input timesteps: {INPUT_STEPS}")
    log(f"Target months (t4): {sorted(TARGET_MONTHS)}")
    log(f"Years scanned: {YEARS}")
    log(f"Feature vars validated: {FEATURE_VARS}")

    index_rows = []

    total_samples = 0
    fire_patches = 0
    non_fire_patches = 0
    per_year_counts = {}
    years_used = []

    missing_feature_skips = 0
    missing_fire_skips = 0

    for year in YEARS:
        ndvi_dates = list_sorted_dates("ndvi16", year)
        if len(ndvi_dates) < INPUT_STEPS + 1:
            continue

        per_year_counts[year] = 0

        # grid shape from first NDVI
        _, sample_ndvi = ndvi_dates[0]
        H, W = raster_shape(sample_ndvi)

        rows = list(range(0, H - PATCH_SIZE + 1, STRIDE))
        cols = list(range(0, W - PATCH_SIZE + 1, STRIDE))

        log(f"Year {year}: {len(ndvi_dates)} timesteps, grid {H}×{W}, patches {len(rows)*len(cols)} per window")

        year_had_any = False

        for i in range(INPUT_STEPS, len(ndvi_dates)):
            t1 = ndvi_dates[i - 3][0]
            t2 = ndvi_dates[i - 2][0]
            t3 = ndvi_dates[i - 1][0]
            t4 = ndvi_dates[i][0]

            if month_from_yyyymmdd(t4) not in TARGET_MONTHS:
                continue

            # ensure features exist for t1-t3 including VPD
            ok, missing_path = feature_paths_exist(year, [t1, t2, t3])
            if not ok:
                missing_feature_skips += 1
                continue

            fire_path = DATA_FIRE / str(year) / f"fire16_{year}_{t4}.tif"
            if not fire_path.exists():
                missing_fire_skips += 1
                continue

            with rasterio.open(fire_path) as fire_src:
                fire_arr = fire_src.read(1)
                
            ndvi_path = DATA_NORM / "ndvi16" / str(year) / f"ndvi16_{year}_{t4}.tif"
            with rasterio.open(ndvi_path) as ndvi_src:
                ndvi_arr = ndvi_src.read(1)


            for r in rows:
                for c in cols:

                    # --- NDVI validity check ---
                    ndvi_patch = ndvi_arr[r:r+PATCH_SIZE, c:c+PATCH_SIZE]
                    valid_ratio = np.mean(ndvi_patch != -9999)

                    if valid_ratio < VALID_RATIO_THRESHOLD:
                        continue

                    # --- Fire label ---
                    fire_patch = fire_arr[r:r+PATCH_SIZE, c:c+PATCH_SIZE]
                    has_fire = int(np.any(fire_patch == 1))

                    index_rows.append({
                        "year": year,
                        "t1": t1,
                        "t2": t2,
                        "t3": t3,
                        "t4": t4,
                        "patch_row": r,
                        "patch_col": c,
                        "has_fire": has_fire,
                        "valid_ratio": float(valid_ratio)
                    })

                    total_samples += 1
                    per_year_counts[year] += 1
                    year_had_any = True

                    if has_fire:
                        fire_patches += 1
                    else:
                        non_fire_patches += 1


        if year_had_any:
            years_used.append(year)

    df = pd.DataFrame(index_rows)
    df.to_csv(INDEX_CSV, index=False)

    summary = {
        "patch_size": PATCH_SIZE,
        "stride": STRIDE,
        "input_timesteps": INPUT_STEPS,
        "target_months": sorted(TARGET_MONTHS),
        "years_scanned": YEARS,
        "years_used": years_used,
        "total_samples": total_samples,
        "fire_patches": fire_patches,
        "non_fire_patches": non_fire_patches,
        "fire_ratio": fire_patches / max(total_samples, 1),
        "samples_per_year": {str(k): v for k, v in per_year_counts.items() if v > 0},
        "skips_missing_feature_triplet": missing_feature_skips,
        "skips_missing_fire_t4": missing_fire_skips,
        "features_validated": FEATURE_VARS
    }

    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    log("Dataset index creation completed")
    log(f"Total samples: {total_samples}")
    log(f"Fire patches: {fire_patches}")
    log(f"Non-fire patches: {non_fire_patches}")
    log(f"Fire ratio: {summary['fire_ratio']:.6f}")
    log(f"Skipped windows due to missing features (incl VPD): {missing_feature_skips}")
    log(f"Skipped windows due to missing fire t4: {missing_fire_skips}")
    log(f"Index CSV written to: {INDEX_CSV}")
    log(f"Summary JSON written to: {SUMMARY_JSON}")

    print("\nFINAL SUMMARY")
    print("-------------")
    print(f"Total samples: {total_samples}")
    print(f"Fire patches: {fire_patches}")
    print(f"Non-fire patches: {non_fire_patches}")
    print(f"Fire ratio: {summary['fire_ratio']:.6f}")
    print(f"Index file: {INDEX_CSV}")

if __name__ == "__main__":
    main()
