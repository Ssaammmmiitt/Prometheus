import numpy as np
import pandas as pd
import rasterio
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import json

# =========================
# CONFIG
# =========================
PROJECT_ROOT = Path("/Users/sammit/Desktop/Projects/Prometheus/").resolve()

DATA_ROOT = PROJECT_ROOT / "data_processed_normalized"
FIRE_ROOT = PROJECT_ROOT / "data_processed" / "fire16"
REPORT_ROOT = PROJECT_ROOT / "reports" / "dataset"

PATCH_SIZE = 32
STRIDE = 16  # 50% overlap
INPUT_STEPS = 3  # Number of timesteps as input
TARGET_MONTHS = [3, 4, 5]  # March, April, May

MIN_VALID_RATIO = 0.7  # At least 70% of patch must be valid (not NODATA)

TRAIN_YEARS = [2018, 2019, 2020, 2021, 2022, 2023]
VAL_YEARS = [2024]
TEST_YEARS = [2025]

NODATA_VALUE = -9999.0

# =========================
# HELPERS
# =========================
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def get_timestamps_for_year(year: int) -> list:
    """Get all available timestamps for a given year from NDVI files"""
    ndvi_year_path = DATA_ROOT / "ndvi16" / str(year)
    if not ndvi_year_path.exists():
        return []
    
    timestamps = []
    for f in sorted(ndvi_year_path.glob("*.tif")):
        # Extract date from filename: ndvi16_2018_20180101.tif
        date_str = f.stem.split('_')[-1]
        timestamps.append(date_str)
    
    return sorted(timestamps)

def extract_month(date_str: str) -> int:
    """Extract month from YYYYMMDD string"""
    return int(date_str[4:6])

def is_patch_valid(patch: np.ndarray, min_valid_ratio: float) -> bool:
    """Check if patch has enough valid (non-NODATA) pixels"""
    valid_mask = patch != NODATA_VALUE
    valid_ratio = valid_mask.sum() / patch.size
    return valid_ratio >= min_valid_ratio

def load_patch(file_path: Path, row: int, col: int, size: int) -> np.ndarray:
    """Load a patch from a raster file"""
    with rasterio.open(file_path) as src:
        patch = src.read(
            1,
            window=((row, row + size), (col, col + size))
        )
    return patch

def compute_fire_statistics(fire_patch: np.ndarray) -> dict:
    """Compute fire occurrence, density, and pixel count"""
    fire_pixels = (fire_patch == 1)
    
    return {
        'has_fire': int(fire_pixels.any()),
        'fire_pixel_count': int(fire_pixels.sum()),
        'fire_density': float(fire_pixels.sum() / fire_patch.size)
    }

# =========================
# MAIN INDEXING FUNCTION
# =========================
def create_dataset_index():
    """
    Create comprehensive dataset index with:
    - Validity filtering (70% valid pixels minimum)
    - Temporal splits (train/val/test by year)
    - Fire statistics
    - Complete spatial coverage
    """
    
    ensure_dir(REPORT_ROOT)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("="*80)
    print("DATASET INDEX GENERATION")
    print("="*80)
    print(f"Data root: {DATA_ROOT}")
    print(f"Fire root: {FIRE_ROOT}")
    print(f"Patch size: {PATCH_SIZE}x{PATCH_SIZE}")
    print(f"Stride: {STRIDE}")
    print(f"Input timesteps: {INPUT_STEPS}")
    print(f"Target months: {TARGET_MONTHS}")
    print(f"Min valid ratio: {MIN_VALID_RATIO}")
    print(f"Train years: {TRAIN_YEARS}")
    print(f"Val years: {VAL_YEARS}")
    print(f"Test years: {TEST_YEARS}")
    
    # Get reference raster dimensions
    ref_file = next((DATA_ROOT / "ndvi16" / str(TRAIN_YEARS[0])).glob("*.tif"))
    with rasterio.open(ref_file) as src:
        height, width = src.shape
        profile = src.profile
    
    print(f"\nRaster dimensions: {height}H x {width}W")
    
    # Generate all possible patch coordinates
    patch_coords = []
    for row in range(0, height - PATCH_SIZE + 1, STRIDE):
        for col in range(0, width - PATCH_SIZE + 1, STRIDE):
            patch_coords.append((row, col))
    
    print(f"Total patch positions: {len(patch_coords)}")
    
    # Collect all available years
    all_years = sorted(set(TRAIN_YEARS + VAL_YEARS + TEST_YEARS))
    
    # Build temporal index
    print("\n" + "="*80)
    print("BUILDING TEMPORAL INDEX")
    print("="*80)
    
    timestamps_by_year = {}
    for year in all_years:
        timestamps = get_timestamps_for_year(year)
        timestamps_by_year[year] = timestamps
        print(f"  {year}: {len(timestamps)} timestamps")
        
        if len(timestamps) == 0:
            print(f"    ⚠️  WARNING: No data for year {year}")
    
    # Build sample index
    samples = []
    
    print("\n" + "="*80)
    print("GENERATING SAMPLES")
    print("="*80)
    
    stats_by_year = {}
    
    for year in tqdm(all_years, desc="Years"):
        timestamps = timestamps_by_year[year]
        
        if len(timestamps) < INPUT_STEPS + 1:
            print(f"\n  Skipping {year}: only {len(timestamps)} timestamps (need {INPUT_STEPS + 1})")
            continue
        
        year_stats = {
            'total_windows': 0,
            'valid_windows': 0,
            'fire_windows': 0,
            'total_patches_checked': 0,
            'valid_patches': 0,
            'fire_patches': 0
        }
        
        # For each valid temporal window
        for t_idx in range(INPUT_STEPS, len(timestamps)):
            year_stats['total_windows'] += 1
            
            # Input timestamps
            input_dates = timestamps[t_idx - INPUT_STEPS:t_idx]
            target_date = timestamps[t_idx]
            
            # Filter by target month
            target_month = extract_month(target_date)
            if target_month not in TARGET_MONTHS:
                continue
            
            year_stats['valid_windows'] += 1
            
            # Check if fire data exists for target
            fire_path = FIRE_ROOT / str(year) / f"fire16_{year}_{target_date}.tif"
            if not fire_path.exists():
                print(f"\n  ⚠️  Missing fire data: {fire_path.name}")
                continue
            
            # Load full fire raster for this timestep
            with rasterio.open(fire_path) as src:
                fire_full = src.read(1)
            
            # Check if this window has any fires
            window_has_fire = (fire_full == 1).any()
            if window_has_fire:
                year_stats['fire_windows'] += 1
            
            # For each patch position
            for row, col in patch_coords:
                year_stats['total_patches_checked'] += 1
                
                # Check validity across all input timesteps
                valid_patch = True
                
                for date in input_dates:
                    ndvi_path = DATA_ROOT / "ndvi16" / str(year) / f"ndvi16_{year}_{date}.tif"
                    
                    if not ndvi_path.exists():
                        valid_patch = False
                        break
                    
                    ndvi_patch = load_patch(ndvi_path, row, col, PATCH_SIZE)
                    
                    if not is_patch_valid(ndvi_patch, MIN_VALID_RATIO):
                        valid_patch = False
                        break
                
                if not valid_patch:
                    continue
                
                # Extract fire patch
                fire_patch = fire_full[row:row+PATCH_SIZE, col:col+PATCH_SIZE]
                
                # Check fire patch validity
                if not is_patch_valid(fire_patch, MIN_VALID_RATIO):
                    continue
                
                year_stats['valid_patches'] += 1
                
                # Compute fire statistics
                fire_stats = compute_fire_statistics(fire_patch)
                
                if fire_stats['has_fire']:
                    year_stats['fire_patches'] += 1
                
                # Add sample
                samples.append({
                    'year': year,
                    't1': input_dates[0],
                    't2': input_dates[1],
                    't3': input_dates[2],
                    't4': target_date,
                    'target_month': target_month,
                    'patch_row': row,
                    'patch_col': col,
                    **fire_stats
                })
        
        stats_by_year[year] = year_stats
        
        # Print year summary
        print(f"\n  {year}:")
        print(f"    Valid windows: {year_stats['valid_windows']}/{year_stats['total_windows']}")
        print(f"    Valid patches: {year_stats['valid_patches']:,}")
        print(f"    Fire patches: {year_stats['fire_patches']:,} "
              f"({year_stats['fire_patches']/max(year_stats['valid_patches'],1)*100:.2f}%)")
    
    # Convert to DataFrame
    df = pd.DataFrame(samples)
    
    if len(df) == 0:
        raise ValueError("No valid samples generated! Check data paths and validity thresholds.")
    
    # Assign splits
    def assign_split(year):
        if year in TRAIN_YEARS:
            return 'train'
        elif year in VAL_YEARS:
            return 'val'
        elif year in TEST_YEARS:
            return 'test'
        else:
            return 'unused'
    
    df['split'] = df['year'].apply(assign_split)
    
    # Save index
    output_csv = REPORT_ROOT / f"dataset_index_p{PATCH_SIZE}_s{STRIDE}_{timestamp}.csv"
    df.to_csv(output_csv, index=False)
    
    # Generate summary report
    print("\n" + "="*80)
    print("DATASET SUMMARY")
    print("="*80)
    print(f"Total samples: {len(df):,}")
    
    print("\n" + "-"*80)
    print("BY SPLIT:")
    print("-"*80)
    for split in ['train', 'val', 'test']:
        split_df = df[df.split == split]
        if len(split_df) == 0:
            continue
        
        fire_count = split_df.has_fire.sum()
        fire_ratio = fire_count / len(split_df)
        
        avg_density = split_df[split_df.has_fire == 1].fire_density.mean() if fire_count > 0 else 0
        
        print(f"\n{split.upper()}:")
        print(f"  Total samples: {len(split_df):,}")
        print(f"  Fire samples: {fire_count:,} ({fire_ratio:.2%})")
        print(f"  No-fire samples: {len(split_df) - fire_count:,}")
        print(f"  Avg fire density (fire patches): {avg_density:.4f}")
    
    print("\n" + "-"*80)
    print("BY YEAR:")
    print("-"*80)
    for year in sorted(df.year.unique()):
        year_df = df[df.year == year]
        fire_count = year_df.has_fire.sum()
        fire_ratio = fire_count / len(year_df)
        split = year_df.iloc[0]['split']
        
        print(f"{year} ({split:5s}): {len(year_df):6,} samples, "
              f"{fire_count:5,} fire ({fire_ratio:6.2%})")
    
    print("\n" + "-"*80)
    print("BY TARGET MONTH:")
    print("-"*80)
    for month in sorted(df.target_month.unique()):
        month_df = df[df.target_month == month]
        fire_count = month_df.has_fire.sum()
        fire_ratio = fire_count / len(month_df)
        month_name = ['', '', '', 'March', 'April', 'May'][month]
        
        print(f"{month_name:8s}: {len(month_df):6,} samples, "
              f"{fire_count:5,} fire ({fire_ratio:6.2%})")
    
    # Spatial coverage check
    unique_patches = df[['patch_row', 'patch_col']].drop_duplicates()
    coverage_ratio = len(unique_patches) / len(patch_coords)
    
    print("\n" + "-"*80)
    print("SPATIAL COVERAGE:")
    print("-"*80)
    print(f"Unique patch locations: {len(unique_patches):,} / {len(patch_coords):,}")
    print(f"Coverage: {coverage_ratio:.1%}")
    
    if coverage_ratio < 0.5:
        print("  ⚠️  WARNING: Low spatial coverage! Consider lowering min_valid_ratio.")
    elif coverage_ratio > 0.8:
        print("  ✓ Good spatial coverage!")
    
    # Save detailed statistics
    summary = {
        'generation_time': timestamp,
        'config': {
            'patch_size': PATCH_SIZE,
            'stride': STRIDE,
            'input_steps': INPUT_STEPS,
            'target_months': TARGET_MONTHS,
            'min_valid_ratio': MIN_VALID_RATIO,
            'train_years': TRAIN_YEARS,
            'val_years': VAL_YEARS,
            'test_years': TEST_YEARS
        },
        'summary': {
            'total_samples': int(len(df)),
            'train_samples': int((df.split == 'train').sum()),
            'val_samples': int((df.split == 'val').sum()),
            'test_samples': int((df.split == 'test').sum()),
            'fire_samples': int(df.has_fire.sum()),
            'fire_ratio': float(df.has_fire.mean()),
            'spatial_coverage': float(coverage_ratio),
            'unique_patches': int(len(unique_patches))
        },
        'by_year': {
            int(year): {
                'samples': int(len(year_df)),
                'fire_samples': int(year_df.has_fire.sum()),
                'fire_ratio': float(year_df.has_fire.mean())
            }
            for year, year_df in df.groupby('year')
        },
        'stats_by_year': stats_by_year
    }
    
    summary_json = REPORT_ROOT / f"dataset_summary_{timestamp}.json"
    with open(summary_json, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*80)
    print("FILES SAVED:")
    print("="*80)
    print(f"Index CSV: {output_csv}")
    print(f"Summary JSON: {summary_json}")
    
    print("\n" + "="*80)
    print("✓ DATASET INDEX COMPLETE")
    print("="*80)
    
    return df

# =========================
# RUN
# =========================
if __name__ == "__main__":
    df = create_dataset_index()