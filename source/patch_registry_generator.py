import numpy as np
import pandas as pd
import rasterio
from pathlib import Path
from affine import Affine

# =========================
# CONFIG
# =========================
PROJECT_ROOT = Path("/Users/sammit/Desktop/Projects/Prometheus/").resolve()

DATA_ROOT = PROJECT_ROOT / "data_processed_normalized"
MASK_PATH = PROJECT_ROOT / "data_raw" / "mask" / "nepal_mask_1km_roiAligned.tif"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "spatial"

PATCH_SIZE = 32  # Non-overlapping 32x32 km patches
MIN_VALID_RATIO = 0.3  # At least 30% of patch must be in Nepal

NODATA_VALUE = -9999.0

# Optional: Region classification (can skip if not needed)
CLASSIFY_REGIONS = False  # Set to True if you want terai/hill/mountain labels

# Elevation thresholds for region classification (meters)
TERAI_MAX = 300
HILL_MAX = 3000

# =========================
# HELPERS
# =========================
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def pixel_to_latlon(row: int, col: int, transform: Affine) -> tuple:
    """Convert pixel coordinates to lat/lon (center of pixel)"""
    lon, lat = transform * (col + 0.5, row + 0.5)
    return lat, lon

def classify_region(elevation: float) -> str:
    """Classify region based on elevation"""
    if elevation < TERAI_MAX:
        return "terai"
    elif elevation < HILL_MAX:
        return "hill"
    else:
        return "mountain"

# =========================
# MAIN
# =========================
def create_patch_registry():
    """
    Create a fixed, non-overlapping patch grid for:
    - Operational inference
    - Frontend visualization
    - Spatial analysis
    """
    
    # Use global CLASSIFY_REGIONS or set locally
    classify_regions = CLASSIFY_REGIONS
    
    ensure_dir(OUTPUT_DIR)
    
    print("="*80)
    print("PATCH REGISTRY CREATION (FIXED GRID)")
    print("="*80)
    print(f"Patch size: {PATCH_SIZE}x{PATCH_SIZE} pixels (no overlap)")
    print(f"Min valid ratio: {MIN_VALID_RATIO}")
    print(f"Region classification: {'Enabled' if classify_regions else 'Disabled'}")
    
    # Load reference raster for dimensions and georeferencing
    ref_file = next((DATA_ROOT / "ndvi16" / "2020").glob("*.tif"))
    
    with rasterio.open(ref_file) as src:
        height, width = src.shape
        transform = src.transform
        crs = src.crs
    
    print(f"\nRaster dimensions: {height}H x {width}W")
    print(f"CRS: {crs}")
    print(f"Pixel size: ~{abs(transform.a):.6f}° (~1 km)")
    
    # Load Nepal mask
    with rasterio.open(MASK_PATH) as src:
        nepal_mask = src.read(1)
    
    # Load elevation if region classification is enabled
    elevation_data = None
    if classify_regions:
        elev_path = DATA_ROOT / "static" / "elevation_static_srtm.tif"
        if elev_path.exists():
            with rasterio.open(elev_path) as src:
                # Elevation is normalized [0,1], denormalize it
                elevation_norm = src.read(1)
                # From your normalization: min=62m, max=8364m
                elevation_data = elevation_norm * (8364 - 62) + 62
                elevation_data[elevation_norm == NODATA_VALUE] = np.nan
            print("Loaded elevation data for region classification")
        else:
            print("⚠️  Elevation file not found, skipping region classification")
            classify_regions = False
    
    # Generate non-overlapping patch grid
    patches = []
    patch_id = 0
    
    print(f"\n{'='*80}")
    print("GENERATING PATCH GRID")
    print(f"{'='*80}")
    
    for row in range(0, height - PATCH_SIZE + 1, PATCH_SIZE):  # Non-overlapping
        for col in range(0, width - PATCH_SIZE + 1, PATCH_SIZE):
            # Extract mask patch
            mask_patch = nepal_mask[row:row+PATCH_SIZE, col:col+PATCH_SIZE]
            
            # Check validity (how much of patch is in Nepal?)
            valid_mask = mask_patch == 1
            valid_ratio = valid_mask.sum() / mask_patch.size
            
            if valid_ratio < MIN_VALID_RATIO:
                continue  # Skip patches mostly outside Nepal
            
            # Calculate center coordinates
            center_row = row + PATCH_SIZE // 2
            center_col = col + PATCH_SIZE // 2
            center_lat, center_lon = pixel_to_latlon(center_row, center_col, transform)
            
            # Calculate corner coordinates (for frontend polygon rendering)
            top_left_lat, top_left_lon = pixel_to_latlon(row, col, transform)
            bottom_right_lat, bottom_right_lon = pixel_to_latlon(
                row + PATCH_SIZE, col + PATCH_SIZE, transform
            )
            
            # Region classification (optional)
            region_type = None
            avg_elevation = None
            
            if classify_regions and elevation_data is not None:
                elev_patch = elevation_data[row:row+PATCH_SIZE, col:col+PATCH_SIZE]
                valid_elev = elev_patch[~np.isnan(elev_patch)]
                
                if len(valid_elev) > 0:
                    avg_elevation = float(np.mean(valid_elev))
                    region_type = classify_region(avg_elevation)
            
            patches.append({
                'patch_id': f"P{patch_id:04d}",
                'patch_row': row,
                'patch_col': col,
                'center_lat': round(center_lat, 6),
                'center_lon': round(center_lon, 6),
                'top_left_lat': round(top_left_lat, 6),
                'top_left_lon': round(top_left_lon, 6),
                'bottom_right_lat': round(bottom_right_lat, 6),
                'bottom_right_lon': round(bottom_right_lon, 6),
                'valid_ratio': round(valid_ratio, 4),
                'avg_elevation_m': round(avg_elevation, 1) if avg_elevation else None,
                'region_type': region_type
            })
            
            patch_id += 1
    
    # Convert to DataFrame
    df = pd.DataFrame(patches)
    
    print(f"\n{'='*80}")
    print("PATCH GRID SUMMARY")
    print(f"{'='*80}")
    print(f"Total patches: {len(df)}")
    print(f"Coverage area: ~{len(df) * 32 * 32:,} km²")
    
    if classify_regions and 'region_type' in df.columns and df.region_type.notna().any():
        print(f"\n{'='*80}")
        print("BY REGION:")
        print(f"{'='*80}")
        region_counts = df.region_type.value_counts()
        for region, count in region_counts.items():
            pct = count / len(df) * 100
            print(f"  {region.capitalize():10s}: {count:3d} patches ({pct:5.1f}%)")
    
    # Geographic extent
    print(f"\n{'='*80}")
    print("GEOGRAPHIC EXTENT:")
    print(f"{'='*80}")
    print(f"Latitude range:  {df.center_lat.min():.4f}° to {df.center_lat.max():.4f}°")
    print(f"Longitude range: {df.center_lon.min():.4f}° to {df.center_lon.max():.4f}°")
    
    # Elevation range (if available)
    if 'avg_elevation_m' in df.columns and df.avg_elevation_m.notna().any():
        print(f"Elevation range: {df.avg_elevation_m.min():.0f}m to {df.avg_elevation_m.max():.0f}m")
    
    # Save to CSV
    output_csv = OUTPUT_DIR / "patch_registry_fixed_grid.csv"
    df.to_csv(output_csv, index=False)
    
    print(f"\n{'='*80}")
    print("SAVED:")
    print(f"{'='*80}")
    print(f"Registry CSV: {output_csv}")
    
    # Create GeoJSON for frontend (optional but useful)
    try:
        import json
        
        features = []
        for _, row in df.iterrows():
            # Create polygon coordinates (rectangle)
            coordinates = [[
                [row.top_left_lon, row.top_left_lat],
                [row.bottom_right_lon, row.top_left_lat],
                [row.bottom_right_lon, row.bottom_right_lat],
                [row.top_left_lon, row.bottom_right_lat],
                [row.top_left_lon, row.top_left_lat]  # Close polygon
            ]]
            
            properties = {
                'patch_id': row.patch_id,
                'center_lat': row.center_lat,
                'center_lon': row.center_lon,
                'valid_ratio': row.valid_ratio
            }
            
            if row.region_type:
                properties['region_type'] = row.region_type
            if row.avg_elevation_m:
                properties['avg_elevation_m'] = row.avg_elevation_m
            
            features.append({
                'type': 'Feature',
                'geometry': {
                    'type': 'Polygon',
                    'coordinates': coordinates
                },
                'properties': properties
            })
        
        geojson = {
            'type': 'FeatureCollection',
            'features': features
        }
        
        output_geojson = OUTPUT_DIR / "patch_grid.geojson"
        with open(output_geojson, 'w') as f:
            json.dump(geojson, f, indent=2)
        
        print(f"GeoJSON: {output_geojson}")
        print("  (Can be loaded directly in frontend maps)")
        
    except Exception as e:
        print(f"⚠️  Could not generate GeoJSON: {e}")
    
    print(f"\n{'='*80}")
    print("✓ PATCH REGISTRY COMPLETE")
    print(f"{'='*80}")
    
    # Sample entries
    print("\nSample patches:")
    print(df.head(5).to_string(index=False))
    
    return df

# =========================
# RUN
# =========================
if __name__ == "__main__":
    df = create_patch_registry()