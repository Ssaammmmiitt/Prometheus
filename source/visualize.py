import os
import rasterio
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ================= CONFIGURATION =================
# Set the root directory where your folders (ndvi16, etc.) are located.
# Use '.' for the current directory.
ROOT_DIR = '/Users/sammit/Desktop/Projects/Prometheus/data_processed' 
OUTPUT_REPORT_CSV = '/Users/sammit/Desktop/Projects/Prometheus/reports/alignment_report.csv'
OUTPUT_PLOT_IMAGE = '/Users/sammit/Desktop/Projects/Prometheus/reports/raster_visualization.png'

# List of folders to include in the scan
TARGET_FOLDERS = ['ndvi16', 'precip16', 'rh16', 'vpd16', 'fire16', 'temp16', 'static']
# =================================================

def check_alignment_and_visualize():
    raster_data = []
    base_path = Path(ROOT_DIR)
    
    print(f"--- Starting Scan in: {base_path.resolve()} ---")

    # 1. Collect Metadata for all TIF files
    # -------------------------------------
    files_found = []
    
    # Walk through specific target folders or all if needed
    for folder_name in TARGET_FOLDERS:
        folder_path = base_path / folder_name
        if not folder_path.exists():
            print(f"Warning: Folder not found: {folder_name}")
            continue
            
        # Recursive search for .tif files
        for file_path in folder_path.rglob('*.tif'):
            files_found.append(file_path)

    if not files_found:
        print("No .tif files found!")
        return

    print(f"Found {len(files_found)} raster files. Checking alignment...")

    # 2. Alignment Check Logic
    # -------------------------------------
    # We take the first file as the 'Master Reference' to compare others against.
    # Ideally, use a static file as reference if available.
    reference_file = files_found[0]
    
    # Try to find a static file to use as reference, otherwise use the first one
    static_files = [f for f in files_found if 'static' in str(f)]
    if static_files:
        reference_file = static_files[0]

    ref_meta = {}
    with rasterio.open(reference_file) as src:
        ref_meta = {
            'crs': src.crs,
            'transform': src.transform,
            'width': src.width,
            'height': src.height,
            'count': src.count
        }
    
    print(f"Reference File: {reference_file.name}")
    print(f"Reference CRS: {ref_meta['crs']}, Shape: {ref_meta['width']}x{ref_meta['height']}")

    aligned_count = 0
    misaligned_count = 0
    samples_for_plot = {} # Store one sample per category

    for file_path in files_found:
        try:
            with rasterio.open(file_path) as src:
                # Check Metadata
                is_aligned = True
                misalignment_reason = []

                # Check CRS
                if src.crs != ref_meta['crs']:
                    is_aligned = False
                    misalignment_reason.append(f"CRS mismatch ({src.crs})")
                
                # Check Dimensions
                if src.width != ref_meta['width'] or src.height != ref_meta['height']:
                    is_aligned = False
                    misalignment_reason.append(f"Shape mismatch ({src.width}x{src.height})")
                
                # Check Transform (Location/Pixel Size)
                # We use a small tolerance for float comparisons in transform
                if not np.allclose(src.transform, ref_meta['transform'], atol=1e-06):
                    is_aligned = False
                    misalignment_reason.append("Transform/Extent mismatch")

                status = "Aligned" if is_aligned else "Misaligned"
                if is_aligned:
                    aligned_count += 1
                else:
                    misaligned_count += 1
                    print(f"MISALIGNMENT: {file_path.name} -> {', '.join(misalignment_reason)}")

                # Log data for CSV
                raster_data.append({
                    'File Path': str(file_path),
                    'File Name': file_path.name,
                    'Category': file_path.parent.parent.name if 'static' not in str(file_path) else 'static',
                    'CRS': str(src.crs),
                    'Shape': f"{src.width}x{src.height}",
                    'Status': status,
                    'Reason': "; ".join(misalignment_reason)
                })

                # Save sample for plotting (one per main folder)
                category = file_path.parent.parent.name if 'static' not in str(file_path) else 'static'
                if category not in samples_for_plot:
                    # Read data for plot (masked to handle nodata)
                    samples_for_plot[category] = {
                        'data': src.read(1, masked=True),
                        'name': file_path.name
                    }

        except Exception as e:
            print(f"Error reading {file_path.name}: {e}")
            raster_data.append({
                'File Path': str(file_path),
                'File Name': file_path.name,
                'Status': 'Error',
                'Reason': str(e)
            })

    # 3. Generate CSV Report
    # -------------------------------------
    df = pd.DataFrame(raster_data)
    df.to_csv(OUTPUT_REPORT_CSV, index=False)
    print(f"\n--- Report Generated: {OUTPUT_REPORT_CSV} ---")
    print(f"Total Files: {len(files_found)}")
    print(f"Aligned: {aligned_count}")
    print(f"Misaligned: {misaligned_count}")

    # 4. Visualization (One sample per category)
    # -------------------------------------
    if samples_for_plot:
        print(f"\n--- Generating Visualization: {OUTPUT_PLOT_IMAGE} ---")
        num_plots = len(samples_for_plot)
        cols = 3
        rows = (num_plots // cols) + (1 if num_plots % cols > 0 else 0)
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        axes = axes.flatten()
        
        for idx, (category, sample) in enumerate(samples_for_plot.items()):
            ax = axes[idx]
            im = ax.imshow(sample['data'], cmap='viridis', interpolation='none')
            ax.set_title(f"Category: {category}\n({sample['name']})", fontsize=10)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.axis('off')

        # Hide empty subplots
        for i in range(num_plots, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        plt.savefig(OUTPUT_PLOT_IMAGE, dpi=150)
        print("Visualization saved.")

if __name__ == "__main__":
    check_alignment_and_visualize()