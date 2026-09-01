import zarr
import sys

def inspect_zarr(path):
    print(f"Inspecting {path}")
    try:
        z = zarr.open(path, mode='r')
        for key in z.keys():
            arr = z[key]
            print(f"{key}: shape={arr.shape}, chunks={arr.chunks}, dtype={arr.dtype}")
    except Exception as e:
        print(f"Error: {e}")

inspect_zarr('data/cube/features_daily.zarr')
