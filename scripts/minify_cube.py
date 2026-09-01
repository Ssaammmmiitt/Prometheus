import zarr
import numcodecs
import pandas as pd
import numpy as np
import shutil
import os

print("Copying metadata...")
src_feat = zarr.open('data/cube/features_daily.zarr', mode='r')

times = src_feat['time'][:]
dates = pd.to_datetime(times, unit='D')
mask_2026 = dates.year == 2026
idx_2026 = np.where(mask_2026)[0]

print(f"Found {len(idx_2026)} days in 2026")

def slice_zarr(src_path, dst_path):
    if os.path.exists(dst_path):
        shutil.rmtree(dst_path)
    src = zarr.open(src_path, mode='r')
    dst = zarr.open(dst_path, mode='w')
    
    dst.attrs.update(src.attrs)
    
    for key in src.keys():
        arr = src[key]
        if 'time' in key or (arr.shape and arr.shape[0] == len(times)):
            print(f"Slicing {key} {arr.shape}...")
            if len(arr.shape) == 3:
                sliced_data = arr.get_orthogonal_selection((idx_2026, slice(None), slice(None)))
            elif len(arr.shape) == 1:
                sliced_data = arr.get_orthogonal_selection((idx_2026,))
            else:
                sliced_data = arr[:] # fallback
            dst.create_array(key, shape=sliced_data.shape, chunks=arr.chunks, dtype=arr.dtype)
            dst[key][:] = sliced_data
            dst[key].attrs.update(arr.attrs)
        else:
            print(f"Copying {key} {arr.shape}...")
            dst.create_array(key, shape=arr.shape, chunks=arr.chunks, dtype=arr.dtype)
            dst[key][:] = arr[:]
            dst[key].attrs.update(arr.attrs)
    print(f"Finished {dst_path}")

slice_zarr('data/cube/features_daily.zarr', 'data/cube/features_2026.zarr')
slice_zarr('data/cube/fire_daily.zarr', 'data/cube/fire_2026.zarr')
