#!/usr/bin/env python3
"""
Phase 2 pre-step: build a LOCAL cache of processed 64x64 depth vectors.
Reads each FFHQ depth sample from the NAS ONCE, processes it (mask/crop/resample),
and saves the flattened vector to a memory-mapped .npy file.

After this completes, IncrementalPCA reads from the cache — turning an
I/O-bound 10-hour job into a 2-minute fit.

Usage:
  python scripts/14_build_depth_cache.py --mode A
  python scripts/14_build_depth_cache.py --mode A_prime
  python scripts/14_build_depth_cache.py --mode C
"""
import os, sys, time, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from geometry_pca.loader import iter_sample_ids
from geometry_pca.depth_encoder import encode_depth_sample, OUT_RES

OUT_DIR = "data/depth_cache"
LIMIT = 70000
DATASET_SIGMA = 0.15


def build_cache(mode, limit):
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"ffhq_depth_{mode}.npy")
    
    if os.path.exists(out_path):
        print(f"Cache already exists: {out_path}")
        return out_path
        
    vec_dim = OUT_RES * OUT_RES
    # Use memmapped array for safe writing
    t0 = time.time()
    valid_count = 0
    
    # First, count valid samples (needed for pre-allocation)
    print(f"Counting valid samples (mode={mode})...")
    for sid in iter_sample_ids(limit):
        vec = encode_depth_sample(sid, mode, dataset_sigma=DATASET_SIGMA)
        if vec is not None:
            valid_count += 1
            if valid_count % 10000 == 0:
                print(f"  counted {valid_count}...")
                
    print(f"Allocating {valid_count} x {vec_dim} array ({valid_count * vec_dim * 4 / 1e9:.1f} GB)")
    
    # Allocate memmap
    mem = np.memmap(out_path, dtype=np.float32, mode='w+', shape=(valid_count, vec_dim))
    
    # Second pass: write
    idx = 0
    for sid in iter_sample_ids(limit):
        vec = encode_depth_sample(sid, mode, dataset_sigma=DATASET_SIGMA)
        if vec is not None:
            mem[idx] = vec
            idx += 1
            if idx % 10000 == 0:
                print(f"  cached {idx}/{valid_count} ({time.time()-t0:.0f}s)")
    mem.flush()
    del mem
    print(f"Cache built: {out_path} ({valid_count} samples, {time.time()-t0:.1f}s)")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["A", "A_prime", "C"])
    args = ap.parse_args()
    build_cache(args.mode, LIMIT)
