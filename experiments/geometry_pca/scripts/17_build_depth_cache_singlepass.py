#!/usr/bin/env python3
"""
Phase 2: SINGLE-PASS depth cache builder (replaces 14_build_depth_cache.py's
per-mode double-read approach).

Reads each FFHQ sample's depth/seg/pose from NAS EXACTLY ONCE, computes ALL THREE
normalizations (A / A_prime / C) from that single read, and writes three local
.npy arrays. Turns 6 NAS passes (3 modes x 2 passes in the old design) into 1.

Why the old design was slow: it was NOT NAS contention — it was redundant reads.
At ~38ms/sample, one pass over 70k is ~44 min; the old code paid that 6x.

Output: data/depth_cache/ffhq_depth_{A,A_prime,C}.npy  (+ ids.json)
"""
import os, sys, time, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from geometry_pca.loader import iter_sample_ids, load_face_keypoints  # noqa
from geometry_pca import depth_encoder as de

OUT_DIR = "data/depth_cache"
LIMIT = 70000
DATASET_SIGMA = 0.15
MODES = ["A", "A_prime", "C"]


def encode_all_modes(sample_id):
    """Load depth/seg/pose ONCE, return {mode: vec} for all 3 modes (or None)."""
    try:
        depth, seg, face = de.load_depth_sample(sample_id)
    except FileNotFoundError:
        return None
    fgmask = seg > 0
    if fgmask.sum() < 50:
        return None
    h, w = depth.shape
    x0, y0, x1, y1 = de.face_bbox_px(face, h, w)
    out = {}
    for mode in MODES:
        norm = de.normalize_depth(depth, fgmask, face, mode, dataset_sigma=DATASET_SIGMA)
        if norm is None:
            return None  # if any mode fails, skip sample entirely (keep modes aligned)
        grid = de.resample_masked(norm, x0, y0, x1, y1)
        out[mode] = grid.reshape(-1).astype(np.float32)
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    # idempotent: skip if all caches exist
    if all(os.path.exists(os.path.join(OUT_DIR, f"ffhq_depth_{m}.npy")) for m in MODES):
        print("All caches already exist; nothing to do.")
        return

    t0 = time.time()
    buffers = {m: [] for m in MODES}
    ids = []
    n_seen = n_ok = 0
    for sid in iter_sample_ids(LIMIT):
        n_seen += 1
        res = encode_all_modes(sid)
        if res is not None:
            for m in MODES:
                buffers[m].append(res[m])
            ids.append(sid)
            n_ok += 1
        if n_seen % 5000 == 0:
            rate = (time.time() - t0) / n_seen * 1000
            print(f"  {n_seen} seen, {n_ok} ok ({rate:.0f}ms/sample, {time.time()-t0:.0f}s elapsed)")

    print(f"Single NAS pass done: {n_ok}/{n_seen} valid in {time.time()-t0:.1f}s")
    for m in MODES:
        arr = np.stack(buffers[m]).astype(np.float32)
        path = os.path.join(OUT_DIR, f"ffhq_depth_{m}.npy")
        np.save(path, arr)
        print(f"  saved {path}  shape={arr.shape}  {arr.nbytes/1e9:.2f}GB")
    with open(os.path.join(OUT_DIR, "ids.json"), "w") as f:
        json.dump(ids, f)
    print("Cache complete.")


if __name__ == "__main__":
    main()
