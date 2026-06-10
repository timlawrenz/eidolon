#!/usr/bin/env python3
"""
Phase 2b: SINGLE-PASS normal cache builder.

Reads each FFHQ sample's normal/seg/pose from NAS EXACTLY ONCE, computes the
64×64×3 masked resample + per-sample head rotation R, and writes:
  data/normal_cache/ffhq_normal_raw.npy  (~3.4 GB float32, 69839 x 64 x 64 x 3)
  data/normal_cache/rotations.npy        (69839 x 3 x 3 float32)
  data/normal_cache/ids.json

All 4 representation variants (raw/xy/rot/rot_xy) can be derived in-memory from
these two files at fit time — rotation commutes with pooling, so we never need
to touch the NAS again.

Idempotent: skips if both .npy files already exist.
Runs as ONE NAS pass; expects ~2-3h wall time at ~100ms/sample.
"""
import os, sys, time, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geometry_pca.loader import iter_sample_ids
from geometry_pca.constants import FACE_SLICE
from geometry_pca.depth_encoder import face_bbox_px
from geometry_pca.normal_encoder import resample_masked_3ch, head_rotation
from geometry_pca.canonical_face import canonical_template

OUT_DIR = "data/normal_cache"
LIMIT = 70000
OUT_RES = 64

CANONICAL_TPL = canonical_template()


def process_one(sample_id):
    """Load normal+seg+pose ONCE, return (grid_64, R) or (None, None) on failure."""
    try:
        normal, seg, face_r3 = load_normal_sample_local(sample_id)
    except (FileNotFoundError, OSError):
        return None, None

    # foreground: any seg class > 0, plus normal magnitude > 0.1 (robust bg rejection)
    fgmask = (seg > 0).astype(np.float32)
    mag = np.linalg.norm(normal, axis=-1)
    fgmask *= (mag > 0.1).astype(np.float32)
    if fgmask.sum() < 50:
        return None, None

    h, w = normal.shape[:2]
    face_2d = face_r3[:, :2]  # (68,2)
    x0, y0, x1, y1 = face_bbox_px(face_r3, h, w)

    # resample normal map (3-channel, NaN-aware)
    # Set background pixels to zero so resample_masked_3ch ignores them
    normal_bg = normal.copy()
    normal_bg[fgmask < 0.5] = 0.0
    grid = resample_masked_3ch(normal_bg, x0, y0, x1, y1, out_res=OUT_RES)

    # estimate head rotation from 68 keypoints (in [-1,1] space as stored)
    R = head_rotation(face_2d, CANONICAL_TPL)

    return grid, R


def load_normal_sample_local(sample_id):
    """Load a single sample (same as normal_encoder.load_normal_sample but inlined)."""
    from geometry_pca.constants import STRATUM_ROOT
    base = os.path.join(STRATUM_ROOT, sample_id)
    normal = np.load(os.path.join(base, "normal.npy")).astype(np.float32)
    seg = np.load(os.path.join(base, "seg.npy"))
    pose = np.load(os.path.join(base, "pose.npy")).astype(np.float32)
    face = pose[FACE_SLICE]
    return normal, seg, face


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    raw_path = os.path.join(OUT_DIR, "ffhq_normal_raw.npy")
    rot_path = os.path.join(OUT_DIR, "rotations.npy")
    if os.path.exists(raw_path) and os.path.exists(rot_path):
        print("Both cache files already exist; nothing to do.")
        return

    t0 = time.time()
    grids = []
    rotlist = []
    ids_out = []
    n_seen, n_ok = 0, 0

    for sid in iter_sample_ids(LIMIT):
        n_seen += 1
        grid, R = process_one(sid)
        if grid is not None:
            grids.append(grid.reshape(-1))      # flatten to 64*64*3 for stacking
            rotlist.append(R)
            ids_out.append(sid)
            n_ok += 1
        if n_seen % 5000 == 0:
            rate = (time.time() - t0) / n_seen * 1000
            print(f"  {n_seen} seen, {n_ok} ok ({rate:.0f}ms/sample, {time.time()-t0:.0f}s elapsed)")

    elapsed = time.time() - t0
    print(f"Single NAS pass done: {n_ok}/{n_seen} valid in {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # Stack and save
    grid_arr = np.stack(grids).reshape(n_ok, OUT_RES, OUT_RES, 3).astype(np.float32)
    rot_arr = np.stack(rotlist).astype(np.float32)
    np.save(raw_path, grid_arr)
    np.save(rot_path, rot_arr)
    print(f"  saved {raw_path}  shape={grid_arr.shape}  {grid_arr.nbytes/1e9:.2f}GB")
    print(f"  saved {rot_path}  shape={rot_arr.shape}")

    with open(os.path.join(OUT_DIR, "ids.json"), "w") as f:
        json.dump(ids_out, f)
    print("Cache complete.")


if __name__ == "__main__":
    main()
