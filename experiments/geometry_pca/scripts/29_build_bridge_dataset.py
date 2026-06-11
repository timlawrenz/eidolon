#!/usr/bin/env python3
"""
Phase 3: FFHQ Bridge Dataset Builder.

Iterates the ~70k FFHQ dataset to collect dinov3_cls tokens and compute the
associated physical targets (z_g, z_a_xy) to train the DINOv3 bridge.

Reads from NAS:
  - dinov3_cls.npy (1024-d)
  - pose.npy (for z_g)
Reuses the in-RAM normal cache to derive z_a_xy without NAS hits.

Outputs:
  data/bridge_dataset.npz: X_dino (N,1024), Y_zg (N,50), Y_za (N,50)
"""
import os, sys, time, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from geometry_pca.loader import iter_sample_ids
from geometry_pca.constants import STRATUM_ROOT, FACE_SLICE
from geometry_pca.zg_inference import encode_zg
from geometry_pca.normal_encoder import derive_variant

LIMIT = 70000
CONF_THRESH = 0.5


def load_encoders():
    zg_enc = dict(np.load("output/encoder_production.npz"))
    za_enc = dict(np.load("output/encoder_za_xy.npz"))
    return zg_enc, za_enc


def process_sample(sid, zg_enc, za_enc, normal_grid, normal_R):
    base = os.path.join(STRATUM_ROOT, sid)
    try:
        dino = np.load(os.path.join(base, "dinov3_cls.npy")).astype(np.float32)
        pose = np.load(os.path.join(base, "pose.npy")).astype(np.float32)
    except (FileNotFoundError, OSError, ValueError):
        return None, None, None

    face = pose[FACE_SLICE]
    if face[:, 2].mean() < CONF_THRESH:
        return None, None, None

    # z_g (geometry)
    zg = encode_zg(face[:, :2], zg_enc)

    # z_a (surface, xy variant). normal_grid is already masked/resampled to 64x64.
    vec = derive_variant(normal_grid, normal_R, "xy")
    comps = za_enc["components"]
    pmean = za_enc["pca_mean"]
    wmu = za_enc["whiten_mu"]
    wsig = za_enc["whiten_sigma"]
    za = (((vec - pmean) @ comps.T) - wmu) / wsig

    return dino, zg, za


def main():
    print("Loading encoders and normal cache...")
    zg_enc, za_enc = load_encoders()
    try:
        normal_raw = np.load("data/normal_cache/ffhq_normal_raw.npy", mmap_mode="r")
        normal_rots = np.load("data/normal_cache/rotations.npy", mmap_mode="r")
        cache_ids = json.load(open("data/normal_cache/ids.json"))
        cache_idx = {sid: i for i, sid in enumerate(cache_ids)}
    except FileNotFoundError as e:
        print(f"FATAL: Cache missing: {e}")
        sys.exit(1)

    print("Iterating FFHQ...")
    X_dino, Y_zg, Y_za = [], [], []
    n_seen, n_ok = 0, 0
    t0 = time.time()

    for sid in iter_sample_ids(LIMIT):
        n_seen += 1
        if sid not in cache_idx:
            continue
        idx = cache_idx[sid]
        
        # Pull grid and R from the memmapped cache
        grid = np.asarray(normal_raw[idx])
        R = np.asarray(normal_rots[idx])
        
        dino, zg, za = process_sample(sid, zg_enc, za_enc, grid, R)
        if dino is not None:
            X_dino.append(dino)
            Y_zg.append(zg)
            Y_za.append(za)
            n_ok += 1
            
        if n_seen % 5000 == 0:
            rate = (time.time() - t0) / n_seen * 1000
            print(f"  {n_seen} seen, {n_ok} ok ({rate:.0f}ms/sample)")

    print(f"\nDone. Collected {n_ok} samples.")
    X_dino = np.stack(X_dino)
    Y_zg = np.stack(Y_zg)
    Y_za = np.stack(Y_za)

    out = "data/bridge_dataset.npz"
    np.savez_compressed(out, X_dino=X_dino, Y_zg=Y_zg, Y_za=Y_za)
    print(f"Saved {out} ({X_dino.nbytes/1e9:.2f} GB dino features)")


if __name__ == "__main__":
    main()
