#!/usr/bin/env python3
"""
Phase 2 (Depth): Fit 3 separate depth encoders (z_d) on FFHQ.
Uses IncrementalPCA — prefers the NAS depth cache (data/ symlink) when available
for speed.

Normalization modes swept:
  - 'A': Masked per-image z-score
  - 'A_prime': Center-only, scaled by fixed dataset sigma
  - 'C': Anatomical anchor (nose=0, scaled by inter-ocular pixel distance)
"""
import os, sys, json, time
import numpy as np
from sklearn.decomposition import IncrementalPCA

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from geometry_pca.loader import iter_sample_ids
from geometry_pca.depth_encoder import encode_depth_sample, OUT_RES

OUT = "output"
LIMIT = 70000
BATCH_SIZE = 5000
K = 50
DATASET_SIGMA = 0.15


def _fit_from_array(pca, X):
    """Fit IPCA and compute whitening stats from an in-memory array X (N, D)."""
    t0 = time.time()
    N = len(X)
    for start in range(0, N, BATCH_SIZE):
        end = min(start + BATCH_SIZE, N)
        pca.partial_fit(X[start:end])
        print(f"  partial_fit {end}/{N} ({time.time()-t0:.0f}s)")
    print(f"  fit done in {time.time()-t0:.1f}s")
    
    print("  Computing whitening stats...")
    whiten_sum = np.zeros(K, dtype=np.float64)
    whiten_sq_sum = np.zeros(K, dtype=np.float64)
    for start in range(0, N, BATCH_SIZE):
        end = min(start + BATCH_SIZE, N)
        scores = pca.transform(X[start:end])
        whiten_sum += np.sum(scores, axis=0)
        whiten_sq_sum += np.sum(scores**2, axis=0)
    whiten_mu = whiten_sum / N
    whiten_var = (whiten_sq_sum / N) - (whiten_mu**2)
    whiten_sigma = np.sqrt(np.maximum(whiten_var, 1e-8))
    return whiten_mu, whiten_sigma


def _fit_from_nas(pca, mode, limit):
    """Fit IPCA from NAS, two-pass: partial_fit then whitening."""
    print(f"  No cache — reading from NAS (slow)...")
    batch = []
    processed_count = 0
    t0 = time.time()
    for sid in iter_sample_ids(limit):
        vec = encode_depth_sample(sid, mode, dataset_sigma=DATASET_SIGMA)
        if vec is not None:
            batch.append(vec)
            processed_count += 1
        if len(batch) >= BATCH_SIZE:
            M = np.stack(batch)
            pca.partial_fit(M)
            batch = []
            print(f"  partial_fit up to {processed_count}...")
    if len(batch) > K:
        M = np.stack(batch)
        pca.partial_fit(M)
    print(f"  first pass done in {time.time()-t0:.1f}s ({processed_count} samples)")
    
    # Second pass: whitening
    print("  Computing whitening stats...")
    whiten_sum = np.zeros(K, dtype=np.float64)
    whiten_sq_sum = np.zeros(K, dtype=np.float64)
    w_count = 0
    batch = []
    for sid in iter_sample_ids(limit):
        vec = encode_depth_sample(sid, mode, dataset_sigma=DATASET_SIGMA)
        if vec is not None:
            batch.append(vec)
        if len(batch) >= BATCH_SIZE:
            M = np.stack(batch)
            scores = pca.transform(M)
            whiten_sum += np.sum(scores, axis=0)
            whiten_sq_sum += np.sum(scores**2, axis=0)
            w_count += len(scores)
            batch = []
    if batch:
        M = np.stack(batch)
        scores = pca.transform(M)
        whiten_sum += np.sum(scores, axis=0)
        whiten_sq_sum += np.sum(scores**2, axis=0)
        w_count += len(scores)
    whiten_mu = whiten_sum / w_count
    whiten_var = (whiten_sq_sum / w_count) - (whiten_mu**2)
    whiten_sigma = np.sqrt(np.maximum(whiten_var, 1e-8))
    return whiten_mu, whiten_sigma, processed_count


def fit_incremental_encoder(mode: str, limit: int) -> dict:
    pca = IncrementalPCA(n_components=K, batch_size=BATCH_SIZE)
    print(f"\n--- Fitting Mode {mode} ---")
    
    cache_path = os.path.join("data", "depth_cache", f"ffhq_depth_{mode}.npy")
    
    if os.path.exists(cache_path):
        print(f"  Using cache: {cache_path}")
        cache = np.load(cache_path, mmap_mode='r')
        whiten_mu, whiten_sigma = _fit_from_array(pca, cache)
        n = len(cache)
    else:
        whiten_mu, whiten_sigma, n = _fit_from_nas(pca, mode, limit)
    
    return {
        "components": pca.components_.astype(np.float32),
        "explained_variance_ratio": pca.explained_variance_ratio_.astype(np.float32),
        "pca_mean": pca.mean_.astype(np.float32),
        "whiten_mu": whiten_mu.astype(np.float32),
        "whiten_sigma": whiten_sigma.astype(np.float32),
        "mode": mode,
        "n_samples": n
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    modes = ["A", "A_prime", "C"]
    summary = {}
    for m in modes:
        enc = fit_incremental_encoder(m, LIMIT)
        out_path = os.path.join(OUT, f"encoder_zd_{m}.npz")
        np.savez_compressed(out_path, **enc)
        evr = enc["explained_variance_ratio"]
        summary[m] = {"n_samples": enc["n_samples"], "retained_variance": float(np.sum(evr))}
        print(f"-> Saved {out_path} (Retained Var: {summary[m]['retained_variance']:.4f})")
    with open(os.path.join(OUT, "zd_fit_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\nDone.")
    for m, s in summary.items():
        print(f"  {m}: {s}")


if __name__ == "__main__":
    main()
