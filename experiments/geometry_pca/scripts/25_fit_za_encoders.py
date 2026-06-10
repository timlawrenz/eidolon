#!/usr/bin/env python3
"""
Phase 2b (Normals): Fit 4 z_a encoders from the normal cache.

Loads the cached 64x64x3 raw normal grids + per-sample rotation matrices into
RAM, derives each of the 4 variants (raw/xy/rot/rot_xy), fits IncrementalPCA
(k=50) on each, and saves the whitened encoders.

PREREQUISITE: data/normal_cache/ must exist (built by scripts/24).
"""
import os, sys, json, time
import numpy as np
from sklearn.decomposition import IncrementalPCA

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geometry_pca.normal_encoder import derive_variant, variant_dim

OUT = "output"
VARIANTS = ["raw", "xy", "rot", "rot_xy"]
K = 50
BATCH_SIZE = 5000

CACHE_RAW = "data/normal_cache/ffhq_normal_raw.npy"
CACHE_ROT = "data/normal_cache/rotations.npy"


def load_cache():
    """Load raw grids (N,64,64,3) and rotations (N,3,3) into RAM."""
    raw = np.load(CACHE_RAW)      # (N,64,64,3) float32
    rots = np.load(CACHE_ROT)     # (N,3,3) float32
    N = len(raw)
    print(f"Loaded cache: {N} samples  raw={raw.nbytes/1e9:.1f}GB  rots={rots.nbytes/1e9:.3f}GB")
    return raw, rots, N


def derive_matrix(raw, rots, variant):
    """Derive (N,D) matrix for one variant from cached raw grids + rotations."""
    D = variant_dim(variant)  # 8192 for *_xy, 12288 for 3-channel
    X = np.empty((len(raw), D), dtype=np.float32)
    for i in range(len(raw)):
        X[i] = derive_variant(raw[i], rots[i], variant)
    return X


def fit_encoder(X):
    """Fit IncrementalPCA k=50 + compute whitening stats. Returns encoder dict."""
    pca = IncrementalPCA(n_components=K, batch_size=BATCH_SIZE)
    N = len(X)
    t0 = time.time()
    for start in range(0, N, BATCH_SIZE):
        end = min(start + BATCH_SIZE, N)
        pca.partial_fit(X[start:end])
        print(f"  partial_fit {end}/{N} ({time.time()-t0:.0f}s)")
    print(f"  fit done in {time.time()-t0:.1f}s")

    # Whitening: mean/std of scores on the fit data
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

    return {
        "components": pca.components_.astype(np.float32),
        "explained_variance_ratio": pca.explained_variance_ratio_.astype(np.float32),
        "pca_mean": pca.mean_.astype(np.float32),
        "whiten_mu": whiten_mu.astype(np.float32),
        "whiten_sigma": whiten_sigma.astype(np.float32),
        "variant": VARIANTS[0],  # overwritten below
        "n_samples": N,
    }


def main():
    if not os.path.exists(CACHE_RAW):
        print(f"ERROR: Cache not found at {CACHE_RAW}. Run scripts/24 first.")
        sys.exit(1)

    os.makedirs(OUT, exist_ok=True)
    raw, rots, N = load_cache()
    summary = {}

    for variant in VARIANTS:
        print(f"\n--- Fitting Variant '{variant}' ---")
        X = derive_matrix(raw, rots, variant)
        print(f"  Matrix shape: {X.shape}  ({X.nbytes/1e9:.1f}GB in RAM)")

        enc = fit_encoder(X)
        enc["variant"] = variant
        out_path = os.path.join(OUT, f"encoder_za_{variant}.npz")
        np.savez_compressed(out_path, **enc)
        evr = enc["explained_variance_ratio"]
        summary[variant] = {"n_samples": N, "retained_variance": float(np.sum(evr)),
                            "dim": X.shape[1]}
        print(f"  -> Saved {out_path} (Retained Var: {summary[variant]['retained_variance']:.4f})")

    with open(os.path.join(OUT, "za_fit_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\nDone.")
    for v, s in summary.items():
        print(f"  {v:10s}: dim={s['dim']}  retained_var={s['retained_variance']:.4f}")


if __name__ == "__main__":
    main()
