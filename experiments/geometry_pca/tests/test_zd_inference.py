"""Tests for z_d depth inference helper — whitening sanity."""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_whiten_sanity():
    """A batch of (K,) Z-scores, whitened, should have per-component mean~0, std~1."""
    from geometry_pca.zd_inference import whiten_scores

    rng = np.random.default_rng(42)
    Z = rng.normal(loc=5.0, scale=3.0, size=(500, 50))
    mu = Z.mean(axis=0)
    sigma = Z.std(axis=0)

    Z_w = whiten_scores(Z, mu, sigma)
    assert Z_w.shape == Z.shape
    # after whitening with the same mu/sigma, per-component stats should be near 0/1
    mu_w = Z_w.mean(axis=0)
    sigma_w = Z_w.std(axis=0)
    assert np.allclose(mu_w, 0.0, atol=1e-6), f"whitened mean not ~0: {np.abs(mu_w).max():.2e}"
    assert np.allclose(sigma_w, 1.0, atol=1e-6), f"whitened std not ~1: {np.abs(sigma_w-1.0).max():.2e}"

def test_encode_zd_shape_and_finite():
    """encode_zd should return a (K,) float32 vector with finite values given valid inputs."""
    from geometry_pca.zd_inference import encode_zd

    # synthetic "depth" 64x64, seg all-1 (full foreground), pose with face at center
    depth = np.ones((64, 64), dtype=np.float32) * 0.5
    seg = np.ones((64, 64), dtype=np.uint8)
    face = np.zeros((68, 3), dtype=np.float32)
    face[:,0] = np.linspace(-0.3, 0.3, 68)  # spread x a bit
    face[:,1] = np.linspace(-0.2, 0.2, 68)  # spread y
    face[:,2] = 1.0

    # toy encoder: components(50,4096), pca_mean(4096,), whiten_mu(50,), whiten_sigma(50,)
    toy = {
        "components": np.eye(50, 4096, dtype=np.float32),
        "pca_mean": np.zeros(4096, dtype=np.float32),
        "whiten_mu": np.zeros(50, dtype=np.float32),
        "whiten_sigma": np.ones(50, dtype=np.float32),
        "mode": "C",
    }
    vec = encode_zd(depth, seg, face, toy, mode="C", dataset_sigma=0.15)
    assert vec.shape == (50,), f"got {vec.shape}"
    assert vec.dtype == np.float32
    assert np.isfinite(vec).all(), "non-finite values in output"
