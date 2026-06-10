"""Test for geometry_pca/za_inference.py — z_a inference helper."""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_encode_za_shape_and_finite():
    """encode_za returns a (K,) float32 vector given valid synthetic inputs."""
    from geometry_pca.za_inference import encode_za

    # synthetic 64x64 normal map, all foreground, somewhat face-like normals
    rng = np.random.default_rng(99)
    normal = np.zeros((64, 64, 3), dtype=np.float32)
    normal[:, :, 2] = 1.0  # camera-facing
    normal[:, :, 0] = rng.random((64, 64)).astype(np.float32) * 0.2 - 0.1
    normal[:, :, 1] = rng.random((64, 64)).astype(np.float32) * 0.2 - 0.1
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)

    seg = np.ones((64, 64), dtype=np.uint8)
    face = np.zeros((68, 3), dtype=np.float32)
    face[:, 0] = np.linspace(-0.3, 0.3, 68)
    face[:, 1] = np.linspace(-0.2, 0.2, 68)
    face[:, 2] = 1.0

    # toy encoder: components(50,D), pca_mean(D,), etc.
    D = 12288
    enc = {
        "components": np.eye(50, D, dtype=np.float32),
        "pca_mean": np.zeros(D, dtype=np.float32),
        "whiten_mu": np.zeros(50, dtype=np.float32),
        "whiten_sigma": np.ones(50, dtype=np.float32),
        "variant": "raw",
    }
    vec = encode_za(normal, seg, face, enc, variant="raw")
    assert vec.shape == (50,), f"got {vec.shape}"
    assert vec.dtype == np.float32
    assert np.isfinite(vec).all(), "non-finite values"


def test_whiten_sanity_on_realistic_scores():
    """Whitening a batch with its own mu/sigma -> mean~0, std~1."""
    from geometry_pca.zd_inference import whiten_scores
    rng = np.random.default_rng(123)
    Z = rng.normal(loc=3.0, scale=2.0, size=(200, 50))
    mu = Z.mean(axis=0); sigma = Z.std(axis=0)
    Zw = whiten_scores(Z, mu, sigma)
    assert np.allclose(Zw.mean(axis=0), 0, atol=1e-6)
    assert np.allclose(Zw.std(axis=0), 1, atol=1e-6)
