import os
import tempfile
import numpy as np
import pytest

from geometry_pca.gpa import gpa_align
from geometry_pca.fit import fit_encoder, save_encoder, load_encoder
from geometry_pca.encode import encode_pose

def test_fit_and_io():
    N, K = 100, 10
    # synthetic shapes
    M = np.random.randn(N, 68, 2).astype(np.float32)
    aligned_M, gpa_mean = gpa_align(M)
    
    encoder = fit_encoder(aligned_M, gpa_mean, k=K)
    
    assert encoder["components"].shape == (K, 136)
    assert encoder["pca_mean"].shape == (136,)
    assert encoder["whiten_mu"].shape == (K,)
    assert encoder["whiten_sigma"].shape == (K,)
    assert encoder["gpa_mean"].shape == (68, 2)
    assert encoder["explained_variance_ratio"].shape == (K,)
    
    # Check monotonicity of explained variance
    evr = encoder["explained_variance_ratio"]
    assert np.all(np.diff(evr) <= 1e-5) # monotonically decreasing
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "enc.npz")
        save_encoder(encoder, path)
        enc_loaded = load_encoder(path)
        
        for k, v in encoder.items():
            np.testing.assert_allclose(v, enc_loaded[k], atol=1e-5)

def test_inference_encode():
    np.random.seed(42)  # determinism: avoid flaky whitening-tolerance failures
    N, K = 50, 5
    M = np.random.randn(N, 68, 2).astype(np.float32)
    # Fit the encoder on the dataset
    aligned_M, gpa_mean = gpa_align(M)
    encoder = fit_encoder(aligned_M, gpa_mean, k=K)
    
    # Test encode on a single raw pose (M[0] is unaligned)
    raw_pose = M[0]
    z_g = encode_pose(raw_pose, encoder)
    
    assert z_g.shape == (K,)
    assert np.all(np.isfinite(z_g))
    
    # Whitening guarantee: if we encode all training samples, 
    # the resulting z_g vectors must have zero mean and unit variance per component.
    Z = np.array([encode_pose(m, encoder) for m in M])
    
    np.testing.assert_allclose(Z.mean(axis=0), 0, atol=1e-2)
    np.testing.assert_allclose(Z.std(axis=0), 1, atol=0.05)
