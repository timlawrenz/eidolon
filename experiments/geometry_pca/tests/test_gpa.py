import numpy as np
import pytest

# We will implement these in geometry_pca.gpa
from geometry_pca.gpa import center_and_scale, get_rotation_matrix, align_single, gpa_align

def test_center_and_scale():
    shape = np.random.randn(68, 2).astype(np.float32) * 5 + np.array([100, -50])
    cs = center_and_scale(shape)
    
    # Centroid should be 0
    np.testing.assert_allclose(cs.mean(axis=0), 0, atol=1e-5)
    # Norm should be 1
    np.testing.assert_allclose(np.linalg.norm(cs), 1.0, atol=1e-5)

def test_align_single_invariance():
    # Create a base shape
    base = np.random.randn(68, 2).astype(np.float32)
    base = center_and_scale(base)
    
    # Apply translation, scale, and rotation
    theta = np.pi / 3  # 60 degrees
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ], dtype=np.float32)
    
    # modify: translate by (10, -20), scale by 4.0, rotate by theta
    modified = (base @ R) * 4.0 + np.array([10.0, -20.0], dtype=np.float32)
    
    # Align modified back to base
    aligned = align_single(modified, base)
    
    # Should perfectly match base
    np.testing.assert_allclose(aligned, base, atol=1e-4)

def test_gpa_align_dataset():
    # Create 5 variations of the same base shape
    base = center_and_scale(np.random.randn(68, 2).astype(np.float32))
    
    M = []
    for i in range(5):
        theta = np.random.uniform(-np.pi, np.pi)
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        scale = np.random.uniform(0.5, 2.0)
        trans = np.random.uniform(-50, 50, size=2)
        
        mod = (base @ R) * scale + trans
        M.append(mod)
        
    M = np.stack(M)
    
    # Run GPA
    aligned_M, mean_shape = gpa_align(M)
    
    # All aligned shapes should match the mean shape
    for i in range(5):
        np.testing.assert_allclose(aligned_M[i], mean_shape, atol=1e-4)
        
    # Mean shape should be centered and scaled
    np.testing.assert_allclose(mean_shape.mean(axis=0), 0, atol=1e-5)
    np.testing.assert_allclose(np.linalg.norm(mean_shape), 1.0, atol=1e-5)
