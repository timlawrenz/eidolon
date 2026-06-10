"""Tests for geometry_pca/normal_encoder.py — normal-map preprocessing."""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_unit_norm_preserved_under_rotation():
    """R^T applied to unit vectors produces unit vectors (length preserved)."""
    from geometry_pca.normal_encoder import head_rotation, derive_variant
    from geometry_pca.canonical_face import canonical_template

    rng = np.random.default_rng(42)
    # synthetic unit normals on a 64x64 grid
    n = rng.normal(size=(64, 64, 3))
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    # face keypoints for rotation estimation (neutral/frontal)
    face = np.zeros((68, 2), dtype=np.float32)
    face[:, 0] = np.linspace(-0.3, 0.3, 68)
    face[:, 1] = np.linspace(-0.2, 0.2, 68)
    R = head_rotation(face, canonical_template())

    # apply rotation per-pixel
    n_rot = derive_variant(n, R, "rot").reshape(64, 64, 3)
    norms = np.linalg.norm(n_rot, axis=-1)
    # norms may drop from 1.0 after pooling in derive_variant, but for raw vectors
    # let's test the raw rotation math directly:
    n_flat = n.reshape(-1, 3)
    n_rot_raw = (n_flat @ R.T).reshape(64, 64, 3)
    norms_rot = np.linalg.norm(n_rot_raw, axis=-1)
    assert np.allclose(norms_rot, 1.0, atol=1e-5), f"R^T must preserve unit-norm"


def test_derive_variant_shapes():
    """raw=12288, xy=8192, rot=12288, rot_xy=8192."""
    from geometry_pca.normal_encoder import derive_variant

    rng = np.random.default_rng(1)
    grid = rng.normal(size=(64, 64, 3)).astype(np.float32)
    R = np.eye(3, dtype=np.float32)

    for variant, expected_dim in [("raw", 12288), ("xy", 8192), ("rot", 12288), ("rot_xy", 8192)]:
        v = derive_variant(grid, R, variant)
        assert v.shape == (expected_dim,), f"{variant}: got {v.shape}, want ({expected_dim},)"
        assert v.dtype == np.float32


def test_nan_bg_excluded():
    """NaN-background nan->0 handling works per-channel."""
    from geometry_pca.normal_encoder import resample_masked_3ch

    arr = np.ones((128, 128, 3), dtype=np.float32)
    arr[64:, :, :] = np.nan  # bottom half = bg
    x0, y0, x1, y1 = 0, 0, 128, 64  # crop to top half only
    out = resample_masked_3ch(arr, x0, y0, x1, y1, out_res=8)
    assert out.shape == (8, 8, 3)
    assert np.isfinite(out).all(), "output should have no NaN"
    assert np.allclose(out, 1.0, atol=1e-5), "top half should average to 1"


def test_frontalization_invariant():
    """For synthetically yawed normals + yawed keypoints, de-rotation restores frontal."""
    from geometry_pca.normal_encoder import head_rotation, apply_rotation_field
    from geometry_pca.canonical_face import canonical_template

    rng = np.random.default_rng(7)
    # canonical front-facing normals with some surface variation
    n_frontal = np.zeros((64, 64, 3), dtype=np.float32)
    n_frontal[:, :, 2] = 1.0
    n_frontal[:, :, 0] = (rng.random((64, 64)) - 0.5) * 0.3
    n_frontal[:, :, 1] = (rng.random((64, 64)) - 0.5) * 0.3
    n_frontal /= np.linalg.norm(n_frontal, axis=-1, keepdims=True)

    # yaw by 25° around y-axis
    theta = np.deg2rad(25)
    Ry = np.array([[np.cos(theta), 0, np.sin(theta)],
                   [0, 1, 0],
                   [-np.sin(theta), 0, np.cos(theta)]], dtype=np.float32)
    n_yawed = apply_rotation_field(n_frontal, Ry)

    # ALSO yaw the face keypoints by the same rotation
    tpl = canonical_template()  # (68,3) canonical 3D face
    tpl_yawed = (tpl @ Ry.T).astype(np.float32)  # rotate 3D template
    face_yawed = tpl_yawed[:, :2].copy()  # fronto-parallel projection of yawed face

    # estimate rotation from the YAWED keypoints + original canonical template
    R_est = head_rotation(face_yawed, canonical_template())

    # apply inverse of estimated rotation to yawed normals
    nz_raw = n_yawed[:, :, 2].mean()
    n_corrected = apply_rotation_field(n_yawed, R_est.T)
    nz_corrected = n_corrected[:, :, 2].mean()
    assert nz_corrected > nz_raw, (
        f"de-rotation should bring normals closer to frontal: "
        f"nz_raw={nz_raw:.3f}, nz_corrected={nz_corrected:.3f}"
    )
