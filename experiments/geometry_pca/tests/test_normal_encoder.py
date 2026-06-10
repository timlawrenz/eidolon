"""Tests for geometry_pca/normal_encoder.py — normal-map preprocessing.

REVIEWER-DRIVEN REWRITE: the original tests were self-consistent (generated
keypoints FROM the canonical template in +Y-up space and bypassed
derive_variant's internal rotation), so they could not catch:
  (a) the unflipped-template convention bug (R=diag(1,-1,-1) for frontal faces)
  (b) derive_variant applying R forward instead of R^T (doubling pose)
These tests build observations in IMAGE convention (+Y down) and exercise the
public derive_variant path.
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _image_frame_keypoints(R=None):
    """Project the canonical template into IMAGE convention (+Y down),
    optionally pre-rotated by R (in image frame). This is what pose.npy-like
    observations look like."""
    from geometry_pca.canonical_face import canonical_template
    tpl = canonical_template().copy()      # +Y up
    tpl[:, 1] *= -1.0                      # -> image frame (+Y down)
    if R is not None:
        tpl = tpl @ R.T                    # rotate in image frame
    return tpl[:, :2].astype(np.float32)   # orthographic projection


def test_frontal_face_yields_identity_rotation():
    """CONVENTION TEST (catches bug a): a frontal face in image convention
    must yield R ~ identity, NOT diag(1,-1,-1)."""
    from geometry_pca.normal_encoder import head_rotation
    from geometry_pca.canonical_face import canonical_template

    face_frontal = _image_frame_keypoints()           # frontal, +Y down
    R = head_rotation(face_frontal, canonical_template())
    assert np.allclose(R, np.eye(3), atol=0.05), (
        f"frontal face must give R~I, got:\n{R}"
    )
    # explicit anti-regression for the spurious flip:
    assert R[1, 1] > 0.9 and R[2, 2] > 0.9, "spurious diag(1,-1,-1) detected"


def test_unit_norm_preserved_under_rotation():
    """R^T applied to unit vectors preserves length (via apply_rotation_field)."""
    from geometry_pca.normal_encoder import apply_rotation_field

    rng = np.random.default_rng(42)
    n = rng.normal(size=(64, 64, 3)).astype(np.float32)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    theta = np.deg2rad(33)
    Ry = np.array([[np.cos(theta), 0, np.sin(theta)],
                   [0, 1, 0],
                   [-np.sin(theta), 0, np.cos(theta)]], dtype=np.float32)
    out = apply_rotation_field(n, Ry.T)
    norms = np.linalg.norm(out, axis=-1)
    assert np.allclose(norms, 1.0, atol=1e-5), "rotation must preserve unit-norm"


def test_derive_variant_shapes_and_dim_helper():
    """raw=12288, xy=8192, rot=12288, rot_xy=8192 — and variant_dim agrees."""
    from geometry_pca.normal_encoder import derive_variant, variant_dim

    rng = np.random.default_rng(1)
    grid = rng.normal(size=(64, 64, 3)).astype(np.float32)
    R = np.eye(3, dtype=np.float32)
    for variant, expected in [("raw", 12288), ("xy", 8192),
                              ("rot", 12288), ("rot_xy", 8192)]:
        v = derive_variant(grid, R, variant)
        assert v.shape == (expected,), f"{variant}: got {v.shape}"
        assert v.dtype == np.float32
        assert variant_dim(variant) == expected, f"variant_dim({variant}) wrong"


def test_derive_variant_derotates_not_doubles():
    """DE-ROTATION TEST (catches bug b): derive_variant('rot') must REMOVE the
    pose, not double it. Yawed normals + the true R must come back ~frontal,
    exercised through the PUBLIC derive_variant path (no manual transpose)."""
    from geometry_pca.normal_encoder import derive_variant, apply_rotation_field

    rng = np.random.default_rng(7)
    # frontal-ish normal field
    n_frontal = np.zeros((64, 64, 3), dtype=np.float32)
    n_frontal[:, :, 2] = 1.0
    n_frontal[:, :, 0] = (rng.random((64, 64), dtype=np.float32) - 0.5) * 0.3
    n_frontal[:, :, 1] = (rng.random((64, 64), dtype=np.float32) - 0.5) * 0.3
    n_frontal /= np.linalg.norm(n_frontal, axis=-1, keepdims=True)

    theta = np.deg2rad(25)
    Ry = np.array([[np.cos(theta), 0, np.sin(theta)],
                   [0, 1, 0],
                   [-np.sin(theta), 0, np.cos(theta)]], dtype=np.float32)
    # observed = R @ n (forward pose applied to canonical normals)
    n_yawed = apply_rotation_field(n_frontal, Ry)

    nz_yawed = n_yawed[:, :, 2].mean()
    # derive_variant must internally apply R^T (de-rotation)
    derot = derive_variant(n_yawed, Ry, "rot").reshape(64, 64, 3)
    nz_derot = derot[:, :, 2].mean()
    nz_frontal = n_frontal[:, :, 2].mean()

    assert nz_derot > nz_yawed + 0.02, (
        f"derive_variant('rot') must move normals TOWARD frontal: "
        f"yawed nz={nz_yawed:.4f}, derot nz={nz_derot:.4f}"
    )
    # strong assertion: de-rotation should approximately RECOVER the frontal field
    assert abs(nz_derot - nz_frontal) < 0.01, (
        f"de-rotation should recover frontal nz ({nz_frontal:.4f}), got {nz_derot:.4f} "
        f"(doubling bug gives ~{np.cos(2*theta)*nz_frontal:.4f})"
    )


def test_end_to_end_rotation_estimation_and_derotation():
    """INTEGRATION (catches a+b together): yawed keypoints in IMAGE convention
    -> head_rotation -> derive_variant('rot') must restore frontal normals."""
    from geometry_pca.normal_encoder import head_rotation, derive_variant, apply_rotation_field
    from geometry_pca.canonical_face import canonical_template

    rng = np.random.default_rng(3)
    theta = np.deg2rad(20)
    Ry = np.array([[np.cos(theta), 0, np.sin(theta)],
                   [0, 1, 0],
                   [-np.sin(theta), 0, np.cos(theta)]], dtype=np.float32)

    # observed keypoints: canonical face yawed by Ry, in IMAGE convention
    face_yawed = _image_frame_keypoints(R=Ry)
    R_est = head_rotation(face_yawed, canonical_template())

    # estimated R should be close to the true Ry
    assert np.allclose(R_est, Ry, atol=0.1), (
        f"estimated rotation should match true yaw:\nR_est=\n{R_est}\nRy=\n{Ry}"
    )

    # normals yawed by the same pose, de-rotated via the public path
    n_frontal = np.zeros((32, 32, 3), dtype=np.float32)
    n_frontal[:, :, 2] = 1.0
    n_yawed = apply_rotation_field(n_frontal, Ry)
    derot = derive_variant(
        np.repeat(np.repeat(n_yawed, 2, axis=0), 2, axis=1),  # 32->64 tile
        R_est, "rot").reshape(64, 64, 3)
    assert derot[:, :, 2].mean() > 0.99, (
        f"end-to-end de-rotation must restore frontal (nz~1), got {derot[:, :, 2].mean():.4f}"
    )


def test_nan_bg_excluded_via_zero_vectors():
    """resample_masked_3ch must treat ZERO-VECTOR background (the real Sapiens
    convention) as invalid — not just pre-set NaNs."""
    from geometry_pca.normal_encoder import resample_masked_3ch

    arr = np.zeros((128, 128, 3), dtype=np.float32)
    # top half = valid unit normals pointing +z; bottom half = zero vectors (bg)
    arr[:64, :, 2] = 1.0
    out = resample_masked_3ch(arr, 0, 0, 128, 128, out_res=8)
    assert out.shape == (8, 8, 3)
    assert np.isfinite(out).all(), "no NaN in output"
    # top rows should average to (0,0,1); bottom rows (pure bg) -> 0
    assert np.allclose(out[:4, :, 2], 1.0, atol=1e-5), "valid normals must survive pooling"
    assert np.allclose(out[4:, :, :], 0.0, atol=1e-6), "pure-background pools to 0"
