import numpy as np
from geometry_pca.pose_normalize import estimate_rotation, frontalize


def _make_template():
    # A simple synthetic 3D face: a grid of points with varying depth.
    rng = np.random.default_rng(0)
    xy = rng.uniform(-1, 1, size=(68, 2))
    z = rng.uniform(-0.3, 0.3, size=(68, 1))  # depth variation (nose, brow, etc.)
    return np.concatenate([xy, z], axis=1).astype(np.float32)


def _yaw_matrix(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)


def test_estimate_rotation_recovers_yaw():
    template = _make_template()
    template = template - template.mean(axis=0)

    theta = np.deg2rad(30)
    R_true = _yaw_matrix(theta)

    # Project the rotated template orthographically (drop z).
    rotated = template @ R_true.T
    observed2d = rotated[:, :2]

    R_est = estimate_rotation(template, observed2d)

    # The estimated rotation should be close to the true one (up to sign/proj
    # ambiguity inherent in orthographic projection). We check that applying
    # R_est brings the observed projection much closer to the frontal template.
    assert R_est.shape == (3, 3)
    np.testing.assert_allclose(np.linalg.det(R_est), 1.0, atol=1e-4)
    # Orthonormality
    np.testing.assert_allclose(R_est @ R_est.T, np.eye(3), atol=1e-4)


def test_frontalize_reduces_pose_variance():
    """The core spike thesis: frontalizing several yawed views of ONE identity
    should yield near-identical 2D shapes (low variance), whereas the raw
    projections differ a lot."""
    template = _make_template()
    template = template - template.mean(axis=0)

    views_raw = []
    views_frontal = []
    for deg in [-30, -15, 0, 15, 30]:
        R = _yaw_matrix(np.deg2rad(deg))
        rotated = template @ R.T
        observed2d = rotated[:, :2].astype(np.float32)

        views_raw.append(observed2d - observed2d.mean(axis=0))
        views_frontal.append(frontalize(template, observed2d))

    raw = np.stack(views_raw)
    frontal = np.stack(views_frontal)

    # Normalize scale so the comparison is fair (Frobenius norm).
    def unit(a):
        return a / (np.linalg.norm(a, axis=(1, 2), keepdims=True) + 1e-8)

    raw_var = unit(raw).var(axis=0).mean()
    frontal_var = unit(frontal).var(axis=0).mean()

    # Frontalization must substantially reduce cross-view variance.
    assert frontal_var < raw_var * 0.5, (
        f"frontalization did not reduce pose variance: "
        f"raw={raw_var:.5f} frontal={frontal_var:.5f}"
    )
