import numpy as np
from geometry_pca.canonical_face import canonical_template
from geometry_pca.pose_normalize import frontalize, estimate_rotation


def test_template_shape_and_centered():
    t = canonical_template()
    assert t.shape == (68, 3)
    np.testing.assert_allclose(t.mean(axis=0), 0, atol=1e-5)


def test_template_has_real_depth_structure():
    """Nose tip (idx 30) must be the most forward point; eye corners set back."""
    t = canonical_template()
    nose_tip_z = t[30, 2]
    # nose tip should be among the most-forward (max Z) landmarks
    assert nose_tip_z >= np.percentile(t[:, 2], 90)
    # jaw sides (idx 0 and 16) should be set back (low Z)
    assert t[0, 2] < 0 and t[16, 2] < 0


def _yaw_matrix(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)


def test_frontalize_with_canonical_reduces_yaw_variance():
    """Using the real canonical template, frontalizing yawed projections of the
    template itself should collapse cross-view variance."""
    t = canonical_template()

    raws, fronts = [], []
    for deg in [-40, -20, 0, 20, 40]:
        R = _yaw_matrix(np.deg2rad(deg))
        proj = (t @ R.T)[:, :2].astype(np.float32)
        proj = proj - proj.mean(axis=0)
        raws.append(proj / (np.linalg.norm(proj) + 1e-8))
        f = frontalize(t, proj)
        fronts.append(f / (np.linalg.norm(f) + 1e-8))

    raw_var = np.stack(raws).var(axis=0).mean()
    front_var = np.stack(fronts).var(axis=0).mean()
    assert front_var < raw_var * 0.5, f"raw={raw_var:.5f} front={front_var:.5f}"
