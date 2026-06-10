"""
EPnP/orthographic pose-normalization spike for the geometry encoder (Phase 1-R).

Goal: factor out out-of-plane head rotation (yaw/pitch) that plain 2D GPA cannot
remove, by estimating a 3D head rotation from the 68 2D keypoints against a
canonical 3D mean-face template, rotating the (lifted) points to a frontal frame,
and reprojecting to 2D for the existing PCA pipeline.

Method: we have a canonical 3D template X (68,3) and observed 2D points y (68,2).
Under a weak-perspective (orthographic + scale) camera, y ~= s * P * R * X + t,
where P drops the z-row. We solve for the full 3D rotation R that best explains
the 2D observation via the standard "orthographic-n-point" least-squares: recover
the two rows of (s*R) that map X->y, then reconstruct the third row as their
cross product (re-orthonormalized). Applying R^T to the lifted template-aligned
points rotates the face back to frontal. This is deterministic linear algebra
(no iterative optimizer, no local minima).
"""
import numpy as np


def estimate_rotation(template3d: np.ndarray, observed2d: np.ndarray) -> np.ndarray:
    """
    Estimate a 3x3 rotation matrix R such that the observed 2D landmarks are
    approximately the orthographic projection of R applied to the 3D template.

    Args:
        template3d: (68, 3) canonical mean-face 3D coordinates (centered).
        observed2d: (68, 2) observed keypoints for one sample (centered).

    Returns:
        R: (3, 3) orthonormal rotation matrix (proper, det=+1).
    """
    X = template3d - template3d.mean(axis=0)
    y = observed2d - observed2d.mean(axis=0)

    # Solve y ~= X @ A  for A (3x2): least squares maps 3D template -> 2D obs.
    # A's columns are the (scaled) first two rows of the rotation.
    A, *_ = np.linalg.lstsq(X, y, rcond=None)  # (3, 2)

    r1 = A[:, 0]
    r2 = A[:, 1]

    # Normalize to unit length (strip the weak-perspective scale).
    n1 = np.linalg.norm(r1)
    n2 = np.linalg.norm(r2)
    if n1 < 1e-8 or n2 < 1e-8:
        return np.eye(3, dtype=np.float32)
    r1 = r1 / n1
    r2 = r2 / n2

    # Re-orthogonalize r2 against r1 (Gram-Schmidt) so the basis is clean.
    r2 = r2 - np.dot(r1, r2) * r1
    n2 = np.linalg.norm(r2)
    if n2 < 1e-8:
        return np.eye(3, dtype=np.float32)
    r2 = r2 / n2

    # Third axis via cross product -> proper right-handed rotation.
    r3 = np.cross(r1, r2)

    R = np.stack([r1, r2, r3], axis=0)  # rows are the rotated basis

    # Guarantee a proper rotation (det = +1) via SVD projection onto SO(3).
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R.astype(np.float32)


def frontalize(template3d: np.ndarray, observed2d: np.ndarray) -> np.ndarray:
    """
    Pose-normalize a single sample: estimate its head rotation, then take the
    canonical template rotated *back* to frontal but warped toward the observed
    shape, and reproject to 2D.

    Strategy: lift the observed 2D into 3D by borrowing the template's z (depth)
    channel, apply R^T to remove the estimated head rotation, and drop z.
    The depth channel is what lets a profile's nose-projection survive as 2D
    x-spread after frontalization (the "depth bonus").

    Args:
        template3d: (68, 3) canonical mean-face 3D coordinates (centered).
        observed2d: (68, 2) observed keypoints for one sample (centered).

    Returns:
        (68, 2) pose-normalized (frontalized) 2D coordinates.
    """
    X = template3d - template3d.mean(axis=0)
    y = observed2d - observed2d.mean(axis=0)

    R = estimate_rotation(X, y)

    # Lift observed 2D to 3D using the template's depth as a prior.
    lifted = np.concatenate([y, X[:, 2:3]], axis=1)  # (68, 3)

    # Remove the estimated head rotation -> frontal frame.
    frontal3d = lifted @ R  # R rows are rotated basis; right-multiply rotates back

    return frontal3d[:, :2].astype(np.float32)
