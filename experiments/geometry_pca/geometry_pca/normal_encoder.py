"""
Normal-map preprocessing for the z_a (albedo/surface) partition.

Pipeline: load normal.npy + seg.npy + pose.npy -> mask foreground (‖n‖>0.1)
-> face-bbox crop (reuses depth_encoder.face_bbox_px) -> NaN-aware per-channel
average-pooling to 64×64 -> derive flattened per-variant vectors.

Variants:
  raw    — (nx,ny,nz) 64×64×3 = 12,288-d
  xy     — (nx,ny)    64×64×2 = 8,192-d  (nz redundant for camera-facing surfaces)
  rot    — Rᵀ·n on raw, 12,288-d         (head-pose DE-rotation)
  rot_xy — Rᵀ·n xy-only, 8,192-d         (both corrections)

Rotation is estimated from the 68 facial keypoints via Phase 1-R's
orthographic-PnP solver (pose_normalize.estimate_rotation), then the INVERSE
(R^T) is applied pointwise to de-rotate normals into the canonical head frame.
Because rotation commutes with average-pooling, all 4 variants can be derived
from ONE cached raw grid + per-sample R.

Pooled vectors are NOT renormalized — the sub-unit magnitude after pooling
encodes local curvature disagreement (signal).

CONVENTION NOTE (reviewer-verified): canonical_template() is +Y UP; pose.npy
keypoints are IMAGE convention (+Y DOWN). head_rotation flips the template
internally (same as frontalize_dataset and zg_inference). Without the flip, a
frontal face yields the spurious R = diag(1,-1,-1).
"""
import numpy as np
import os
from geometry_pca.constants import STRATUM_ROOT, FACE_SLICE
from geometry_pca.pose_normalize import estimate_rotation
from geometry_pca.depth_encoder import face_bbox_px, resample_masked

OUT_RES = 64


def load_normal_sample(sample_id):
    """Load normal, seg, and face keypoints for one sample.

    Returns: normal (HxWx3 f32), seg (HxW uint8), face (68,3 f32)
    Raises FileNotFoundError if any file is missing.
    """
    base = os.path.join(STRATUM_ROOT, sample_id)
    normal = np.load(os.path.join(base, "normal.npy")).astype(np.float32)
    seg = np.load(os.path.join(base, "seg.npy"))
    pose = np.load(os.path.join(base, "pose.npy")).astype(np.float32)
    face = pose[FACE_SLICE]  # (68,3)
    return normal, seg, face


def resample_masked_3ch(arr, x0, y0, x1, y1, out_res=OUT_RES):
    """Per-channel NaN-aware average-pool for a 3-channel map.

    Background (zero-vector / near-zero magnitude) pixels are converted to NaN
    so pooling ignores them. Returns (out_res, out_res, 3) float32.
    """
    mag = np.linalg.norm(arr, axis=-1, keepdims=True)
    valid = (mag > 0.1).astype(np.float32)
    out = np.zeros((out_res, out_res, 3), dtype=np.float32)
    for ch in range(3):
        channel = arr[:, :, ch].copy()
        channel[valid[:, :, 0] == 0] = np.nan
        out[:, :, ch] = resample_masked(channel, x0, y0, x1, y1, out_res=out_res)
    return out


def head_rotation(face_2d, canonical_tpl):
    """Estimate head rotation R (3x3) from 68 face keypoints.

    CONVENTION: canonical_template() is +Y UP; pose.npy keypoints are IMAGE
    convention (+Y DOWN). The template is flipped internally (same correction
    as pose_normalize.frontalize_dataset and zg_inference) so R is estimated
    in the image frame: a frontal face yields R ≈ identity. Without the flip,
    a frontal face returns the spurious rotation diag(1,-1,-1).

    Args:
        face_2d: (68,2) face keypoints in [-1,1] image-normalized space (+Y down)
        canonical_tpl: (68,3) 3D canonical template (+Y up, as returned by
                       canonical_face.canonical_template)

    Returns: R (3,3) float32 proper rotation matrix (det=+1), image-frame,
             mapping canonical/frontal -> observed pose.
    """
    tpl = canonical_tpl.copy()
    tpl[:, 1] *= -1.0  # +Y up (template) -> +Y down (image/pose convention)
    return estimate_rotation(tpl, face_2d).astype(np.float32)


def apply_rotation_field(normal_grid, R):
    """Apply rotation matrix R to every pixel's normal vector (pointwise R @ n).

    NOTE: this applies R as given (FORWARD rotation). For DE-rotation
    (canonical-frame normals), the caller passes R.T — derive_variant does
    this for the rot/rot_xy variants.

    Args:
        normal_grid: (64,64,3) normal vectors
        R: (3,3) rotation matrix

    Returns: (64,64,3) rotated normals (NOT renormalized — preserves ‖n‖<1
             from pooling, which encodes curvature disagreement)
    """
    orig = normal_grid.shape
    flat = normal_grid.reshape(-1, 3)
    rot = (flat @ R.T).reshape(orig)  # (R @ n) per pixel == n @ R.T row-wise
    return rot.astype(np.float32)


def derive_variant(grid_64, R, variant):
    """Derive a flattened vector from a 64×64×3 normal grid + rotation matrix.

    For rot/rot_xy the INVERSE rotation R^T is applied (de-rotation into the
    canonical head frame): n_canonical = R^T @ n_observed. Reviewer-verified:
    applying R forward would DOUBLE the pose instead of removing it.

    Args:
        grid_64: (64,64,3) float32, resampled masked normals (raw, NOT rotated)
        R: (3,3) float32 head rotation (canonical -> observed, from head_rotation)
        variant: "raw"|"xy"|"rot"|"rot_xy"

    Returns: (D,) float32 vector; D = 12288 (3ch) or 8192 (xy)
    """
    if variant == "raw":
        return grid_64.reshape(-1).astype(np.float32)
    elif variant == "xy":
        return grid_64[:, :, :2].reshape(-1).astype(np.float32)
    elif variant == "rot":
        derot = apply_rotation_field(grid_64, R.T)  # R^T = de-rotation
        return derot.reshape(-1).astype(np.float32)
    elif variant == "rot_xy":
        derot = apply_rotation_field(grid_64, R.T)  # R^T = de-rotation
        return derot[:, :, :2].reshape(-1).astype(np.float32)
    else:
        raise ValueError(f"unknown variant: {variant}")


def variant_dim(variant):
    """Dimensionality of a variant's flattened vector. ('xy' suffix = 2ch)"""
    if variant not in ("raw", "xy", "rot", "rot_xy"):
        raise ValueError(f"unknown variant: {variant}")
    return 8192 if variant.endswith("xy") else 12288
