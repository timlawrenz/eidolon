"""
Normal-map preprocessing for the z_a (albedo/surface) partition.

Pipeline: load normal.npy + seg.npy + pose.npy -> mask foreground (‖n‖>0.1)
-> face-bbox crop (reuses depth_encoder.face_bbox_px) -> NaN-aware per-channel
average-pooling to 64×64 -> derive flattened per-variant vectors.

Variants:
  raw    — (nx,ny,nz) 64×64×3 = 12,288-d
  xy     — (nx,ny)    64×64×2 = 8,192-d  (nz redundant for camera-facing surfaces)
  rot    — R^T·n on raw, 12,288-d         (head-pose de-rotation)
  rot_xy — R^T·n xy-only, 8,192-d         (both corrections)

Rotation is estimated from the 68 facial keypoints via Phase 1-R's
orthographic-PnP solver (pose_normalize.estimate_rotation), then applied
pointwise to normal vectors. Because rotation commutes with average-pooling,
all 4 variants can be derived from ONE cached raw grid + per-sample R.

Pooled vectors are NOT renormalized — the sub-unit magnitude after pooling
encodes local curvature disagreement (signal).
"""
import numpy as np
import os
from geometry_pca.constants import STRATUM_ROOT, FACE_SLICE
from geometry_pca.pose_normalize import estimate_rotation
from geometry_pca.depth_encoder import face_bbox_px, resample_masked

# iBUG-68 indices for the canonical 300W 3D template used by estimate_rotation
# (same indices depth_encoder uses for nose/eye anchors, but estimate_rotation
#  uses ALL 68 points against the full 3D canonical template)
NOSE_TIP = 30
EYE_R_OUTER = 36
EYE_L_OUTER = 45
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

    Normal maps have background == zero vectors, which we convert to NaN
    so pooling ignores them. Returns (out_res, out_res, 3) float32.
    """
    # Set zero-vectors (background) to NaN per-channel
    # (slightly conservative: any pixel with |n| < 0.1 is treated as bg)
    mag = np.linalg.norm(arr, axis=-1, keepdims=True)
    valid = (mag > 0.1).astype(np.float32)
    out = np.zeros((out_res, out_res, 3), dtype=np.float32)
    for ch in range(3):
        channel = arr[:, :, ch].copy()
        channel[valid[:, :, 0] == 0] = np.nan
        out[:, :, ch] = resample_masked(channel, x0, y0, x1, y1, out_res=out_res)
    return out


def head_rotation(face_2d, canonical_template):
    """Estimate head rotation R (3x3) from 68 face keypoints.

    Args:
        face_2d: (68,2) face keypoints in [-1,1] image-normalized space
        canonical_template: (68,3) 3D canonical template (from production encoder)

    Returns: R (3,3) float32 proper rotation matrix (det=+1).
    """
    return estimate_rotation(canonical_template, face_2d).astype(np.float32)


def apply_rotation_field(normal_grid, R):
    """Rotate every pixel's normal vector by R (pointwise R @ n).

    Args:
        normal_grid: (64,64,3) normal vectors
        R: (3,3) rotation matrix

    Returns: (64,64,3) rotated normals (NOT renormalized — preserves ‖n‖<1 from pooling)
    """
    orig = normal_grid.shape
    flat = normal_grid.reshape(-1, 3)
    rot = (flat @ R.T).reshape(orig)
    return rot.astype(np.float32)


def derive_variant(grid_64, R, variant):
    """Derive a flattened vector from a 64×64×3 normal grid + rotation matrix.

    Args:
        grid_64: (64,64,3) float32, resampled masked normals (raw, NOT rot-applied)
        R: (3,3) float32 head rotation matrix
        variant: "raw"|"xy"|"rot"|"rot_xy"

    Returns: (D,) float32 vector where D is the variant dimension
    """
    if variant == "raw":
        return grid_64.reshape(-1).astype(np.float32)
    elif variant == "xy":
        return grid_64[:, :, :2].reshape(-1).astype(np.float32)
    elif variant == "rot":
        rot_grid = apply_rotation_field(grid_64, R)
        return rot_grid.reshape(-1).astype(np.float32)
    elif variant == "rot_xy":
        rot_grid = apply_rotation_field(grid_64, R)
        return rot_grid[:, :, :2].reshape(-1).astype(np.float32)
    else:
        raise ValueError(f"unknown variant: {variant}")
