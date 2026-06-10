"""
Phase 2: volumetric DEPTH encoder (z_d) preprocessing.

Pipeline: load depth.npy + seg.npy + pose.npy -> mask foreground -> face-crop
(via 68-pt keypoint bbox) -> NORMALIZE (swept: A / A_prime / C) on raw-res
foreground -> mask-aware resample to 64x64 -> flatten.

Normalization variants (the Fisher gate picks the winner):
  A        masked per-image z-score: (z - mean_fg) / std_fg
  A_prime  center-only, dataset-level scale: (z - mean_fg) / DATASET_SIGMA
  C        anatomical anchor: (z - z_nose) / D_eyes   (preserves relief magnitude,
           neutralizes camera zoom via inter-ocular pixel distance)
"""
import numpy as np
import os
from geometry_pca.constants import STRATUM_ROOT, FACE_SLICE

DEPTH_GRID = 1024
OUT_RES = 64
# iBUG-68 indices (within the 0..67 face array)
NOSE_TIP = 30
EYE_R_OUTER = 36
EYE_L_OUTER = 45


def _pose_to_px(xy_norm):
    """Map pose [-1,1] coords to depth-grid pixels."""
    return (xy_norm + 1.0) / 2.0 * DEPTH_GRID


def load_depth_sample(sample_id):
    base = os.path.join(STRATUM_ROOT, sample_id)
    depth = np.load(os.path.join(base, "depth.npy")).astype(np.float32)
    seg = np.load(os.path.join(base, "seg.npy"))
    pose = np.load(os.path.join(base, "pose.npy")).astype(np.float32)
    face = pose[FACE_SLICE]  # (68,3)
    return depth, seg, face


def face_bbox_px(face, pad=0.35):
    """Bounding box (in depth pixels) around the 68 face keypoints, padded."""
    px = _pose_to_px(face[:, :2])
    mn = px.min(axis=0); mx = px.max(axis=0)
    span = (mx - mn).max() * (1 + pad)
    cx, cy = (mn + mx) / 2
    x0 = int(np.clip(cx - span / 2, 0, DEPTH_GRID - 1))
    y0 = int(np.clip(cy - span / 2, 0, DEPTH_GRID - 1))
    x1 = int(np.clip(cx + span / 2, 1, DEPTH_GRID))
    y1 = int(np.clip(cy + span / 2, 1, DEPTH_GRID))
    return x0, y0, x1, y1


def normalize_depth(depth, fgmask, face, mode, dataset_sigma=None):
    """Return depth normalized per `mode`, with background set to NaN (so the
    mask-aware resample can ignore it)."""
    z = depth.copy()
    fg = fgmask & (depth > 0)
    if fg.sum() < 50:
        return None
    if mode == "A":
        mu = z[fg].mean(); sd = z[fg].std()
        if sd < 1e-6:
            return None
        out = (z - mu) / sd
    elif mode == "A_prime":
        mu = z[fg].mean()
        sd = dataset_sigma if dataset_sigma else 1.0
        out = (z - mu) / sd
    elif mode == "C":
        # Anatomical anchor. Depth is in [0,1] image-normalized units; D_eyes must
        # be in the SAME units (fraction of image), NOT raw pixels, or the ratio
        # collapses. Convert inter-ocular distance to image-fraction [0,1].
        px = _pose_to_px(face[:, :2])
        def depth_at(i):
            x, y = int(np.clip(px[i, 0], 0, DEPTH_GRID - 1)), int(np.clip(px[i, 1], 0, DEPTH_GRID - 1))
            return z[y, x]
        z_nose = depth_at(NOSE_TIP)
        re = px[EYE_R_OUTER]; le = px[EYE_L_OUTER]
        d_eyes_px = np.hypot(*(le - re))
        d_eyes_frac = d_eyes_px / DEPTH_GRID   # -> [0,1], commensurate with depth
        if d_eyes_frac < 1e-4:
            return None
        out = (z - z_nose) / d_eyes_frac
    else:
        raise ValueError(mode)
    out = out.astype(np.float32)
    out[~fg] = np.nan  # background -> NaN, excluded from resample
    return out


def resample_masked(arr, x0, y0, x1, y1, out_res=OUT_RES):
    """Mask-aware downsample: crop, then average-pool ignoring NaN (background).
    Vectorized via reduceat — ~100x faster than the nested-loop version."""
    crop = arr[y0:y1, x0:x1]
    h, w = crop.shape
    if h < out_res or w < out_res:
        # too small to pool meaningfully; nearest-resize via index
        yi = np.linspace(0, h - 1, out_res).astype(int)
        xi = np.linspace(0, w - 1, out_res).astype(int)
        small = crop[np.ix_(yi, xi)]
        return np.nan_to_num(small, nan=0.0).astype(np.float32)
    valid = ~np.isnan(crop)
    vals = np.where(valid, crop, 0.0)
    ys = np.linspace(0, h, out_res + 1).astype(int)[:-1]
    xs = np.linspace(0, w, out_res + 1).astype(int)[:-1]
    # sum over row-blocks then col-blocks
    sum_r = np.add.reduceat(vals, ys, axis=0)
    cnt_r = np.add.reduceat(valid.astype(np.float32), ys, axis=0)
    sum_rc = np.add.reduceat(sum_r, xs, axis=1)
    cnt_rc = np.add.reduceat(cnt_r, xs, axis=1)
    out = np.divide(sum_rc, cnt_rc, out=np.zeros_like(sum_rc), where=cnt_rc > 0)
    return out.astype(np.float32)


def encode_depth_sample(sample_id, mode, dataset_sigma=None):
    """Full preprocess for one sample -> flattened (OUT_RES*OUT_RES,) vector, or None."""
    try:
        depth, seg, face = load_depth_sample(sample_id)
    except FileNotFoundError:
        return None
    fgmask = seg > 0
    norm = normalize_depth(depth, fgmask, face, mode, dataset_sigma)
    if norm is None:
        return None
    x0, y0, x1, y1 = face_bbox_px(face)
    grid = resample_masked(norm, x0, y0, x1, y1)
    return grid.reshape(-1)
