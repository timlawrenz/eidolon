"""
z_a albedo/surface inference helper — reusable encode path for the AUC gate
and future inference. Operates on in-memory arrays (normal/seg/face).

Mirrors the zd_inference pattern: full preprocess (fg-mask -> bbox-crop ->
resample -> variant derivation -> PCA -> whiten) but for normal maps.
"""
import numpy as np
from geometry_pca.normal_encoder import (
    resample_masked_3ch, head_rotation, derive_variant
)
from geometry_pca.canonical_face import canonical_template
from geometry_pca.depth_encoder import face_bbox_px

CANONICAL_TPL = canonical_template()


def encode_za(normal, seg, face, encoder, variant="raw"):
    """Full preprocess + encode for one sample.

    Args:
        normal: (H,W,3) float32 normal map (Sapiens)
        seg:     (H,W) uint8 seg map (28-class)
        face:    (68,3) float32 COCO-WholeBody face points
        encoder: dict with keys components (K,D), pca_mean (D,),
                 whiten_mu (K,), whiten_sigma (K,)
        variant: "raw"|"xy"|"rot"|"rot_xy"

    Returns:
        z_a: (K,) float32 whitened score, or None if preprocessing fails
    """
    # foreground mask
    mag = np.linalg.norm(normal, axis=-1)
    fgmask = ((seg > 0) & (mag > 0.1)).astype(np.float32)
    if fgmask.sum() < 50:
        return None

    # set bg to zero (resample_masked_3ch treats zero-magnitude as NaN internally)
    normal_bg = normal.copy()
    normal_bg[fgmask < 0.5] = 0.0

    # face bbox crop (reuses depth_encoder's bbox logic)
    face_2d = face[:, :2]
    x0, y0, x1, y1 = face_bbox_px(face, normal.shape[0], normal.shape[1])

    # resample to 64x64x3
    grid = resample_masked_3ch(normal_bg, x0, y0, x1, y1)

    # estimate head rotation from keypoints
    R = head_rotation(face_2d, CANONICAL_TPL)

    # derive variant vector
    vec = derive_variant(grid, R, variant)

    # PCA + whiten
    comps = encoder["components"]
    pmean = encoder["pca_mean"]
    wmu = encoder["whiten_mu"]
    wsig = encoder["whiten_sigma"]
    raw = (vec - pmean) @ comps.T
    z_a = (raw - wmu) / wsig
    return z_a.astype(np.float32)
