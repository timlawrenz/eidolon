"""
z_g geometry inference helper — reusable encode path mirroring the closure pattern
in 07_gate_sweep.py but lifted into an importable module for the gate extractor.
"""
import numpy as np
from geometry_pca.pose_normalize import frontalize
from geometry_pca.gpa import center_and_scale, align_single


def encode_zg(face_2d, production_encoder):
    """Encode a single (68,2) face into the whitened z_g vector.

    Args:
        face_2d: (68,2) float32 face keypoints (COCO-WholeBody face slice,
                 x/y only, in raw [-1,1] image-normalized coordinates)
        production_encoder: dict from output/encoder_production.npz with keys:
            components (50,136), pca_mean (136,), whiten_mu (50,),
            whiten_sigma (50,), gpa_mean (68,2), canonical_template (68,3)

    Returns:
        z_g: (50,) float32 whitened geometry vector
    """
    tpl = production_encoder["canonical_template"].copy()
    tpl[:, 1] *= -1  # Y-flip for coordinate system correction
    gmean = production_encoder["gpa_mean"]
    comps = production_encoder["components"]
    pmean = production_encoder["pca_mean"]
    wmu = production_encoder["whiten_mu"]
    wsig = production_encoder["whiten_sigma"]

    centered = center_and_scale(face_2d)
    frontal = frontalize(tpl, centered)
    aligned = align_single(frontal, gmean).reshape(-1)
    raw = (aligned - pmean) @ comps.T
    return ((raw - wmu) / wsig).astype(np.float32)
