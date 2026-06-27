"""
AuraFace preprocessing — deterministic nuisance removal.

Removes two measured nuisances from AuraFace (512-d) vectors before they enter
identity analysis or DiT conditioning:

  a) PC1 (domain axis): separates FFHQ from Hegre capture style. R²≈0 at identity,
     but it's the largest variance direction and a domain confound for any pooled
     analysis.

  b) Yaw direction: single direction that encodes ~41% of head-pose (horizontal)
     variance as measured by held-out ridge regression against DWPose yaw.
     Removing it costs zero identity discrimination (ΔAUC < 0.0001).

Both directions are saved in the reference artifact at:
    experiments/geometry_pca/output/auraface_preprocess.npz
computed from 140,217 pooled FFHQ+Hegre AuraFace vectors on 2026-06-27.

Usage:
    from geometry_pca.auraface_preprocessing import clean_auraface
    v_clean = clean_auraface(raw_auraface_vector)          # single (512,)
    V_clean = clean_auraface(raw_batch, renormalize=True)  # (N,512)
"""

import numpy as np
from pathlib import Path

_REF_PATH = Path(__file__).resolve().parent.parent / "output" / "auraface_preprocess.npz"
_REF = None


def _load_ref():
    global _REF
    if _REF is None:
        d = np.load(_REF_PATH)
        _REF = {
            "mu": d["pooled_mean"],
            "pc1": d["pc1_direction"],
            "yaw": d["yaw_direction"],
        }
    return _REF


def clean_auraface(v, *, renormalize=True):
    """Remove domain (PC1) and pose (yaw) nuisances from AuraFace vector(s).

    Args:
        v: (512,) float array or (N,512) float array — raw AuraFace embedding(s)
        renormalize: if True, L2-normalize the result back to the unit hypersphere
                     (the raw output of projection is slightly off-sphere)

    Returns:
        (same shape as v) cleaned vector(s), float64
    """
    ref = _load_ref()
    mu = ref["mu"]
    pc1 = ref["pc1"]
    yaw = ref["yaw"]

    v = np.atleast_2d(np.asarray(v, dtype=np.float64))
    was_1d = (v == np.atleast_2d(np.squeeze(v))).all() and v.shape[0] != v.shape[1]

    vc = v - mu                               # center
    vc = vc - np.outer(vc @ pc1, pc1)        # remove domain axis
    vc = vc - np.outer(vc @ yaw, yaw)        # remove yaw component

    if renormalize:
        norms = np.linalg.norm(vc, axis=1, keepdims=True)
        vc = vc / (norms + 1e-12)

    return vc.squeeze() if was_1d else vc
