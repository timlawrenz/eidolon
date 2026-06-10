"""
z_d depth inference helper — reusable encode path for the Fisher gate and future
inference. Operates on in-memory arrays (depth/seg/face), not disk, so the gate
extractor can feed it tensors directly from enriched hegre directories.

Runs the same preprocessing pipeline as depth_encoder.encode_depth_sample but
stays decoupled from the sample-level loader — the gate owns materialisation.
"""
import numpy as np
from geometry_pca import depth_encoder as de


def encode_zd(depth, seg, face, encoder, mode="C", dataset_sigma=0.15):
    """Full preprocess + encode for a single sample.

    Args:
        depth:  (H,W) float32 depth map (Sapiens)
        seg:    (H,W) uint8 seg map (28-class)
        face:   (68,3) float32 COCO-WholeBody face points
        encoder: dict with keys components (K,D), pca_mean (D,),
                 whiten_mu (K,), whiten_sigma (K,)
        mode: "A" | "A_prime" | "C"
        dataset_sigma: dataset-level depth std for A_prime mode

    Returns:
        z_d: (K,) float32 whitened score, or None if preproc fails
    """
    fgmask = seg > 0
    if fgmask.sum() < 50:
        return None

    norm = de.normalize_depth(depth, fgmask, face, mode, dataset_sigma=dataset_sigma)
    if norm is None:
        return None

    x0, y0, x1, y1 = de.face_bbox_px(face, depth.shape[0], depth.shape[1])
    grid = de.resample_masked(norm, x0, y0, x1, y1)
    vec = grid.reshape(-1).astype(np.float32)

    comps = encoder["components"]
    pmean = encoder["pca_mean"]
    wmu = encoder["whiten_mu"]
    wsig = encoder["whiten_sigma"]

    # project through PCA then whiten
    raw = (vec - pmean) @ comps.T
    z_d = (raw - wmu) / wsig
    return z_d.astype(np.float32)


def whiten_scores(Z, mu, sigma):
    """Whiten (N,K) raw PCA scores with per-component mu, sigma. Z-score in place.

    Used by the gate harness to re-standardize z_d on the hegre gate distribution
    itself (domain-shift fix: the FFHQ-fit whitening doesn't apply to hegre).
    """
    return (Z - mu) / np.maximum(sigma, 1e-8)
