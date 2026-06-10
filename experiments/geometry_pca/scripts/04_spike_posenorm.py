#!/usr/bin/env python3
"""
Phase 1-R EPnP spike: pose-normalize the dataset, re-fit PCA, and re-run the
validation gate (scree + traversals) plus a synthetic pose-invariance probe.
Changes exactly one variable vs Phase 1: alignment is now 3D-aware frontalization
instead of plain 2D GPA.
"""
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geometry_pca.loader import load_face_keypoints, iter_sample_ids
from geometry_pca.gpa import gpa_align, center_and_scale
from geometry_pca.fit import fit_encoder, save_encoder
from geometry_pca.pose_normalize import frontalize, estimate_rotation

OUT = "output"
LIMIT = 2000
K = 50


def load_shapes(limit):
    shapes = []
    for sid in iter_sample_ids(limit):
        try:
            coords, conf = load_face_keypoints(sid, return_conf=True)
            if np.mean(conf) >= 0.5:
                shapes.append(coords)
        except FileNotFoundError:
            continue
    return np.stack(shapes)


def build_template(shapes):
    """Canonical 3D template: GPA-mean (x,y) + a depth prior z.
    Depth prior: nose/brow forward, eye-sockets/jaw-sides back. We derive a
    simple radial depth from the mean shape (center forward) as a neutral prior;
    the frontalization's 'depth bonus' will refine via observed x-spread."""
    _, mean_xy = gpa_align(shapes)              # (68,2), centered+scaled
    # Neutral depth prior: points near the vertical midline (nose ridge) sit
    # forward (+z), outer points (jaw, ears) sit back (-z).
    x = mean_xy[:, 0]
    z = (0.15 - np.abs(x) * 0.4).astype(np.float32)   # forward at center
    template = np.concatenate([mean_xy, z[:, None]], axis=1).astype(np.float32)
    return template - template.mean(axis=0)


def main():
    os.makedirs(OUT, exist_ok=True)
    print(f"Loading {LIMIT} shapes...")
    shapes = load_shapes(LIMIT)
    print(f"Loaded {len(shapes)} valid faces.")

    print("Building canonical 3D template...")
    template = build_template(shapes)

    print("Frontalizing all shapes (3D-aware pose normalization)...")
    frontal = np.stack([frontalize(template, center_and_scale(s)) for s in shapes])

    # Re-apply a light 2D GPA on the frontalized shapes to fix residual roll/scale.
    aligned, gpa_mean = gpa_align(frontal)

    print(f"Fitting PCA (k={K}) on pose-normalized shapes...")
    encoder = fit_encoder(aligned, gpa_mean, k=K)
    save_encoder(encoder, os.path.join(OUT, "encoder_posenorm.npz"))
    evr = encoder["explained_variance_ratio"]
    print(f"Retained {evr.sum()*100:.2f}% variance at k={K}.")

    # --- Traversal plots for C1-C5 (same gate as Phase 1) ---
    print("Generating pose-normalized traversals C1-C5...")
    pca_mean = encoder["pca_mean"]
    for i in range(5):
        comp = encoder["components"][i]
        sig = encoder["whiten_sigma"][i]; mu = encoder["whiten_mu"][i]
        plt.figure(figsize=(6, 6))
        for z, c in zip([-3, -1.5, 0, 1.5, 3], ['red','orange','gray','blue','purple']):
            sh = (pca_mean + (z*sig+mu)*comp).reshape(-1, 2)
            plt.scatter(sh[:, 0], -sh[:, 1], c=c, label=f'{z}σ', alpha=0.6, s=15)
        plt.title(f'[POSE-NORM] Component {i+1} Traversal (±3σ)')
        plt.axis('equal'); plt.legend(); plt.grid(True, ls=':', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT, f"posenorm_traversal_C{i+1}.png"))
        plt.close()

    # --- POSE-INVARIANCE PROBE (the new gate) ---
    # Take ONE identity, synthesize yaw variants, encode each, measure z_g spread.
    print("Running pose-invariance probe...")

    def encode(shape2d):
        f = frontalize(template, center_and_scale(shape2d))
        # align single frontalized shape to gpa_mean
        from geometry_pca.gpa import align_single
        a = align_single(f, gpa_mean).reshape(-1)
        score = (a - pca_mean) @ encoder["components"].T
        return (score - encoder["whiten_mu"]) / encoder["whiten_sigma"]

    def yaw3d(shape2d, deg, template):
        # lift to 3D w/ template depth, rotate by yaw, reproject
        c = center_and_scale(shape2d)
        lifted = np.concatenate([c, template[:, 2:3]], axis=1)
        th = np.deg2rad(deg)
        R = np.array([[np.cos(th),0,np.sin(th)],[0,1,0],[-np.sin(th),0,np.cos(th)]])
        return (lifted @ R.T)[:, :2].astype(np.float32)

    base = shapes[0]
    zs_posenorm = []
    zs_raw = []
    # raw encoder (Phase 1 style: no frontalization) for comparison
    raw_aligned, raw_mean = gpa_align(shapes)
    raw_enc = fit_encoder(raw_aligned, raw_mean, k=K)
    from geometry_pca.encode import encode_pose

    for deg in [-30, -15, 0, 15, 30]:
        yawed = yaw3d(base, deg, template)
        zs_posenorm.append(encode(yawed))
        zs_raw.append(encode_pose(yawed, raw_enc))

    zs_posenorm = np.stack(zs_posenorm)
    zs_raw = np.stack(zs_raw)

    # Variance of z_g across the synthetic pose set (lower = more pose-invariant)
    posenorm_spread = float(zs_posenorm.std(axis=0).mean())
    raw_spread = float(zs_raw.std(axis=0).mean())

    metrics = {
        "retained_variance": float(evr.sum()),
        "pose_invariance_probe": {
            "posenorm_zg_std_across_yaw": posenorm_spread,
            "raw_zg_std_across_yaw": raw_spread,
            "improvement_ratio": raw_spread / posenorm_spread if posenorm_spread > 0 else None,
        },
    }
    with open(os.path.join(OUT, "metrics_posenorm.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print("Spike complete. Inspect output/posenorm_traversal_C*.png")


if __name__ == "__main__":
    main()
