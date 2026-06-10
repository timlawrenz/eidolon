#!/usr/bin/env python3
"""
Phase 1-R PRODUCTION fit: pose-invariant geometry encoder on the FULL FFHQ set.

Pipeline (one variable changed vs Phase 1 — alignment is now 3D-aware):
  load pose -> slice 68 face pts -> frontalize against the canonical 300W 3D
  template -> light 2D GPA -> PCA -> whiten. The canonical template is persisted
  into the encoder so inference reproduces the exact frontalization.
"""
import os
import sys
import json
import time
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geometry_pca.loader import load_face_keypoints, iter_sample_ids
from geometry_pca.gpa import gpa_align, center_and_scale
from geometry_pca.fit import fit_encoder, save_encoder
from geometry_pca.pose_normalize import frontalize_dataset
from geometry_pca.canonical_face import canonical_template

OUT = "output"


def load_shapes(limit, conf_thresh=0.5):
    shapes = []
    kept = 0
    for sid in iter_sample_ids(limit):
        try:
            coords, conf = load_face_keypoints(sid, return_conf=True)
            if np.mean(conf) >= conf_thresh:
                shapes.append(center_and_scale(coords))
                kept += 1
        except FileNotFoundError:
            continue
    return np.stack(shapes), kept


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=70000)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--out", type=str, default=os.path.join(OUT, "encoder_production.npz"))
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()

    print(f"Loading up to {args.limit} samples...")
    shapes, kept = load_shapes(args.limit)
    print(f"Loaded {kept} valid faces in {time.time()-t0:.1f}s.")

    print("Frontalizing against canonical 300W 3D template (z_scale=1.0)...")
    template = canonical_template()  # anatomical depth at face value (z_scale=1.0)
    t1 = time.time()
    frontal = frontalize_dataset(shapes, template)
    print(f"Frontalized {len(frontal)} shapes in {time.time()-t1:.1f}s.")

    print("Light 2D GPA on frontalized shapes...")
    aligned, gpa_mean = gpa_align(frontal)

    print(f"Fitting PCA (k={args.k})...")
    encoder = fit_encoder(aligned, gpa_mean, k=args.k)
    encoder["canonical_template"] = template.astype(np.float32)  # persist for inference
    save_encoder(encoder, args.out)

    evr = encoder["explained_variance_ratio"]
    print(f"Retained {evr.sum()*100:.3f}% variance at k={args.k}.")

    # Scree
    cum = np.cumsum(evr)
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(cum)+1), cum, marker='o')
    plt.axhline(0.99, color='r', ls='--', label='99%')
    plt.title(f'[PRODUCTION 70k] Scree — {kept} faces')
    plt.xlabel('Components'); plt.ylabel('Cumulative EVR'); plt.grid(True); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "prod_scree.png")); plt.close()

    # Traversals C1-C5
    pca_mean = encoder["pca_mean"]
    for i in range(5):
        comp = encoder["components"][i]; sig = encoder["whiten_sigma"][i]; mu = encoder["whiten_mu"][i]
        plt.figure(figsize=(6, 6))
        for z, c in zip([-3, -1.5, 0, 1.5, 3], ['red','orange','gray','blue','purple']):
            sh = (pca_mean + (z*sig+mu)*comp).reshape(-1, 2)
            plt.scatter(sh[:, 0], -sh[:, 1], c=c, label=f'{z}σ', alpha=0.6, s=15)
        plt.title(f'[PROD 70k] Component {i+1} Traversal (±3σ)')
        plt.axis('equal'); plt.legend(); plt.grid(True, ls=':', alpha=0.5)
        plt.tight_layout(); plt.savefig(os.path.join(OUT, f"prod_traversal_C{i+1}.png")); plt.close()

    metrics = {
        "n_faces": int(kept),
        "k": args.k,
        "retained_variance": float(evr.sum()),
        "total_time_s": round(time.time()-t0, 1),
    }
    with open(os.path.join(OUT, "metrics_production.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"Production encoder saved to {args.out}")


if __name__ == "__main__":
    main()
