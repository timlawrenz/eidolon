#!/usr/bin/env python3
"""
Phase 1-R Step 2b: the REAL-IMAGE identity-separability gate.

Sweeps the canonical-template depth multiplier z_scale and ranks each by the
Fisher discriminant ratio J = S_B / S_W computed on real multi-pose hegre
images encoded through the pose-normalized geometry encoder.

Guards against degenerate collapse:
  - report S_B and S_W SEPARATELY (reject high-ratio-via-collapsing-S_B)
  - report the C1-specific ratio (the axis where yaw hid)
  - z_scale=0 (flat template == 2D GPA) is the NULL HYPOTHESIS

The ENCODER is fit on FFHQ (frontalized at the same z_scale); the GATE images
(hegre) are held-out and only encoded, never fit.
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geometry_pca.loader import load_face_keypoints, iter_sample_ids
from geometry_pca.gpa import gpa_align, center_and_scale, align_single
from geometry_pca.fit import fit_encoder
from geometry_pca.pose_normalize import frontalize, frontalize_dataset
from geometry_pca.canonical_face import canonical_template
from geometry_pca.fisher import fisher_ratios

FIT_LIMIT = 5000
K = 50
Z_SCALES = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]


def load_ffhq(limit):
    shapes = []
    for sid in iter_sample_ids(limit):
        try:
            c, conf = load_face_keypoints(sid, return_conf=True)
            if np.mean(conf) >= 0.5:
                shapes.append(center_and_scale(c))
        except FileNotFoundError:
            continue
    return np.stack(shapes)


def main():
    print(f"Fitting encoders on {FIT_LIMIT} FFHQ samples...")
    ffhq = load_ffhq(FIT_LIMIT)
    print(f"  {len(ffhq)} FFHQ faces loaded.")

    gate = np.load("data/hegre_gate_keypoints.npz")
    names_all = list(gate["names"])
    # Drop identities that failed visual verification:
    #   muriel  -> contained a male partner face + a blurred non-face crop
    #   natalia-a -> one ambiguous possible-intruder crop
    DROP = {"muriel", "natalia-a"}
    keep_mask = np.array([nm not in DROP for nm in [names_all[i] for i in gate["y"]]])
    X = gate["X"][keep_mask]
    y_raw = gate["y"][keep_mask]
    # remap labels to contiguous 0..K-1 over kept identities
    kept_ids = sorted(set(int(v) for v in y_raw))
    remap = {old: new for new, old in enumerate(kept_ids)}
    y = np.array([remap[int(v)] for v in y_raw])
    names = np.array([names_all[i] for i in kept_ids])
    print(f"  Gate: {len(X)} real images across {len(names)} CLEAN identities (dropped {sorted(DROP)}).")

    tpl0 = canonical_template()
    results = []
    for zs in Z_SCALES:
        tpl = tpl0.copy(); tpl[:, 2] *= zs
        tflip = tpl.copy(); tflip[:, 1] *= -1

        # Fit encoder on FFHQ frontalized at this z_scale
        frontal = frontalize_dataset(ffhq, tpl)
        aligned, gmean = gpa_align(frontal)
        enc = fit_encoder(aligned, gmean, k=K)
        pmean, comps, wmu, wsig = enc["pca_mean"], enc["components"], enc["whiten_mu"], enc["whiten_sigma"]

        # Encode the held-out hegre gate images
        def encode(shape2d):
            f = frontalize(tflip, center_and_scale(shape2d))
            a = align_single(f, gmean).reshape(-1)
            return ((a - pmean) @ comps.T - wmu) / wsig

        Z = np.stack([encode(s) for s in X])
        J, S_B, S_W, J_Ci, _, _ = fisher_ratios(Z, y)
        J1 = float(J_Ci[0])
        results.append({
            "z_scale": zs, "J_global": round(J, 4),
            "S_B": round(float(S_B), 4), "S_W": round(float(S_W), 4),
            "J_C1": round(J1, 4),
        })
        print(f"z_scale={zs:4.2f}  J={J:7.4f}  S_B={S_B:7.4f}  S_W={S_W:7.4f}  J_C1={J1:7.4f}")

    best = max(results, key=lambda r: r["J_global"])
    null = next(r for r in results if r["z_scale"] == 0.0)
    verdict = {
        "best_z_scale": float(best["z_scale"]),
        "best_J": float(best["J_global"]),
        "null_J_flat_2dgpa": float(null["J_global"]),
        "3d_beats_flat": bool(best["z_scale"] != 0.0 and best["J_global"] > null["J_global"] * 1.05),
        "results": results,
    }
    with open("data/gate_sweep_results.json", "w") as f:
        json.dump(verdict, f, indent=2)
    print("\n=== VERDICT ===")
    print(json.dumps({k: v for k, v in verdict.items() if k != "results"}, indent=2))


if __name__ == "__main__":
    main()
