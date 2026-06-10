#!/usr/bin/env python3
"""
z_d verification-AUC gate (canonical instrument). Uses geometry_pca.verification.

This is the corrected partition gate that replaced the flawed trace-J gate (21).
For z_d it confirms the FAIL: depth adds no identity signal over z_g.
Reusable for z_a (Phase 2b) via geometry_pca.verification.partition_gate.
"""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from geometry_pca.verification import verification_auc, partition_gate

MODES = ["A", "A_prime", "C"]


def main():
    print("z_d VERIFICATION-AUC GATE (canonical instrument)")
    print("=" * 68)

    Xg = np.load(f"data/zd_gate_{MODES[0]}.npz")["X_g"]
    y = np.load(f"data/zd_gate_{MODES[0]}.npz")["y"]
    print(f"Gate: {len(Xg)} images, {len(np.unique(y))} identities")
    print("AUC: 0.5=chance, 1.0=perfect. Partition PASSES iff delta > eps.\n")

    auc_g, s_g, d_g = verification_auc(Xg, y)
    print(f"{'z_g BASELINE':<26s} AUC={auc_g:.4f}  (same_sim={s_g:+.3f} diff_sim={d_g:+.3f})")
    print("-" * 68)

    results = {"baseline_auc": float(auc_g), "eps": 0.01, "modes": {}}
    for mode in MODES:
        Xd = np.load(f"data/zd_gate_{mode}.npz")["X_d"]
        gate = partition_gate(Xg, Xd, y, eps=0.01)
        auc_d, *_ = verification_auc(Xd, y)
        print(f"\nMode {mode}:")
        print(f"  z_d alone    AUC={auc_d:.4f}")
        print(f"  [z_g|z_d]    AUC={gate['auc_concatenated']:.4f}   "
              f"delta={gate['delta']:+.4f}  -> {gate['verdict']}")
        results["modes"][mode] = {"zd_alone_auc": float(auc_d), **gate}

    overall = "PASS" if any(r["verdict"] == "PASS" for r in results["modes"].values()) else "FAIL"
    print(f"\n{'='*68}\nOVERALL: {overall}")
    results["overall_verdict"] = overall

    with open("data/zd_verification_auc.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print("Saved data/zd_verification_auc.json")


if __name__ == "__main__":
    main()
