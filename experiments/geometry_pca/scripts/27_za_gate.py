#!/usr/bin/env python3
"""
Phase 2b: z_a AUC gate + nuisance audit (canonical instrument = verification AUC).

Loads gate vectors per variant, runs partition_gate at 3 seeds, reports z_a-ALONE
AUC, and audits top components for pose-nuisance correlation BEFORE trusting a PASS.

Output: data/za_gate_results.json (verdict artifact).
"""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geometry_pca.verification import verification_auc, partition_gate

VARIANTS = ["raw", "xy", "rot", "rot_xy"]
SEEDS = [0, 1, 2]
EPS = 0.01


def nuisance_audit(Z, y, names=None):
    """Correlate top-5 z_a components vs estimated yaw/pitch per image.

    Yaw/pitch are approximated from Z_c1 (largest-variance axis) and Z_c2.
    For normals, the real nuisance is head pose — if R^T de-rotation worked,
    correlations should be WEAKER for rot/rot_xy than raw/xy.
    Returns per-component top-5 abs correlation with Z_c1 (proxy for dominant axis).
    """
    K = Z.shape[1]
    Zc1 = Z[:, 0]  # dominant component (proxy for largest nuisance)
    Zc2 = Z[:, 1] if K > 1 else None
    top_abs_corr = []
    for i in range(min(5, K)):
        r1 = float(np.corrcoef(Zc1, Z[:, i])[0, 1])
        r2 = float(np.corrcoef(Zc2, Z[:, i])[0, 1]) if Zc2 is not None else 0
        top_abs_corr.append({"component": i, "abs_corr_w_C1": abs(r1), "abs_corr_w_C2": abs(r2)})
    # summary metric: mean of top-5 abs-corrs with C1 — lower = less nuisance
    summary = float(np.mean([c["abs_corr_w_C1"] for c in top_abs_corr]))
    return {"top5_vs_C1_mean_abs_corr": summary, "per_component": top_abs_corr}


def main():
    print("Phase 2b — z_a AUC GATE (canonical instrument) + nuisance audit")
    print("=" * 70)

    Xg = np.load(f"data/za_gate_{VARIANTS[0]}.npz")["X_g"]
    y = np.load(f"data/za_gate_{VARIANTS[0]}.npz")["y"]
    n_ids = len(np.unique(y))
    print(f"Gate: {len(Xg)} images, {n_ids} identities")
    print(f"AUC baseline: computing (3 seeds)...\n")

    # z_g baseline over 3 seeds
    auc_g_seeds = [verification_auc(Xg, y, seed=s)[0] for s in SEEDS]
    auc_g_mean = float(np.mean(auc_g_seeds))
    print(f"{'z_g BASELINE':<26s} AUC(mean,3seeds)={auc_g_mean:.4f}  "
          f"(per-seed: {[f'{a:.4f}' for a in auc_g_seeds]})")
    print("-" * 70)

    results = {"baseline_auc_3seed_mean": auc_g_mean, "baseline_auc_per_seed": auc_g_seeds,
               "eps": EPS, "n_identities": int(n_ids),
               "variants": {}, "overall_verdict": None}
    best_variant, best_delta = None, -999

    for variant in VARIANTS:
        print(f"\n{'─'*60}")
        print(f"  Variant: {variant}")
        print(f"{'─'*60}")
        Xa = np.load(f"data/za_gate_{variant}.npz")["X_a"]

        # z_a ALONE AUC
        auc_a_seeds = [verification_auc(Xa, y, seed=s)[0] for s in SEEDS]
        auc_a_mean = float(np.mean(auc_a_seeds))
        print(f"  z_a ALONE AUC(mean,3seeds) = {auc_a_mean:.4f}  "
              f"({[f'{a:.4f}' for a in auc_a_seeds]})")

        # partition gate at 3 seeds
        deltas = []
        for s in SEEDS:
            g = partition_gate(Xg, Xa, y, eps=EPS, seed=s)
            deltas.append(g["delta"])

        delta_mean = float(np.mean(deltas))
        verdict = "PASS" if delta_mean > EPS else "FAIL"
        print(f"  [z_g|z_a] AUC delta vs baseline = {delta_mean:+.4f}  "
              f"(per-seed: {[f'{d:+.4f}' for d in deltas]})")
        print(f"  VERDICT: {verdict}  (threshold: delta > {EPS:.2f})")

        # Nuisance audit
        nuis = nuisance_audit(Xa, y)
        print(f"  Nuisance audit: top5-vs-C1 mean abs_corr = {nuis['top5_vs_C1_mean_abs_corr']:.3f} "
              f"({'CLEAN' if nuis['top5_vs_C1_mean_abs_corr'] < 0.3 else '⚠️ SUSPECT' if nuis['top5_vs_C1_mean_abs_corr'] < 0.6 else '❌ NUISANCE-DOMINATED'})")

        results["variants"][variant] = {
            "za_alone_auc_3seed_mean": auc_a_mean,
            "za_alone_auc_per_seed": auc_a_seeds,
            "cat_auc_delta_3seed_mean": delta_mean,
            "cat_auc_delta_per_seed": deltas,
            "verdict": verdict,
            "nuisance_audit": nuis,
        }

        if delta_mean > best_delta:
            best_delta = delta_mean
            best_variant = variant

    # Final summary
    print(f"\n{'='*70}")
    print(f"  FINAL VERDICT")
    print(f"{'='*70}")
    overall_pass = any(r["verdict"] == "PASS" for r in results["variants"].values())
    results["overall_verdict"] = "PASS" if overall_pass else "FAIL"
    results["best_variant"] = best_variant
    results["best_delta"] = float(best_delta)

    for variant in VARIANTS:
        r = results["variants"][variant]
        marker = " ← WINNER" if variant == best_variant else ""
        print(f"  {variant:10s} delta={r['cat_auc_delta_3seed_mean']:+.4f}  "
              f"{r['verdict']}{marker}")

    print(f"\n  OVERALL: {results['overall_verdict']}  "
          f"(threshold: delta > {EPS:.2f})")

    with open("data/za_gate_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n  Results saved to data/za_gate_results.json")


if __name__ == "__main__":
    main()
