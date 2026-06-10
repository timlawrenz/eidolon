#!/usr/bin/env python3
"""
Phase 2b: z_a AUC gate + nuisance audit (canonical instrument = verification AUC).

Reviewer fixes baked in:
- Nuisance audit correlates top-5 z_a components against ACTUAL per-image
  yaw/pitch (saved by extractor 26), not against the components themselves.
- Asserts all per-variant npz files share one extraction run_stamp before
  computing anything (alignment guard against partial regeneration).

Output: data/za_gate_results.json (verdict artifact).
"""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geometry_pca.verification import verification_auc, partition_gate

VARIANTS = ["raw", "xy", "rot", "rot_xy"]
SEEDS = [0, 1, 2]
EPS = 0.01


def nuisance_audit(X_a, yaw, pitch):
    """Correlate top-5 z_a components against ACTUAL estimated yaw/pitch.

    For normals, head pose is the principal nuisance (the analog of camera
    distance for depth). High |corr| of a top component with yaw/pitch means
    that component encodes pose, not identity. The rot/rot_xy variants should
    show systematically LOWER correlations than raw/xy if de-rotation works.
    """
    per_component = []
    for i in range(min(5, X_a.shape[1])):
        r_yaw = float(np.corrcoef(X_a[:, i], yaw)[0, 1])
        r_pitch = float(np.corrcoef(X_a[:, i], pitch)[0, 1])
        per_component.append({
            "component": i,
            "abs_corr_yaw": abs(r_yaw),
            "abs_corr_pitch": abs(r_pitch),
        })
    max_pose_corr = max(max(c["abs_corr_yaw"], c["abs_corr_pitch"]) for c in per_component)
    mean_pose_corr = float(np.mean([max(c["abs_corr_yaw"], c["abs_corr_pitch"])
                                    for c in per_component]))
    return {"max_pose_corr_top5": max_pose_corr,
            "mean_pose_corr_top5": mean_pose_corr,
            "per_component": per_component}


def audit_label(mean_corr):
    if mean_corr < 0.3:
        return "CLEAN"
    if mean_corr < 0.6:
        return "SUSPECT"
    return "NUISANCE-DOMINATED"


def main():
    print("Phase 2b — z_a AUC GATE (canonical instrument) + nuisance audit")
    print("=" * 70)

    # Load all variants; enforce single extraction run (alignment guard)
    data = {v: np.load(f"data/za_gate_{v}.npz") for v in VARIANTS}
    stamps = {v: str(data[v]["run_stamp"]) for v in VARIANTS}
    if len(set(stamps.values())) != 1:
        print("FATAL: per-variant gate files come from DIFFERENT extraction runs:")
        for v, s in stamps.items():
            print(f"  {v}: {s}")
        print("Re-run scripts/26_extract_za_gate.py (full, no --variant) first.")
        sys.exit(1)

    Xg = data[VARIANTS[0]]["X_g"]
    y = data[VARIANTS[0]]["y"]
    yaw = data[VARIANTS[0]]["yaw"]
    pitch = data[VARIANTS[0]]["pitch"]
    n_ids = len(np.unique(y))
    print(f"Gate: {len(Xg)} images, {n_ids} identities  (run {stamps[VARIANTS[0]]})")
    print(f"Pose spread: yaw std={np.degrees(yaw.std()):.1f}°  pitch std={np.degrees(pitch.std()):.1f}°\n")

    # z_g baseline over 3 seeds
    auc_g_seeds = [verification_auc(Xg, y, seed=s)[0] for s in SEEDS]
    auc_g_mean = float(np.mean(auc_g_seeds))
    print(f"{'z_g BASELINE':<26s} AUC(mean,3seeds)={auc_g_mean:.4f}  "
          f"(per-seed: {[f'{a:.4f}' for a in auc_g_seeds]})")
    print("-" * 70)

    results = {"baseline_auc_3seed_mean": auc_g_mean,
               "baseline_auc_per_seed": auc_g_seeds,
               "eps": EPS, "n_images": int(len(Xg)), "n_identities": int(n_ids),
               "run_stamp": stamps[VARIANTS[0]],
               "variants": {}, "overall_verdict": None}
    best_variant, best_delta = None, -999.0

    for variant in VARIANTS:
        print(f"\n{'─'*60}")
        print(f"  Variant: {variant}")
        print(f"{'─'*60}")
        Xa = data[variant]["X_a"]

        auc_a_seeds = [verification_auc(Xa, y, seed=s)[0] for s in SEEDS]
        auc_a_mean = float(np.mean(auc_a_seeds))
        print(f"  z_a ALONE AUC(mean,3seeds) = {auc_a_mean:.4f}  "
              f"({[f'{a:.4f}' for a in auc_a_seeds]})")

        deltas = [partition_gate(Xg, Xa, y, eps=EPS, seed=s)["delta"] for s in SEEDS]
        delta_mean = float(np.mean(deltas))
        verdict = "PASS" if delta_mean > EPS else "FAIL"
        print(f"  [z_g|z_a] AUC delta = {delta_mean:+.4f}  "
              f"(per-seed: {[f'{d:+.4f}' for d in deltas]})")
        print(f"  VERDICT: {verdict}  (threshold: delta > {EPS:.2f})")

        nuis = nuisance_audit(Xa, yaw, pitch)
        label = audit_label(nuis["mean_pose_corr_top5"])
        print(f"  Nuisance audit (top-5 vs ACTUAL yaw/pitch): "
              f"mean={nuis['mean_pose_corr_top5']:.3f} max={nuis['max_pose_corr_top5']:.3f} "
              f"-> {label}")

        results["variants"][variant] = {
            "za_alone_auc_3seed_mean": auc_a_mean,
            "za_alone_auc_per_seed": auc_a_seeds,
            "cat_auc_delta_3seed_mean": delta_mean,
            "cat_auc_delta_per_seed": deltas,
            "verdict": verdict,
            "nuisance_audit": nuis,
            "nuisance_label": label,
        }
        if delta_mean > best_delta:
            best_delta, best_variant = delta_mean, variant

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
              f"{r['verdict']}  nuisance={r['nuisance_label']}{marker}")
    print(f"\n  OVERALL: {results['overall_verdict']}  (threshold: delta > {EPS:.2f})")

    with open("data/za_gate_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n  Results saved to data/za_gate_results.json")


if __name__ == "__main__":
    main()
