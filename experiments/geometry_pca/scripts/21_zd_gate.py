#!/usr/bin/env python3
"""
Phase 2: TRAP-AWARE z_d Fisher gate harness.

⚠️ DEPRECATED AS A PASS/FAIL GATE (2026-06-10). The trace-Fisher ratio
J = tr(S_B)/tr(S_W) is a WEIGHTED AVERAGE of component Js for concatenated
vectors, so J_cat <= max(J_zg, J_zd) — it is BLIND to complementarity and tests
*replacement*, not *addition*. The canonical partition gate is now verification
AUC: see geometry_pca/verification.py and scripts/23_zd_verification_auc.py.
This script is retained ONLY as a legacy diagnostic (per-component J_Ci, ablation).

Loads the gate vectors (z_g, z_d) per normalization mode, re-standardizes z_d on the
hegre gate distribution itself (domain-shift fix), reports per-component Fisher J_Ci
to separate identity from nuisance, runs top-component ablation, and applies the
(now-deprecated) trace-J gate: J([z_g | z_d]) > J(z_g) × 1.15.

Output: data/zd_gate_results.json (legacy trace-J verdict artifact).
"""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geometry_pca.fisher import fisher_ratios, restandardize

MODES = ["A", "A_prime", "C"]
SKIP_BY_ID = {}  # populated from the first .npz loaded


def load_gate(mode):
    """Load gate data for one mode. Returns X_g, X_d, y, names."""
    data = np.load(f"data/zd_gate_{mode}.npz")
    global SKIP_BY_ID
    SKIP_BY_ID = json.loads(str(data.get("skip_by_id", "{}")))
    return data["X_g"], data["X_d"], data["y"], data["names"]


def concat_gate(X_g, X_d):
    """Concatenate z_g and z_d into (N, 100)."""
    return np.hstack([X_g, X_d])


def pairwise_J_table(J_Ci_zg, J_Ci_zd):
    """Print side-by-side per-component J for z_g vs z_d."""
    print(f"\n{'comp':>6s}  {'J(z_g_i)':>10s}  {'J(z_d_i)':>10s}   notes")
    for i in range(min(10, len(J_Ci_zg))):
        note = ""
        if i == 0:
            note = "  <- C1 (suspected nuisance: camera-distance for C, pitch for A)"
        elif i == 1:
            note = "  <- C2"
        print(f"  C{i+1:02d}   {J_Ci_zg[i]:10.4f}  {J_Ci_zd[i]:10.4f}{note}")
    print("  ...")


def run_mode(mode, X_g, X_d_raw, y):
    """Full trap-aware gate for one normalization mode."""
    print(f"\n{'='*70}")
    print(f"  MODE: {mode}")
    print(f"{'='*70}")

    # ── 1. z_g baseline ─────────────────────────────────────
    J_zg, S_B_zg, S_W_zg, J_Ci_zg, S_B_i_zg, S_W_i_zg = fisher_ratios(X_g, y)
    print(f"\n  z_g BASELINE: J={J_zg:.4f}  S_B={S_B_zg:.4f}  S_W={S_W_zg:.4f}")
    print(f"  J_C1(z_g)={J_Ci_zg[0]:.4f}  J_C2(z_g)={J_Ci_zg[1]:.4f}")

    # ── 2. z_d raw (FFHQ-fit whitening, no shift correction) ─
    J_zd_raw, S_B_zd_raw, S_W_zd_raw, J_Ci_zd_raw, S_B_i_zd_raw, S_W_i_zd_raw = fisher_ratios(X_d_raw, y)
    print(f"\n  z_d RAW (FFHQ whitening, domain-shifted): J={J_zd_raw:.4f}  S_B={S_B_zd_raw:.4f}  S_W={S_W_zd_raw:.4f}")
    J_cat_raw, S_B_cat_raw, S_W_cat_raw, _, _, _ = fisher_ratios(concat_gate(X_g, X_d_raw), y)
    ratio_raw = J_cat_raw / max(J_zg, 1e-12)
    print(f"  J([z_g|z_d]) raw: {J_cat_raw:.4f}  (ratio vs. baseline: {ratio_raw:.2f}x)")

    # ── 3. z_d RE-STANDARDIZED on the gate distribution (domain-shift fix) ──
    X_d_rs = restandardize(X_d_raw)
    J_zd_rs, S_B_zd_rs, S_W_zd_rs, J_Ci_zd_rs, S_B_i_zd_rs, S_W_i_zd_rs = fisher_ratios(X_d_rs, y)
    print(f"\n  z_d RE-STANDARDIZED (gate-distribution whiten): J={J_zd_rs:.4f}  S_B={S_B_zd_rs:.4f}  S_W={S_W_zd_rs:.4f}")
    J_cat_rs, S_B_cat_rs, S_W_cat_rs, _, _, _ = fisher_ratios(concat_gate(X_g, X_d_rs), y)
    ratio_rs = J_cat_rs / max(J_zg, 1e-12)
    print(f"  J([z_g|z_d]) re-std: {J_cat_rs:.4f}  (ratio vs. baseline: {ratio_rs:.2f}x)")

    # per-component joint J_Ci
    _, _, _, J_Ci_joint, _, _ = fisher_ratios(concat_gate(X_g, X_d_rs), y)
    print(f"\n  Per-component J_Ci (z_g | z_d re-std, k=0..9 shown):")

    # ── 4. Top-component ablation ─────────────────────────────
    print(f"\n  --- TOP-COMPONENT ABLATION ---")
    # Full re-standardized z_d
    full_J, full_SB, full_SW = J_cat_rs, S_B_cat_rs, S_W_cat_rs

    # Drop C1 only
    X_d_noC1 = X_d_rs[:, 1:]  # drop component 0
    J_noC1, SB_noC1, SW_noC1, _, _, _ = fisher_ratios(concat_gate(X_g, X_d_noC1), y)
    ratio_noC1 = J_noC1 / max(J_zg, 1e-12)

    # Drop C1 + C2
    X_d_noC12 = X_d_rs[:, 2:]  # drop components 0,1
    J_noC12, SB_noC12, SW_noC12, _, _, _ = fisher_ratios(concat_gate(X_g, X_d_noC12), y)
    ratio_noC12 = J_noC12 / max(J_zg, 1e-12)

    print(f"  full z_d:         J={full_J:.4f}  {ratio_rs:.2f}x baseline")
    print(f"  minus C1:         J={J_noC1:.4f}  {ratio_noC1:.2f}x  (S_B={SB_noC1:.4f} S_W={SW_noC1:.4f})")
    print(f"  minus C1,C2:      J={J_noC12:.4f}  {ratio_noC12:.2f}x  (S_B={SB_noC12:.4f} S_W={SW_noC12:.4f})")

    # ── 5. Decision ──────────────────────────────────────────
    GATE = 1.15
    variants = [
        ("full z_d (re-std)", full_J, ratio_rs),
        ("z_d minus C1", J_noC1, ratio_noC1),
        ("z_d minus C1,C2", J_noC12, ratio_noC12),
    ]
    best_name, best_J, best_ratio = max(variants, key=lambda v: v[1])

    passed = best_ratio >= GATE
    verdict = "PASS" if passed else "FAIL"

    print(f"\n  {'─'*50}")
    print(f"  GATE: J > J(z_g) × {GATE}  →  best={best_name} J={best_J:.4f} ({best_ratio:.2f}x)")
    print(f"  VERDICT (mode {mode}): {verdict}")
    print(f"  {'─'*50}")

    return {
        "mode": mode,
        "verdict": verdict,
        "J_zg": float(J_zg),
        "S_B_zg": float(S_B_zg), "S_W_zg": float(S_W_zg),
        "J_Ci_zg": J_Ci_zg[:10].tolist(),
        "z_d_raw": {"J": float(J_zd_raw), "S_B": float(S_B_zd_raw), "S_W": float(S_W_zd_raw),
                     "J_cat": float(J_cat_raw), "ratio": float(ratio_raw)},
        "z_d_restandardized": {"J": float(J_zd_rs), "S_B": float(S_B_zd_rs), "S_W": float(S_W_zd_rs),
                                "J_cat": float(J_cat_rs), "ratio": float(ratio_rs),
                                "J_Ci": J_Ci_zd_rs[:10].tolist()},
        "ablation": {
            "full": {"J": float(full_J), "ratio": float(ratio_rs),
                     "S_B": float(full_SB), "S_W": float(full_SW)},
            "minus_C1": {"J": float(J_noC1), "ratio": float(ratio_noC1),
                         "S_B": float(SB_noC1), "S_W": float(SW_noC1)},
            "minus_C1C2": {"J": float(J_noC12), "ratio": float(ratio_noC12),
                           "S_B": float(SB_noC12), "S_W": float(SW_noC12)},
        },
        "best_variant": best_name,
        "best_J": float(best_J),
        "best_ratio": float(best_ratio),
        "n_images": int(len(X_g)),
        "n_identities": int(len(np.unique(y))),
    }


def main():
    print("Phase 2 — TRAP-AWARE Fisher Gate")
    print("="*70)

    X_g, _, y, names = load_gate(MODES[0])
    n_imgs, n_ids = len(X_g), len(np.unique(y))
    print(f"Gate set: {n_imgs} images, {n_ids} identities")
    print(f"z_g baseline (shared across all modes): computing...")

    all_results = {}
    best_mode, best_J, best_ratio = None, 0, 0

    for mode in MODES:
        _, X_d_raw, _, _ = load_gate(mode)
        res = run_mode(mode, X_g, X_d_raw, y)
        all_results[mode] = res
        if res["best_ratio"] > best_ratio:
            best_ratio = res["best_ratio"]; best_J = res["best_J"]; best_mode = mode

    # ── Final summary ────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print(f"  FINAL VERDICT")
    print(f"{'='*70}")
    GATE = 1.15
    for mode in MODES:
        r = all_results[mode]
        marker = " ← WINNER (mode selected)" if mode == best_mode else ""
        print(f"  {mode:10s}: best={r['best_variant']:20s}  J={r['best_J']:.4f}  {r['best_ratio']:.2f}x  {r['verdict']}{marker}")

    overall_pass = any(r["verdict"] == "PASS" for r in all_results.values())
    print(f"\n  OVERALL: {'PASS' if overall_pass else 'FAIL'}  "
          f"(threshold: J > J(z_g) × {GATE})")

    # ── Save ──────────────────────────────────────────────────
    output = {
        "gate_threshold": GATE,
        "overall_verdict": "PASS" if overall_pass else "FAIL",
        "winning_mode": best_mode,
        "J_zg_baseline": float(all_results[MODES[0]]["J_zg"]),
        "n_images": n_imgs,
        "n_identities": n_ids,
        "reaon_method": "restandardize_z_d_on_gate_distribution",
        "modes": all_results,
    }
    os.makedirs("data", exist_ok=True)
    with open("data/zd_gate_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to data/zd_gate_results.json")


if __name__ == "__main__":
    main()
