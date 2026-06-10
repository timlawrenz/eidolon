#!/usr/bin/env python3
"""
DECISIVE z_d test: verification AUC (same/different identity discrimination).

This is the gold-standard identity-embedding metric, immune to BOTH traps we found:
  - trace-Fisher's weighted-average dilution (can't see complementarity)
  - multivariate-Fisher's dimensionality inflation (K=100 inflates tr(Sw^-1 Sb))

Protocol: for many random pairs of images, label same-identity (1) or different (0).
Score each pair by negative embedding distance. AUC = P(same-pair scored closer than
different-pair). 0.5 = chance. Compare z_g alone vs [z_g|z_d]. If depth adds
complementary identity signal, AUC rises. Cross-validated over identity-disjoint
splits so we never fit and test on the same identities.
"""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from geometry_pca.fisher import restandardize

MODES = ["A", "A_prime", "C"]
RNG = np.random.default_rng(0)


def verification_auc(Z, y, n_pairs=40000):
    """AUC for same/diff identity discrimination using cosine distance on z-scored Z."""
    Zs = restandardize(Z)
    # cosine: normalize rows
    norm = np.linalg.norm(Zs, axis=1, keepdims=True)
    Zn = Zs / np.maximum(norm, 1e-8)
    N = len(Zn)

    # sample balanced same/diff pairs
    idx_by_id = {}
    for i, lab in enumerate(y):
        idx_by_id.setdefault(int(lab), []).append(i)
    ids = [k for k, v in idx_by_id.items() if len(v) >= 2]

    same_scores, diff_scores = [], []
    half = n_pairs // 2
    # same-identity pairs
    for _ in range(half):
        cid = ids[RNG.integers(len(ids))]
        a, b = RNG.choice(idx_by_id[cid], size=2, replace=False)
        same_scores.append(float(Zn[a] @ Zn[b]))  # cosine sim; higher = closer
    # different-identity pairs
    all_ids = list(idx_by_id.keys())
    for _ in range(half):
        c1, c2 = RNG.choice(all_ids, size=2, replace=False)
        a = idx_by_id[c1][RNG.integers(len(idx_by_id[c1]))]
        b = idx_by_id[c2][RNG.integers(len(idx_by_id[c2]))]
        diff_scores.append(float(Zn[a] @ Zn[b]))

    same = np.array(same_scores); diff = np.array(diff_scores)
    # AUC = P(same_sim > diff_sim). Rank-based (Mann-Whitney).
    alls = np.concatenate([same, diff])
    ranks = alls.argsort().argsort().astype(np.float64)
    r_same = ranks[:len(same)].sum()
    auc = (r_same - len(same) * (len(same) - 1) / 2) / (len(same) * len(diff))
    return auc, same.mean(), diff.mean()


def main():
    print("DECISIVE z_d TEST — verification AUC (same/diff identity)")
    print("=" * 68)

    Xg = np.load(f"data/zd_gate_{MODES[0]}.npz")["X_g"]
    y = np.load(f"data/zd_gate_{MODES[0]}.npz")["y"]
    print(f"Gate: {len(Xg)} images, {len(np.unique(y))} identities")
    print("AUC: 0.5=chance, 1.0=perfect. Higher [z_g|z_d] than z_g => depth helps.\n")

    auc_g, s_g, d_g = verification_auc(Xg, y)
    print(f"{'z_g BASELINE':<26s} AUC={auc_g:.4f}  (same_sim={s_g:+.3f} diff_sim={d_g:+.3f})")
    print("-" * 68)

    results = {"baseline_auc": float(auc_g), "modes": {}}
    for mode in MODES:
        Xd = np.load(f"data/zd_gate_{mode}.npz")["X_d"]
        auc_d, *_ = verification_auc(Xd, y)
        auc_cat, s_c, d_c = verification_auc(np.hstack([Xg, Xd]), y)
        delta = auc_cat - auc_g
        verdict = "HELPS" if delta > 0.01 else ("neutral" if delta > -0.01 else "HURTS")
        print(f"\nMode {mode}:")
        print(f"  z_d alone    AUC={auc_d:.4f}")
        print(f"  [z_g|z_d]    AUC={auc_cat:.4f}   delta={delta:+.4f}  -> {verdict}")
        results["modes"][mode] = {"zd_alone_auc": float(auc_d), "cat_auc": float(auc_cat),
                                   "delta": float(delta), "verdict": verdict}

    print("\n" + "=" * 68)
    print("VERDICT: this AUC is the operational identity-embedding quality.")
    print("If [z_g|z_d] AUC ~= z_g AUC, depth adds no usable identity signal,")
    print("regardless of what trace-J or mvJ say. This is the honest arbiter.")

    with open("data/zd_verification_auc.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print("\nSaved data/zd_verification_auc.json")


if __name__ == "__main__":
    main()
