#!/usr/bin/env python3
"""
Diagnostic: re-test z_d complementarity with metrics that can SEE cross-dimensional
structure, which trace-Fisher J = tr(S_B)/tr(S_W) cannot. Investigates whether the
x1.02 FAIL is a metric artifact (trace-J = weighted avg of component Js, blind to
complementary axes) rather than a true absence of identity signal.

NOT a gate rerun. Pure analysis on the existing data/zd_gate_*.npz vectors.
"""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from geometry_pca.fisher import fisher_ratios, restandardize

MODES = ["A", "A_prime", "C"]


def trace_J(Z, y):
    J, *_ = fisher_ratios(Z, y)
    return J


def multivariate_J(Z, y, shrink=0.1):
    """Proper LDA criterion J = tr(S_W^-1 S_B), with shrinkage regularization of S_W
    (needed: K can exceed smallest class size). Captures cross-dimensional separability."""
    classes = np.unique(y)
    K = Z.shape[1]
    mu = Z.mean(axis=0)
    Sw = np.zeros((K, K)); Sb = np.zeros((K, K))
    for c in classes:
        Zc = Z[y == c]
        muc = Zc.mean(axis=0)
        d = Zc - muc
        Sw += d.T @ d
        delta = (muc - mu).reshape(-1, 1)
        Sb += len(Zc) * (delta @ delta.T)
    Sw /= len(Z); Sb /= len(Z)
    # shrinkage: Sw_reg = (1-a)Sw + a*avg_diag*I  (Ledoit-Wolf style)
    avg_var = np.trace(Sw) / K
    Sw_reg = (1 - shrink) * Sw + shrink * avg_var * np.eye(K)
    return float(np.trace(np.linalg.solve(Sw_reg, Sb)))


def knn_identity_accuracy(Z, y, k=5):
    """Leave-one-out kNN identity retrieval accuracy. Operational test: given a
    person's vector, do its nearest neighbors share its identity? Z-scored first."""
    Zs = restandardize(Z)
    N = len(Zs)
    # pairwise squared distances
    sq = np.sum(Zs**2, axis=1)
    D = sq[:, None] + sq[None, :] - 2 * Zs @ Zs.T
    np.fill_diagonal(D, np.inf)  # exclude self
    correct = 0
    for i in range(N):
        nn = np.argsort(D[i])[:k]
        votes = y[nn]
        # majority vote
        vals, counts = np.unique(votes, return_counts=True)
        pred = vals[np.argmax(counts)]
        if pred == y[i]:
            correct += 1
    return correct / N


def main():
    print("Z_D COMPLEMENTARITY RE-TEST — metrics that see cross-dim structure")
    print("=" * 70)

    Xg = np.load(f"data/zd_gate_{MODES[0]}.npz")["X_g"]
    y = np.load(f"data/zd_gate_{MODES[0]}.npz")["y"]
    n_ids = len(np.unique(y))
    print(f"Gate: {len(Xg)} images, {n_ids} identities\n")

    # baselines on z_g alone
    Xg_rs = restandardize(Xg)
    base_trace = trace_J(Xg_rs, y)
    base_mvj = multivariate_J(Xg_rs, y)
    base_knn = knn_identity_accuracy(Xg, y)
    print(f"{'z_g BASELINE':<28s}  traceJ={base_trace:.4f}  mvJ={base_mvj:.3f}  kNN-acc={base_knn:.3f}")
    print("-" * 70)

    results = {"baseline": {"trace_J": base_trace, "mv_J": base_mvj, "knn_acc": base_knn,
                            "n_images": int(len(Xg)), "n_identities": int(n_ids)}, "modes": {}}

    for mode in MODES:
        Xd = np.load(f"data/zd_gate_{mode}.npz")["X_d"]
        Xd_rs = restandardize(Xd)
        cat = np.hstack([Xg_rs, Xd_rs])

        t = trace_J(cat, y)
        m = multivariate_J(cat, y)
        kacc = knn_identity_accuracy(np.hstack([Xg, Xd]), y)

        # z_d alone
        zd_t = trace_J(Xd_rs, y)
        zd_m = multivariate_J(Xd_rs, y)
        zd_knn = knn_identity_accuracy(Xd, y)

        print(f"\nMode {mode}:")
        print(f"  z_d ALONE                  traceJ={zd_t:.4f}  mvJ={zd_m:.3f}  kNN-acc={zd_knn:.3f}")
        print(f"  [z_g|z_d]                  traceJ={t:.4f}  mvJ={m:.3f}  kNN-acc={kacc:.3f}")
        print(f"  delta vs z_g baseline:     traceJ {t/base_trace:.2f}x   mvJ {m/base_mvj:.2f}x   kNN {kacc-base_knn:+.3f}")

        results["modes"][mode] = {
            "zd_alone": {"trace_J": zd_t, "mv_J": zd_m, "knn_acc": zd_knn},
            "cat": {"trace_J": t, "mv_J": m, "knn_acc": kacc},
            "ratios": {"trace_J": t/base_trace, "mv_J": m/base_mvj, "knn_delta": kacc-base_knn},
        }

    print("\n" + "=" * 70)
    print("INTERPRETATION:")
    print("  - If mvJ and kNN-acc rise with [z_g|z_d] but traceJ doesn't,")
    print("    the FAIL was a METRIC artifact: trace-J can't see complementarity.")
    print("  - If NONE of the three improve, z_d genuinely lacks identity signal.")

    with open("data/zd_complementarity_diagnostic.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print("\nSaved data/zd_complementarity_diagnostic.json")


if __name__ == "__main__":
    main()
