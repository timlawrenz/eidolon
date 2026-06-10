"""
Verification-AUC identity metric — the CANONICAL partition gate for Eidolon.

Replaces trace-Fisher J = tr(S_B)/tr(S_W), which is mathematically a weighted
average of component Js for concatenated vectors (J_cat <= max(J_zg, J_zd)) and is
therefore blind to complementarity — it tests *replacement*, not *addition*. See
docs/02_EXPERIMENTS_AND_RESULTS.md [Metric fix].

Verification AUC = P(a same-identity pair is scored more similar than a
different-identity pair). 0.5 = chance, 1.0 = perfect. It is scale-invariant,
threshold-independent, and immune to the dimensionality inflation that makes
multivariate-J tr(S_W^-1 S_B) untrustworthy at high K.

A partition z_x earns its place iff AUC([z_g | ... | z_x]) > AUC(baseline) + eps.
"""
import numpy as np
from geometry_pca.fisher import restandardize


def verification_auc(Z, y, n_pairs=40000, seed=0):
    """Same/different-identity verification AUC via cosine similarity.

    Args:
        Z: (N, K) embedding matrix
        y: (N,) integer identity labels
        n_pairs: total sampled pairs (half same-id, half different-id)
        seed: RNG seed for reproducible pair sampling

    Returns:
        (auc, mean_same_sim, mean_diff_sim)
        auc: Mann-Whitney AUC, P(same_sim > diff_sim). 0.5 = chance.
    """
    rng = np.random.default_rng(seed)

    # z-score then L2-normalize rows -> cosine similarity = dot product
    Zs = restandardize(Z)
    Zn = Zs / np.maximum(np.linalg.norm(Zs, axis=1, keepdims=True), 1e-8)

    idx_by_id = {}
    for i, lab in enumerate(y):
        idx_by_id.setdefault(int(lab), []).append(i)
    multi_ids = [k for k, v in idx_by_id.items() if len(v) >= 2]
    all_ids = list(idx_by_id.keys())
    if len(multi_ids) == 0 or len(all_ids) < 2:
        raise ValueError("need >=1 identity with >=2 samples and >=2 identities total")

    half = n_pairs // 2
    same_scores = np.empty(half)
    diff_scores = np.empty(half)

    for j in range(half):
        cid = multi_ids[rng.integers(len(multi_ids))]
        a, b = rng.choice(idx_by_id[cid], size=2, replace=False)
        same_scores[j] = Zn[a] @ Zn[b]

    for j in range(half):
        c1, c2 = rng.choice(all_ids, size=2, replace=False)
        a = idx_by_id[c1][rng.integers(len(idx_by_id[c1]))]
        b = idx_by_id[c2][rng.integers(len(idx_by_id[c2]))]
        diff_scores[j] = Zn[a] @ Zn[b]

    # Mann-Whitney AUC = P(same_sim > diff_sim)
    alls = np.concatenate([same_scores, diff_scores])
    ranks = alls.argsort().argsort().astype(np.float64)
    r_same = ranks[:half].sum()
    auc = (r_same - half * (half - 1) / 2) / (half * half)
    return float(auc), float(same_scores.mean()), float(diff_scores.mean())


def partition_gate(X_g, X_extra, y, eps=0.01, n_pairs=40000, seed=0):
    """Evaluate whether a candidate partition X_extra adds identity signal over X_g.

    Returns dict with baseline AUC, concatenated AUC, delta, and PASS/FAIL verdict.
    A partition passes iff AUC([z_g | z_extra]) > AUC(z_g) + eps.
    """
    auc_base, _, _ = verification_auc(X_g, y, n_pairs=n_pairs, seed=seed)
    cat = np.hstack([X_g, X_extra])
    auc_cat, _, _ = verification_auc(cat, y, n_pairs=n_pairs, seed=seed)
    delta = auc_cat - auc_base
    return {
        "auc_baseline": auc_base,
        "auc_concatenated": auc_cat,
        "delta": delta,
        "eps": eps,
        "verdict": "PASS" if delta > eps else "FAIL",
    }
