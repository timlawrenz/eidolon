"""Fisher discriminant ratio — shared between the legacy gate sweep (07) and the
Phase 2 z_d gate (21)."""
import numpy as np


def fisher_ratios(Z, y):
    """Compute Fisher discriminant ratio of identity separability.

    J = S_B / S_W where S_B = mean between-identity scatter, S_W = mean within-identity
    scatter, both computed from the (N,K) encoded vectors Z and integer identity labels y.

    Also returns per-component J_Ci = S_B_i / S_W_i for every feature axis.

    Returns: (J_global, S_B, S_W, J_Ci, S_B_i, S_W_i)
      J_global: scalar, aggregate Fisher ratio
      S_B, S_W: scalars, global between/within scatter (normalised by N)
      J_Ci: (K,) per-component Fisher ratios
      S_B_i, S_W_i: (K,) per-component scatters
    """
    classes = np.unique(y)
    K = Z.shape[1]
    mu = Z.mean(axis=0)
    n_total = len(Z)

    S_W = 0.0
    S_B = 0.0
    S_W_i = np.zeros(K, dtype=np.float64)
    S_B_i = np.zeros(K, dtype=np.float64)

    for c in classes:
        Zc = Z[y == c]
        muc = Zc.mean(axis=0)
        n_c = len(Zc)
        S_W += np.sum((Zc - muc) ** 2)
        S_B += n_c * np.sum((muc - mu) ** 2)
        S_W_i += np.sum((Zc - muc) ** 2, axis=0)
        S_B_i += n_c * (muc - mu) ** 2

    S_W /= n_total
    S_B /= n_total
    S_W_i /= n_total
    S_B_i /= n_total

    J = S_B / S_W if S_W > 1e-12 else 0.0
    J_Ci = np.divide(S_B_i, S_W_i, out=np.zeros_like(S_B_i), where=S_W_i > 1e-12)
    return J, S_B, S_W, J_Ci, S_B_i, S_W_i


def restandardize(Z):
    """Z-score (N,K) array on its own per-component mean/std. Returns Z' with ~unit
    per-component std on this distribution, killing domain shift in the whitening."""
    mu = Z.mean(axis=0, keepdims=True)
    sigma = Z.std(axis=0, keepdims=True)
    return (Z - mu) / np.maximum(sigma, 1e-8)
