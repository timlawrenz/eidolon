"""Tests for the verification-AUC identity metric (canonical partition gate)."""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_auc_chance_on_random():
    """Random embeddings with random labels => AUC ~ 0.5 (chance)."""
    from geometry_pca.verification import verification_auc
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(600, 20))
    y = rng.integers(0, 30, size=600)
    auc, _, _ = verification_auc(Z, y, n_pairs=20000, seed=1)
    assert 0.45 < auc < 0.55, f"random data should be ~chance, got {auc:.3f}"


def test_auc_high_on_separable():
    """Embeddings clustered tightly by identity => AUC near 1.0."""
    from geometry_pca.verification import verification_auc
    rng = np.random.default_rng(1)
    n_ids, per = 30, 20
    centers = rng.normal(scale=10.0, size=(n_ids, 20))  # well-separated centers
    Z, y = [], []
    for c in range(n_ids):
        Z.append(centers[c] + rng.normal(scale=0.1, size=(per, 20)))  # tight clusters
        y += [c] * per
    Z = np.vstack(Z); y = np.array(y)
    auc, same, diff = verification_auc(Z, y, n_pairs=20000, seed=2)
    assert auc > 0.95, f"tight clusters should be highly separable, got {auc:.3f}"
    assert same > diff, "same-identity cosine sim should exceed different-identity"


def test_auc_deterministic_with_seed():
    """Same seed => identical AUC (reproducibility)."""
    from geometry_pca.verification import verification_auc
    rng = np.random.default_rng(2)
    Z = rng.normal(size=(400, 15)); y = rng.integers(0, 20, size=400)
    a1, _, _ = verification_auc(Z, y, n_pairs=10000, seed=42)
    a2, _, _ = verification_auc(Z, y, n_pairs=10000, seed=42)
    assert a1 == a2, "AUC must be deterministic for a fixed seed"
