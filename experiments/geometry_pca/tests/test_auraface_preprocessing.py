"""Tests for auraface_preprocessing module — clean_auraface() and project_to_lda()."""
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "experiments" / "geometry_pca"))
from geometry_pca.auraface_preprocessing import clean_auraface, project_to_lda, lda_to_full


# --- Shared fixtures ---

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def raw_vectors(rng):
    """Synthetic AuraFace vectors near the unit sphere (plausible raw embeddings)."""
    return rng.normal(0, 1, (10, 512)).astype(np.float64)


# --- clean_auraface() smoke tests ---

def test_clean_auraface_single_shape(raw_vectors):
    v = raw_vectors[0]
    out = clean_auraface(v)
    assert out.shape == (512,)


def test_clean_auraface_batch_shape(raw_vectors):
    out = clean_auraface(raw_vectors)
    assert out.shape == (10, 512)


def test_clean_auraface_renormalize(raw_vectors):
    out = clean_auraface(raw_vectors, renormalize=True)
    norms = np.linalg.norm(out, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)


def test_clean_auraface_no_renormalize(raw_vectors):
    out = clean_auraface(raw_vectors, renormalize=False)
    norms = np.linalg.norm(out, axis=1)
    # Without renormalize, norms should deviate from 1.0 (projections pull off sphere)
    assert not np.allclose(norms, 1.0, atol=1e-6)


def test_clean_auraface_yaw_removed(raw_vectors):
    from geometry_pca.auraface_preprocessing import _load_ref
    ref = _load_ref()
    yaw_dir = ref["yaw"]
    out = clean_auraface(raw_vectors)
    proj = np.abs(out @ yaw_dir)
    assert np.all(proj < 1e-6), f"yaw projection not zero: {proj.max()}"


def test_clean_auraface_pc1_removed(raw_vectors):
    from geometry_pca.auraface_preprocessing import _load_ref
    ref = _load_ref()
    pc1 = ref["pc1"]
    out = clean_auraface(raw_vectors)
    proj = np.abs(out @ pc1)
    assert np.all(proj < 0.01), f"pc1 projection too large: {proj.max()}"


# --- project_to_lda() tests ---

def test_project_to_lda_single_shape(raw_vectors):
    """GREEN: project_to_lda maps (512,) → (64,)"""
    v = raw_vectors[0]
    coords = project_to_lda(v)
    assert coords.shape == (64,), f"expected (64,) got {coords.shape}"


def test_project_to_lda_batch_shape(raw_vectors):
    """project_to_lda maps (N,512) → (N,64)"""
    coords = project_to_lda(raw_vectors)
    assert coords.shape == (10, 64)


def test_project_to_lda_finite(raw_vectors):
    """Output coordinates are finite."""
    coords = project_to_lda(raw_vectors)
    assert np.isfinite(coords).all()


def test_lda_to_full_shape(raw_vectors):
    """lda_to_full maps (64,) → (512,)"""
    coords = project_to_lda(raw_vectors[0])
    full = lda_to_full(coords)
    assert full.shape == (512,)


def test_lda_to_full_batch_shape(raw_vectors):
    """lda_to_full maps (N,64) → (N,512)"""
    coords = project_to_lda(raw_vectors)
    full = lda_to_full(coords)
    assert full.shape == (10, 512)


def test_lda_roundtrip_approximate(raw_vectors):
    """lda_to_full → project_to_lda → lda_to_full is idempotent (reconstruction invariant)."""
    v = raw_vectors[0]
    coords = project_to_lda(v)
    reconstructed = lda_to_full(coords)
    # Re-project → re-reconstruct should give the SAME reconstruction vector
    coords2 = project_to_lda(reconstructed)
    reconstructed2 = lda_to_full(coords2)
    assert np.allclose(reconstructed, reconstructed2, atol=1e-6)


def test_preprocessing_pipeline(raw_vectors):
    """Full pipeline: clean_auraface → project_to_lda produces valid coordinates."""
    v = raw_vectors[0]
    cleaned = clean_auraface(v)
    coords = project_to_lda(cleaned)
    assert coords.shape == (64,)
    assert np.isfinite(coords).all()

    # Verify PC1 and yaw are removed BEFORE LDA projection
    from geometry_pca.auraface_preprocessing import _load_ref, _load_lda
    ref = _load_ref()
    cleaned_norm = cleaned / (np.linalg.norm(cleaned) + 1e-12)
    yaw_proj = np.abs(cleaned_norm @ ref["yaw"])
    assert yaw_proj < 1e-6, f"yaw not removed before LDA: {yaw_proj}"
    pc1_proj = np.abs(cleaned_norm @ ref["pc1"])
    assert pc1_proj < 0.01, f"pc1 not removed before LDA: {pc1_proj}"
