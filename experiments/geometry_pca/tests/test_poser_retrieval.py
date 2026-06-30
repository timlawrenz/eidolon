import pytest
import numpy as np
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from geometry_pca.poser import PoserRetrievalHarness
from priors.models_torch import AdaLNResNetCrossAttn

@pytest.fixture
def dummy_artifacts(tmp_path):
    lda_path = tmp_path / "dummy_lda.npz"
    np.savez(lda_path, 
             lda_basis=np.random.randn(512, 64).astype(np.float32),
             pooled_mean=np.random.randn(512).astype(np.float32))
    
    model_kwargs = dict(d_in=64, d_out=64, d_hidden=32, n_blocks=2, d_cond=1024, n_heads=2, n_queries=2)
    model = AdaLNResNetCrossAttn(**model_kwargs)
    model_path = tmp_path / "dummy_arm_b.pt"
    torch.save({"model_state": model.state_dict(), "arm": "B"}, model_path)
    
    return model_path, lda_path, model_kwargs

def test_poser_retrieval_predict_pin(dummy_artifacts):
    model_path, lda_path, model_kwargs = dummy_artifacts
    
    harness = PoserRetrievalHarness(
        prior_model_path=model_path,
        lda_basis_path=lda_path,
        device="cpu",
        model_kwargs=model_kwargs
    )
    
    dummy_t5 = np.random.randn(1, 512, 1024).astype(np.float32)
    
    pin = harness.predict_pin(dummy_t5)
    
    assert pin.shape == (1, 64)
    assert isinstance(pin, np.ndarray)


def test_poser_retrieval_knn(dummy_artifacts):
    model_path, lda_path, model_kwargs = dummy_artifacts
    harness = PoserRetrievalHarness(
        prior_model_path=model_path,
        lda_basis_path=lda_path,
        device="cpu",
        model_kwargs=model_kwargs
    )
    
    # Create dummy database of 10 vectors
    db_vectors = np.random.randn(10, 64).astype(np.float32)
    db_labels = [f"id_{i}" for i in range(10)]
    
    harness.build_index(db_vectors, db_labels)
    
    # Query with exact match of the 3rd vector
    query_pin = db_vectors[2:3].copy()
    
    results = harness.retrieve(query_pin, k=3)
    
    assert len(results) == 3
    assert results[0]["label"] == "id_2"
    # distance should be very close to 0 (or similarity 1)
    assert np.isclose(results[0]["distance"], 0.0, atol=1e-5)


def test_poser_retrieval_manipulate(dummy_artifacts):
    model_path, lda_path, model_kwargs = dummy_artifacts
    harness = PoserRetrievalHarness(
        prior_model_path=model_path,
        lda_basis_path=lda_path,
        device="cpu",
        model_kwargs=model_kwargs
    )
    
    # Create dummy database of 10 vectors with a specific std on axis 0
    db_vectors = np.random.randn(100, 64).astype(np.float32)
    db_vectors[:, 0] = db_vectors[:, 0] * 5.0 # std=5.0 roughly
    harness.build_index(db_vectors, [str(i) for i in range(100)])
    
    # Exact computation of population std
    std_axis_0 = np.std(db_vectors[:, 0])
    
    query_pin = np.zeros((1, 64), dtype=np.float32)
    
    # Manipulate +1 sigma on axis 0
    new_pin = harness.manipulate_pin(query_pin, axis=0, step_sigma=1.0)
    
    assert new_pin[0, 0] == pytest.approx(std_axis_0, rel=1e-4)
    assert new_pin[0, 1] == 0.0 # untouched


def test_predict_pin_applies_mask(dummy_artifacts):
    """The Prior was trained on masked T5 (valid tokens only, capped, zero-padded).
    predict_pin MUST apply the same masking: padding tokens must not affect output.
    A pin computed with garbage in the masked-out positions must equal the pin
    computed with zeros in those positions."""
    model_path, lda_path, model_kwargs = dummy_artifacts
    harness = PoserRetrievalHarness(
        prior_model_path=model_path, lda_basis_path=lda_path,
        device="cpu", model_kwargs=model_kwargs)

    rng = np.random.RandomState(0)
    # 10 valid tokens, rest padding
    t5 = rng.randn(1, 512, 1024).astype(np.float32)
    mask = np.zeros((1, 512), dtype=bool)
    mask[0, :10] = True

    # Two versions: same valid tokens, DIFFERENT garbage in padding positions
    t5_a = t5.copy()
    t5_b = t5.copy()
    t5_b[0, 10:] = rng.randn(502, 1024).astype(np.float32) * 100.0  # wildly different padding

    # With correct masking, both must produce the SAME pin (padding ignored).
    torch.manual_seed(123)
    pin_a = harness.predict_pin(t5_a, mask=mask)
    torch.manual_seed(123)
    pin_b = harness.predict_pin(t5_b, mask=mask)

    assert pin_a.shape == (1, 64)
    np.testing.assert_allclose(pin_a, pin_b, atol=1e-5,
                               err_msg="Padding tokens leaked into the pin — mask not applied")


def test_poser_retrieval_metrics():
    # evaluate_recall is not implemented yet
    from geometry_pca.poser import evaluate_recall, generate_random_null
    
    # Dummy index of 5 vectors
    index_vectors = np.array([
        [1, 0, 0], # label A
        [0, 1, 0], # label A
        [0, 0, 1], # label B
        [1, 1, 0], # label C
        [0, 1, 1], # label C
    ], dtype=np.float32)
    index_labels = ["A", "A", "B", "C", "C"]
    
    # Query vectors
    queries = np.array([
        [0.9, 0.1, 0.0], # Should match index 0 (A)
        [0.1, 0.9, 0.9], # Should match index 4 (C)
    ], dtype=np.float32)
    query_labels = ["A", "B"] # second query is wrong class, should miss
    
    # Recall@1
    recall_1 = evaluate_recall(queries, index_vectors, query_labels, index_labels, k=1)
    # Query 0: closest is index 0 ("A"), matches query_label "A". Hit!
    # Query 1: closest is index 4 ("C"), mismatch with "B". Miss!
    assert recall_1 == 0.5
    
    # generate random null
    null_queries = generate_random_null(queries)
    assert null_queries.shape == queries.shape
    # norms should match
    assert np.allclose(np.linalg.norm(null_queries, axis=1), np.linalg.norm(queries, axis=1))

