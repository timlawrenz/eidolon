import numpy as np
import pytest
from geometry_pca.loader import load_face_keypoints, iter_sample_ids, load_matrix
from geometry_pca.constants import N_FACE_PTS

def test_iter_sample_ids():
    ids = list(iter_sample_ids(limit=5))
    assert len(ids) == 5
    assert ids == ["00000", "00001", "00002", "00003", "00004"]

def test_load_face_keypoints():
    # Test on a real sample from the NAS
    sample_id = "00000"
    kpts = load_face_keypoints(sample_id)
    
    # Check shape and dtype
    assert kpts.shape == (N_FACE_PTS, 2)
    assert kpts.dtype == np.float32
    
    # Check range (should be mostly in [-1, 1], allow slight epsilon just in case)
    assert np.all(kpts >= -1.1)
    assert np.all(kpts <= 1.1)

def test_load_matrix():
    # Load a tiny matrix
    limit = 3
    M = load_matrix(limit=limit, drop_low_conf=False)
    
    assert M.shape == (limit, N_FACE_PTS, 2)
    assert M.dtype == np.float32

    # Verify drop_low_conf works by running it (we might not know exact count, 
    # but it should be <= limit and shape should be correct)
    M_filtered = load_matrix(limit=limit, conf_threshold=0.1, drop_low_conf=True)
    assert M_filtered.shape[0] <= limit
    assert M_filtered.shape[1:] == (N_FACE_PTS, 2)
