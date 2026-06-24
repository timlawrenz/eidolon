import numpy as np
import pytest
import sqlite3
import torch
from unittest.mock import MagicMock, patch

try:
    from tools.hegre_dataset.review.flame_projector import extract_canonical_shape
except ImportError:
    extract_canonical_shape = None

@pytest.fixture
def mock_db(tmp_path):
    db_path = tmp_path / "review.db"
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("CREATE TABLE personas (id INTEGER PRIMARY KEY, name TEXT)")
    c.execute("CREATE TABLE images (id INTEGER PRIMARY KEY, persona_id INTEGER, image_path TEXT, status TEXT)")
    
    # Insert Persona
    c.execute("INSERT INTO personas (id, name) VALUES (1, 'anna')")
    
    # Insert Images (2 approved, 1 tainted)
    c.execute("INSERT INTO images (persona_id, image_path, status) VALUES (1, 'anna/img1.jpg', 'approved')")
    c.execute("INSERT INTO images (persona_id, image_path, status) VALUES (1, 'anna/img2.jpg', 'approved')")
    c.execute("INSERT INTO images (persona_id, image_path, status) VALUES (1, 'anna/img3.jpg', 'tainted:unusable')")
    
    conn.commit()
    conn.close()
    return db_path

@pytest.mark.skipif(extract_canonical_shape is None, reason="Not implemented yet")
def test_extract_canonical_shape_averages_correctly(mock_db, tmp_path):
    # Create mock dummy images
    (tmp_path / "anna").mkdir()
    
    # Create mock SMIRK encoder that returns predictable shapes
    # Shape of outputs for SMIRK: dict with 'shape_params' (B, 300)
    class MockSmirkEncoder:
        def __init__(self, *args, **kwargs):
            self.called = False
            
        def to(self, device):
            return self
        def eval(self):
            return self
        def load_state_dict(self, sd):
            pass
        def __call__(self, x):
            # Return a deterministic shape vector based on batch size
            B = x.shape[0]
            
            # First image gets vector of 1s, second gets vector of -1s
            # So average should be EXACTLY 0.
            ret = torch.ones((B, 300), dtype=torch.float32)
            if self.called:
                ret = -torch.ones((B, 300), dtype=torch.float32)
            self.called = True
                
            return {"shape_params": ret}

    # Patch the SmirkEncoder instantiation to use our mock
    with patch("tools.hegre_dataset.review.flame_projector.SmirkEncoder", MockSmirkEncoder):
        with patch("tools.hegre_dataset.review.flame_projector.torch.load", return_value={}):
            # Mock the crop_for_smirk to just return a dummy 224x224 tensor
            with patch("tools.hegre_dataset.review.flame_projector.crop_for_smirk", return_value=torch.zeros((1, 3, 224, 224))):
                
                avg_shape = extract_canonical_shape(
                    db_path=mock_db,
                    dataset_root=tmp_path,
                    persona_name="anna"
                )
                
                # We provided 2 approved images. 
                # Mock returns +1s for the first, -1s for the second.
                # Average should be 0s.
                assert avg_shape.shape == (300,)
                assert np.allclose(avg_shape, 0.0, atol=1e-5), "Shape was not correctly averaged!"
