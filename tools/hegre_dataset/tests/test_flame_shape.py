import numpy as np
import pytest
import sqlite3
import torch
from unittest.mock import MagicMock, patch
from pathlib import Path

try:
    from tools.hegre_dataset.review.flame_projector import extract_canonical_shape
except ImportError:
    extract_canonical_shape = None

try:
    from tools.hegre_dataset.review.flame_projector import get_smirk_model
except ImportError:
    get_smirk_model = None


@pytest.fixture(autouse=True)
def _reset_smirk_cache():
    """Reset the module-level SmirkEncoder singleton between tests."""
    import tools.hegre_dataset.review.flame_projector as fp
    fp._smirk_model = None
    fp._smirk_checkpoint_path = None
    fp._smirk_device_str = None


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


def _setup_flame_test_tree(tmp_path, persona="anna", img_names=("img1", "img2")):
    """Create the directory tree that extract_canonical_shape expects."""
    # Image files
    (tmp_path / persona).mkdir(parents=True, exist_ok=True)
    for name in img_names:
        (tmp_path / persona / f"{name}.jpg").write_bytes(b"")

    # Stratum tree with pose.npy files
    stratum_dir = tmp_path / "stratum" / persona
    for name in img_names:
        pose_dir = stratum_dir / name
        pose_dir.mkdir(parents=True)
        # DWPose format: 133 keypoints, 3 columns (x, y, confidence)
        pose = np.zeros((133, 3), dtype=np.float32)
        for i in range(36, 42):   # left eye
            pose[i] = [-0.15, 0.1, 1.0]
        for i in range(42, 48):   # right eye
            pose[i] = [0.15, 0.1, 1.0]
        pose[30] = [0.0, 0.0, 1.0]           # nose tip
        pose[0] = [-0.25, -0.2, 1.0]          # jaw corner
        pose[16] = [0.25, -0.2, 1.0]          # jaw corner
        np.save(pose_dir / "pose.npy", pose)


# ──────────────────────────────────────────────────────────
# Model singleton tests
# ──────────────────────────────────────────────────────────

@pytest.mark.skipif(get_smirk_model is None, reason="get_smirk_model not implemented yet")
def test_get_smirk_model_returns_same_instance(tmp_path):
    """Calling get_smirk_model twice with the same checkpoint returns the same object."""
    ckpt_a = tmp_path / "ckpt_a.pt"
    ckpt_a.write_bytes(b"")

    # Mock SmirkEncoder so we don't need CUDA / timm in test env
    class FakeEncoder:
        def __init__(self, *args, **kwargs):
            pass
        def to(self, device):
            return self
        def eval(self):
            return self
        def load_state_dict(self, sd):
            pass

    with patch("tools.hegre_dataset.review.flame_projector.SmirkEncoder", FakeEncoder):
        with patch("tools.hegre_dataset.review.flame_projector.torch.load", return_value={}):
            model1 = get_smirk_model(ckpt_a)
            model2 = get_smirk_model(ckpt_a)

    assert model1 is model2, "get_smirk_model must return the same cached instance"


@pytest.mark.skipif(get_smirk_model is None, reason="get_smirk_model not implemented yet")
def test_get_smirk_model_different_checkpoint_returns_new_model(tmp_path):
    """A different checkpoint path returns a different model instance."""
    ckpt_a = tmp_path / "ckpt_a.pt"
    ckpt_b = tmp_path / "ckpt_b.pt"
    ckpt_a.write_bytes(b"")
    ckpt_b.write_bytes(b"")

    class FakeEncoder:
        def __init__(self, *args, **kwargs):
            pass
        def to(self, device):
            return self
        def eval(self):
            return self
        def load_state_dict(self, sd):
            pass

    with patch("tools.hegre_dataset.review.flame_projector.SmirkEncoder", FakeEncoder):
        with patch("tools.hegre_dataset.review.flame_projector.torch.load", return_value={}):
            model_a = get_smirk_model(ckpt_a)
            model_b = get_smirk_model(ckpt_b)

    assert model_a is not model_b, (
        "Different checkpoint paths must return different model instances"
    )


@pytest.mark.skipif(extract_canonical_shape is None, reason="Not implemented yet")
def test_extract_canonical_shape_reuses_cached_model(mock_db, tmp_path):
    """Two calls to extract_canonical_shape construct SmirkEncoder only once."""
    _setup_flame_test_tree(tmp_path)

    call_count = {"count": 0}

    class CountedMockSmirkEncoder:
        def __init__(self, *args, **kwargs):
            call_count["count"] += 1
            self.call_idx = 0

        def to(self, device):
            return self

        def eval(self):
            return self

        def load_state_dict(self, sd):
            pass

        def __call__(self, x):
            B = x.shape[0]
            ret = torch.ones((B, 300), dtype=torch.float32)
            if self.call_idx > 0:
                ret = -torch.ones((B, 300), dtype=torch.float32)
            self.call_idx += 1
            return {"shape_params": ret}

    ckpt_dir = tmp_path / "experiments" / "flame_spike" / "smirk" / "pretrained_models"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "SMIRK_em1.pt").write_bytes(b"")

    with patch("tools.hegre_dataset.review.flame_projector.SmirkEncoder", CountedMockSmirkEncoder):
        with patch("tools.hegre_dataset.review.flame_projector.torch.load", return_value={}):
            with patch("tools.hegre_dataset.review.flame_projector.crop_for_smirk",
                       return_value=torch.zeros((1, 3, 224, 224))):
                with patch("tools.hegre_dataset.review.flame_projector.cv2.imread",
                           return_value=np.zeros((300, 300, 3), dtype=np.uint8)):

                    extract_canonical_shape(
                        ds=mock_ds,
                        persona_name="anna"
                    )

                    first_count = call_count["count"]

                    extract_canonical_shape(
                        ds=mock_ds,
                        persona_name="anna"
                    )

                    assert call_count["count"] == first_count, (
                        f"SmirkEncoder constructed {call_count['count']} times "
                        f"(first call count: {first_count}); should equal 1 after singleton"
                    )


@pytest.mark.skipif(extract_canonical_shape is None, reason="Not implemented yet")
def test_extract_canonical_shape_averages_correctly(mock_db, tmp_path):
    _setup_flame_test_tree(tmp_path)

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
            B = x.shape[0]
            ret = torch.ones((B, 300), dtype=torch.float32)
            if self.called:
                ret = -torch.ones((B, 300), dtype=torch.float32)
            self.called = True
            return {"shape_params": ret}

    with patch("tools.hegre_dataset.review.flame_projector.SmirkEncoder", MockSmirkEncoder):
        with patch("tools.hegre_dataset.review.flame_projector.torch.load", return_value={}):
            with patch("tools.hegre_dataset.review.flame_projector.crop_for_smirk",
                       return_value=torch.zeros((1, 3, 224, 224))):
                with patch("tools.hegre_dataset.review.flame_projector.cv2.imread",
                           return_value=np.zeros((300, 300, 3), dtype=np.uint8)):

                    avg_shape = extract_canonical_shape(
                        ds=mock_ds,
                        persona_name="anna"
                    )

                    assert avg_shape.shape == (300,)
                    assert np.allclose(avg_shape, 0.0, atol=1e-5), "Shape was not correctly averaged!"
