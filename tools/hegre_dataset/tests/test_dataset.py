"""Tests for tools.hegre_dataset.dataset — Artifact, Photo, Persona, HegreDataset."""
import numpy as np
from pathlib import Path

from tools.hegre_dataset.dataset import Artifact, Persona


class TestArtifact:
    def test_artifact_is_ndarray(self):
        """Artifact should behave as a numpy array."""
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        path = Path("/tmp/test.npy")
        a = Artifact(data, path)
        assert isinstance(a, np.ndarray)
        assert np.mean(a) == 2.0
        assert a.shape == (3,)

    def test_artifact_has_path(self):
        """Artifact carries its source path."""
        data = np.array([1.0, 2.0, 3.0])
        path = Path("/tmp/test.npy")
        a = Artifact(data, path)
        assert a.path == path

    def test_artifact_slice_preserves_path(self):
        """Slicing preserves Artifact type with source path (numpy subclass behavior)."""
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        a = Artifact(data, Path("/tmp/test.npy"))
        row = a[0]
        assert isinstance(row, np.ndarray)
        assert isinstance(row, Artifact)
        assert row.path == a.path

    def test_artifact_reduction_preserves_type(self):
        """np.mean returns Artifact (numpy subclass preservation)."""
        data = np.array([1.0, 2.0, 3.0])
        a = Artifact(data, Path("/tmp/test.npy"))
        result = np.mean(a)
        assert isinstance(result, Artifact)
        assert result.path == a.path
        assert result == 2.0

    def test_artifact_dtype_preserved(self):
        """dtype of the original array is preserved."""
        data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        a = Artifact(data, Path("x.npy"))
        assert a.dtype == np.float64

    def test_artifact_numpy_save_load_roundtrip(self, tmp_path):
        """np.save + np.load preserves the data, but reload is plain ndarray."""
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        a = Artifact(data, tmp_path / "test.npy")
        np.save(str(a.path), a)
        loaded = np.load(str(a.path))
        np.testing.assert_array_equal(loaded, data)
        assert not isinstance(loaded, Artifact)


class TestPersona:
    def test_persona_from_row(self):
        """Persona constructed from DB row has id and name."""
        p = Persona(id=7, name="anna-l")
        assert p.id == 7
        assert p.name == "anna-l"
        assert repr(p) == "Persona(id=7, name='anna-l')"

    def test_persona_equality(self):
        """Two Personas with same id are equal."""
        p1 = Persona(id=7, name="anna-l")
        p2 = Persona(id=7, name="Anna")
        assert p1 == p2

    def test_persona_hash(self):
        """Personas hash by id."""
        p1 = Persona(id=7, name="anna-l")
        p2 = Persona(id=7, name="other")
        assert hash(p1) == hash(p2)
        assert len({p1, p2}) == 1  # set dedup
