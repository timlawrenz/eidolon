"""Tests for tools.hegre_dataset.dataset — PG-backed HegreDataset.

Tests use mocked SQLAlchemy to verify adapter behavior and loud-failure
guard without requiring a real PostgreSQL connection.
"""
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

from tools.hegre_dataset.dataset import (
    Artifact, Persona, Photo, HegreDataset, HAS_SQLALCHEMY,
    _adapt_sql, _DBConnection, _CursorWrapper,
)

# ── Adapter tests (no DB needed) ───────────────────────────────────────

class TestAdaptSql:
    def test_no_params(self):
        sql, params = _adapt_sql("SELECT 1", None)
        assert sql == "SELECT 1"
        assert params is None

    def test_single_param(self):
        sql, params = _adapt_sql("SELECT * FROM t WHERE x = ?", (5,))
        assert sql == "SELECT * FROM t WHERE x = :p0"
        assert params == {"p0": 5}

    def test_multiple_params(self):
        sql, params = _adapt_sql(
            "INSERT INTO t (a, b) VALUES (?, ?)", ("hello", 42)
        )
        assert sql == "INSERT INTO t (a, b) VALUES (:p0, :p1)"
        assert params == {"p0": "hello", "p1": 42}

    def test_scalar_param_wrapped_in_tuple(self):
        sql, params = _adapt_sql("SELECT * FROM t WHERE id = ?", 7)
        assert sql == "SELECT * FROM t WHERE id = :p0"
        assert params == {"p0": 7}

    def test_mixed_placeholders(self):
        sql, params = _adapt_sql(
            "UPDATE t SET x = ?, y = ? WHERE z = ?", (1, 2, 3)
        )
        assert sql == "UPDATE t SET x = :p0, y = :p1 WHERE z = :p2"
        assert params == {"p0": 1, "p1": 2, "p2": 3}


class TestMockedHegreDataset:
    """Test HegreDataset with a mocked SQLAlchemy engine."""

    @pytest.fixture
    def mock_engine(self):
        """Returns a MagicMock that simulates a SQLAlchemy engine."""
        engine = MagicMock()
        conn = MagicMock()
        engine.connect.return_value = conn
        return engine

    @pytest.fixture
    def ds(self, tmp_path, mock_engine):
        """HegreDataset with mocked engine, no review.db at root."""
        return HegreDataset(root=tmp_path, engine=mock_engine)

    def test_constructs_with_engine(self, ds, mock_engine):
        """HegreDataset accepts engine kwarg without raising."""
        assert ds._engine is mock_engine
        assert ds.root is not None

    def test_db_returns_connection(self, ds):
        """ds.db returns a _DBConnection wrapper."""
        conn = ds.db
        assert isinstance(conn, _DBConnection)

    def test_db_writable_same_as_db(self, ds):
        """db_writable is the same as db (PG handles concurrency)."""
        # Each call creates a new _DBConnection from the engine pool
        assert isinstance(ds.db_writable, _DBConnection)

    def test_loud_failure_when_review_db_exists(self, tmp_path):
        """HegreDataset raises if review.db file still exists."""
        (tmp_path / "review.db").write_text("stale")
        with pytest.raises(RuntimeError, match="review.db still exists"):
            HegreDataset(root=tmp_path, engine=MagicMock())

    @pytest.mark.skipif(not HAS_SQLALCHEMY, reason="SQLAlchemy not installed")
    def test_personas_queries_engine(self, ds, mock_engine):
        """ds.personas executes SELECT against the adapter."""
        # Set up mock cursor result — _CursorWrapper uses .all(), not .mappings()
        mock_result = MagicMock()
        mock_result.keys.return_value = ["id", "name"]
        mock_result.all.return_value = [(1, "anna-l")]
        ds.db._conn.execute.return_value = mock_result

        personas = ds.personas
        assert "anna-l" in personas
        assert personas["anna-l"].id == 1

    def test_stratum_dir(self, tmp_path, mock_engine):
        """stratum_dir returns root/stratum."""
        ds = HegreDataset(root=tmp_path, engine=mock_engine)
        assert ds.stratum_dir == tmp_path / "stratum"


# ── Artifact tests (unchanged from original) ─────────────────────────────

class TestArtifact:
    def test_artifact_is_ndarray(self):
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        a = Artifact(data, Path("/tmp/test.npy"))
        assert isinstance(a, np.ndarray)
        assert np.mean(a) == 2.0
        assert a.shape == (3,)

    def test_artifact_has_path(self):
        data = np.array([1.0, 2.0, 3.0])
        a = Artifact(data, Path("/tmp/test.npy"))
        assert a.path == Path("/tmp/test.npy")

    def test_artifact_slice_preserves_path(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        a = Artifact(data, Path("/tmp/test.npy"))
        row = a[0]
        assert isinstance(row, Artifact)
        assert row.path == a.path

    def test_artifact_dtype_preserved(self):
        data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        a = Artifact(data, Path("x.npy"))
        assert a.dtype == np.float64

    def test_artifact_numpy_save_load_roundtrip(self, tmp_path):
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        a = Artifact(data, tmp_path / "test.npy")
        np.save(str(a.path), a)
        loaded = np.load(str(a.path))
        np.testing.assert_array_equal(loaded, data)
        assert not isinstance(loaded, Artifact)


# ── Persona tests (unchanged) ───────────────────────────────────────────

class TestPersona:
    def test_persona_from_row(self):
        p = Persona(id=7, name="anna-l")
        assert p.id == 7
        assert p.name == "anna-l"
        assert repr(p) == "Persona(id=7, name='anna-l')"

    def test_persona_equality(self):
        p1 = Persona(id=7, name="anna-l")
        p2 = Persona(id=7, name="Anna")
        assert p1 == p2

    def test_persona_hash(self):
        p1 = Persona(id=7, name="anna-l")
        p2 = Persona(id=7, name="other")
        assert hash(p1) == hash(p2)
        assert len({p1, p2}) == 1

    def test_persona_lda_average_path(self):
        p = Persona(id=7, name="anna-l")
        assert p.lda_average_path(Path("/tmp/ds")) == Path("/tmp/ds/averages/anna-l.lda.npy")

    def test_persona_lda_average(self, tmp_path):
        avg_dir = tmp_path / "averages"
        avg_dir.mkdir()
        data = np.array([0.1, 0.2], dtype=np.float64)
        np.save(avg_dir / "anna-l.lda.npy", data)

        p = Persona(id=7, name="anna-l")
        result = p.lda_average(tmp_path)
        assert isinstance(result, Artifact)
        np.testing.assert_array_equal(result, data)
        assert result.path == avg_dir / "anna-l.lda.npy"

    def test_persona_lda_average_missing_raises(self, tmp_path):
        p = Persona(id=7, name="anna-l")
        with pytest.raises(FileNotFoundError):
            p.lda_average(tmp_path)


# ── Photo tests (unchanged) ─────────────────────────────────────────────

class TestPhoto:
    def test_photo_attrs(self):
        p = Photo(persona_name="anna-l", image_path="faces/anna-l/shoot1/img.jpg",
                  dataset_root=Path("/tmp/ds"))
        assert p.persona_name == "anna-l"
        assert p.image_path == "faces/anna-l/shoot1/img.jpg"

    def test_photo_auraface_path(self):
        p = Photo(persona_name="anna-l", image_path="faces/anna-l/shoot1/img.jpg",
                  dataset_root=Path("/tmp/ds"))
        assert p.auraface_path == Path("/tmp/ds/auraface/faces/anna-l/shoot1/img.npy")

    def test_photo_z_g_path(self):
        p = Photo(persona_name="anna-l", image_path="faces/anna-l/shoot1/img.jpg",
                  dataset_root=Path("/tmp/ds"))
        assert p.z_g_path == Path("/tmp/ds/zg/faces/anna-l/shoot1/img.npy")

    def test_photo_lda_path(self):
        p = Photo(persona_name="anna-l", image_path="faces/anna-l/shoot1/img.jpg",
                  dataset_root=Path("/tmp/ds"))
        assert p.lda_path == Path("/tmp/ds/lda/faces/anna-l/shoot1/img.npy")

    def test_photo_has_auraface_true(self, tmp_path):
        af = tmp_path / "auraface/faces/anna-l/shoot1"
        af.mkdir(parents=True)
        (af / "img.npy").write_bytes(b"fake")
        p = Photo(persona_name="anna-l", image_path="faces/anna-l/shoot1/img.jpg",
                  dataset_root=tmp_path)
        assert p.has_auraface is True

    def test_photo_has_auraface_false(self, tmp_path):
        p = Photo(persona_name="anna-l", image_path="faces/anna-l/shoot1/img.jpg",
                  dataset_root=tmp_path)
        assert p.has_auraface is False

    def test_photo_is_complete_all_present(self, tmp_path):
        for sub in ["auraface", "zg", "lda"]:
            d = tmp_path / sub / "faces/anna-l/shoot1"
            d.mkdir(parents=True)
            (d / "img.npy").write_bytes(b"fake")
        p = Photo(persona_name="anna-l", image_path="faces/anna-l/shoot1/img.jpg",
                  dataset_root=tmp_path)
        assert p.is_complete is True

    def test_photo_is_complete_missing_zg(self, tmp_path):
        for sub in ["auraface", "lda"]:
            d = tmp_path / sub / "faces/anna-l/shoot1"
            d.mkdir(parents=True)
            (d / "img.npy").write_bytes(b"fake")
        p = Photo(persona_name="anna-l", image_path="faces/anna-l/shoot1/img.jpg",
                  dataset_root=tmp_path)
        assert p.is_complete is False

    def test_photo_auraface_loads_artifact(self, tmp_path):
        data = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        af = tmp_path / "auraface/faces/anna-l/shoot1"
        af.mkdir(parents=True)
        np.save(af / "img.npy", data)

        p = Photo(persona_name="anna-l", image_path="faces/anna-l/shoot1/img.jpg",
                  dataset_root=tmp_path)
        result = p.auraface
        assert isinstance(result, Artifact)
        np.testing.assert_array_equal(result, data)
        assert result.path == af / "img.npy"

    def test_photo_auraface_missing_raises(self, tmp_path):
        p = Photo(persona_name="anna-l", image_path="faces/anna-l/shoot1/img.jpg",
                  dataset_root=tmp_path)
        with pytest.raises(FileNotFoundError):
            _ = p.auraface
