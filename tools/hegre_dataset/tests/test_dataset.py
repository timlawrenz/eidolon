"""Tests for tools.hegre_dataset.dataset — Artifact, Photo, Persona, HegreDataset."""
import sqlite3
import pytest
import numpy as np
from pathlib import Path

from tools.hegre_dataset.dataset import Artifact, Persona, Photo, HegreDataset


# ── helpers ────────────────────────────────────────────────────────────

def _seed_tmp_db(db_path: Path):
    """Create a minimal review.db with schema and one persona."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS personas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL
        );
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            persona_id INTEGER NOT NULL REFERENCES personas(id),
            image_path TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'unreviewed'
        );
    """)
    conn.execute("INSERT INTO personas (id, name) VALUES (1, 'anna-l')")
    conn.execute(
        "INSERT INTO images (persona_id, image_path, status) "
        "VALUES (1, 'faces/anna-l/shoot1/img.jpg', 'approved')"
    )
    conn.commit()
    conn.close()


# ── Artifact ────────────────────────────────────────────────────────────

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

    def test_artifact_reduction_preserves_type(self):
        data = np.array([1.0, 2.0, 3.0])
        a = Artifact(data, Path("/tmp/test.npy"))
        result = np.mean(a)
        assert isinstance(result, Artifact)
        assert result.path == a.path
        assert result == 2.0

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


# ── Persona ─────────────────────────────────────────────────────────────

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


# ── Photo ───────────────────────────────────────────────────────────────

class TestPhoto:
    def test_photo_attrs(self):
        p = Photo(persona_name="anna-l", image_path="faces/anna-l/shoot1/img.jpg",
                  dataset_root=Path("/tmp/ds"))
        assert p.persona_name == "anna-l"
        assert p.image_path == "faces/anna-l/shoot1/img.jpg"
        assert p.dataset_root == Path("/tmp/ds")

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


# ── HegreDataset ────────────────────────────────────────────────────────

class TestHegreDataset:
    def test_constructs_with_valid_root(self, tmp_path):
        ds = HegreDataset(root=tmp_path)
        assert ds.root == tmp_path

    def test_db_connection(self, tmp_path):
        db_path = tmp_path / "review.db"
        _seed_tmp_db(db_path)
        ds = HegreDataset(root=tmp_path)
        conn = ds.db
        assert isinstance(conn, sqlite3.Connection)
        row = conn.execute("SELECT COUNT(*) FROM personas").fetchone()
        assert row[0] == 1

    def test_persona_by_name(self, tmp_path):
        _seed_tmp_db(tmp_path / "review.db")
        ds = HegreDataset(root=tmp_path)
        p = ds.persona("anna-l")
        assert isinstance(p, Persona)
        assert p.name == "anna-l"
        assert p.id == 1

    def test_persona_by_id(self, tmp_path):
        _seed_tmp_db(tmp_path / "review.db")
        ds = HegreDataset(root=tmp_path)
        p = ds.persona(1)
        assert p.name == "anna-l"
        assert p.id == 1

    def test_persona_not_found_raises(self, tmp_path):
        _seed_tmp_db(tmp_path / "review.db")
        ds = HegreDataset(root=tmp_path)
        with pytest.raises(KeyError, match="nobody"):
            ds.persona("nobody")

    def test_personas_property(self, tmp_path):
        _seed_tmp_db(tmp_path / "review.db")
        ds = HegreDataset(root=tmp_path)
        personas = ds.personas
        assert "anna-l" in personas
        assert personas["anna-l"].id == 1

    def test_photo_lookup(self, tmp_path):
        _seed_tmp_db(tmp_path / "review.db")
        photo = HegreDataset(root=tmp_path).photo(
            "anna-l", "faces/anna-l/shoot1/img.jpg"
        )
        assert isinstance(photo, Photo)
        assert photo.persona_name == "anna-l"
        assert photo.image_path == "faces/anna-l/shoot1/img.jpg"

    def test_photo_lookup_not_approved_raises(self, tmp_path):
        db_path = tmp_path / "review.db"
        _seed_tmp_db(db_path)
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            "INSERT INTO images (persona_id, image_path, status) "
            "VALUES (1, 'faces/anna-l/shoot1/bad.jpg', 'unreviewed')"
        )
        conn.commit()
        conn.close()
        ds = HegreDataset(root=tmp_path)
        with pytest.raises(ValueError, match="not approved"):
            ds.photo("anna-l", "faces/anna-l/shoot1/bad.jpg")

    def test_persona_photos(self, tmp_path):
        db_path = tmp_path / "review.db"
        _seed_tmp_db(db_path)
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            "INSERT INTO images (persona_id, image_path, status) "
            "VALUES (1, 'faces/anna-l/shoot2/b.jpg', 'approved')"
        )
        conn.execute(
            "INSERT INTO images (persona_id, image_path, status) "
            "VALUES (1, 'faces/anna-l/shoot2/c.jpg', 'unreviewed')"
        )
        conn.commit()
        conn.close()

        ds = HegreDataset(root=tmp_path)
        photos = ds.persona("anna-l").photos
        assert len(photos) == 2
        assert all(isinstance(ph, Photo) for ph in photos)
        paths = {ph.image_path for ph in photos}
        assert paths == {"faces/anna-l/shoot1/img.jpg", "faces/anna-l/shoot2/b.jpg"}
        assert "faces/anna-l/shoot2/c.jpg" not in paths

    def test_photo_count(self, tmp_path):
        _seed_tmp_db(tmp_path / "review.db")
        ds = HegreDataset(root=tmp_path)
        p = ds.persona("anna-l")
        assert p.photo_count == 1

    def test_z_g_centroid(self, tmp_path):
        db_path = tmp_path / "review.db"
        _seed_tmp_db(db_path)
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            "INSERT INTO images (persona_id, image_path, status) "
            "VALUES (1, 'faces/anna-l/s2/b.jpg', 'approved')"
        )
        conn.commit()
        conn.close()

        zg1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        zg2 = np.array([0.0, 2.0, 0.0], dtype=np.float32)
        for sub, arr in [("shoot1", zg1), ("s2", zg2)]:
            d = tmp_path / "zg/faces/anna-l" / sub
            d.mkdir(parents=True)
            np.save(d / ("img.npy" if sub == "shoot1" else "b.npy"), arr)

        ds = HegreDataset(root=tmp_path)
        centroid = ds.persona("anna-l").z_g_centroid
        expected = (zg1 + zg2) / 2.0
        np.testing.assert_array_equal(centroid, expected)

    def test_z_g_centroid_none_when_no_zg(self, tmp_path):
        _seed_tmp_db(tmp_path / "review.db")
        ds = HegreDataset(root=tmp_path)
        assert ds.persona("anna-l").z_g_centroid is None

    def test_auraface_centroid(self, tmp_path):
        _seed_tmp_db(tmp_path / "review.db")
        af1 = np.array([0.1, 0.2], dtype=np.float32)
        d = tmp_path / "auraface/faces/anna-l/shoot1"
        d.mkdir(parents=True)
        np.save(d / "img.npy", af1)

        ds = HegreDataset(root=tmp_path)
        centroid = ds.persona("anna-l").auraface_centroid
        np.testing.assert_array_equal(centroid, af1)

    def test_db_writable_can_insert(self, tmp_path):
        """HegreDataset.db_writable returns a connection that can write."""
        _seed_tmp_db(tmp_path / "review.db")
        ds = HegreDataset(root=tmp_path)
        conn = ds.db_writable
        conn.execute(
            "INSERT INTO images (persona_id, image_path, status) "
            "VALUES (1, 'faces/anna-l/new.jpg', 'approved')"
        )
        conn.commit()
        row = ds.db.execute("SELECT COUNT(*) FROM images").fetchone()
        assert row[0] == 2  # original + new

    def test_stratum_dir(self, tmp_path):
        """HegreDataset.stratum_dir returns the stratum output directory."""
        ds = HegreDataset(root=tmp_path)
        assert ds.stratum_dir == tmp_path / "stratum"
