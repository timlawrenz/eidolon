"""Tests for SQLAlchemy ORM models."""
import pytest
from sqlalchemy import create_engine, inspect

from tools.hegre_dataset.models import Base, Persona, Set, Image


@pytest.fixture
def engine():
    """In-memory SQLite engine with our schema (PG-compatible via ORM)."""
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    return eng


class TestModelDefinitions:
    def test_table_names(self):
        """All three tables have correct names."""
        assert Persona.__tablename__ == "personas"
        assert Set.__tablename__ == "sets"
        assert Image.__tablename__ == "images"

    def test_persona_columns(self, engine):
        """Persona has id and name columns."""
        cols = {c["name"]: c for c in inspect(engine).get_columns("personas")}
        assert "id" in cols
        assert "name" in cols
        assert cols["name"]["nullable"] is False

    def test_sets_columns(self, engine):
        """Sets has id, persona_id, slug."""
        cols = {c["name"] for c in inspect(engine).get_columns("sets")}
        assert "id" in cols
        assert "persona_id" in cols
        assert "slug" in cols

    def test_images_columns(self, engine):
        """Images has all expected columns including distances and vector."""
        cols = {c["name"] for c in inspect(engine).get_columns("images")}
        for expected in ["id", "persona_id", "set_id", "image_path",
                         "source_image", "face_index", "status",
                         "reviewed_at", "zg_distance", "af_distance",
                         "auraface_embedding"]:
            assert expected in cols, f"Missing column: {expected}"

    def test_images_vector_column_type(self):
        """auraface_embedding is a PGVector column type."""
        col = Image.__table__.columns["auraface_embedding"]
        type_name = str(col.type)
        assert "VECTOR" in type_name.upper() or "vector" in type_name.lower()

    def test_fk_constraints(self, engine):
        """Foreign key constraints exist on sets.persona_id, images.persona_id, images.set_id."""
        fks = inspect(engine).get_foreign_keys("sets")
        assert any(fk["referred_table"] == "personas" for fk in fks)

        fks = inspect(engine).get_foreign_keys("images")
        assert any(fk["referred_table"] == "personas" for fk in fks)
        assert any(fk["referred_table"] == "sets" for fk in fks)

    def test_unique_constraint_on_sets(self, engine):
        """(persona_id, slug) is unique on sets."""
        uqs = inspect(engine).get_unique_constraints("sets")
        column_sets = [set(uq["column_names"]) for uq in uqs]
        assert {"persona_id", "slug"} in column_sets

    def test_unique_constraint_on_images(self, engine):
        """(persona_id, set_id, source_image, face_index) is unique on images."""
        uqs = inspect(engine).get_unique_constraints("images")
        column_sets = [set(uq["column_names"]) for uq in uqs]
        assert {"persona_id", "set_id", "source_image", "face_index"} in column_sets

    def test_default_values(self, engine):
        """face_index defaults to 1, status defaults to 'unreviewed'."""
        cols = {c["name"]: c for c in inspect(engine).get_columns("images")}
        assert cols["face_index"]["default"] == "1"
        assert cols["status"]["default"] == "'unreviewed'"
