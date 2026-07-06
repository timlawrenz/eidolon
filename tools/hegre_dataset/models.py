"""SQLAlchemy ORM models for the Hegre review database.

These models define the canonical schema. Alembic auto-generates migrations
from them. The HegreDataset class uses these via SQLAlchemy engine connections.

All tables, columns, constraints, and indexes are defined here.
Runtime-added columns (zg_distance, af_distance) from the old SQLite schema
are included as proper columns with explicit types.
"""

from sqlalchemy import (
    Column, Integer, String, Float, DateTime, ForeignKey, UniqueConstraint, Index
)
from sqlalchemy.orm import declarative_base, relationship
from pgvector.sqlalchemy import Vector

Base = declarative_base()


class Persona(Base):
    """A persona (identity) in the Hegre dataset."""
    __tablename__ = "personas"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, unique=True, nullable=False)

    sets = relationship("Set", back_populates="persona")
    images = relationship("Image", back_populates="persona")


class Set(Base):
    """A photo shoot belonging to a persona."""
    __tablename__ = "sets"
    __table_args__ = (UniqueConstraint("persona_id", "slug"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    persona_id = Column(
        Integer, ForeignKey("personas.id"), nullable=False
    )
    slug = Column(String, nullable=False)

    persona = relationship("Persona", back_populates="sets")
    images = relationship("Image", back_populates="set")


class Image(Base):
    """A single extracted face crop from a photo shoot."""
    __tablename__ = "images"
    __table_args__ = (
        UniqueConstraint("persona_id", "set_id", "source_image", "face_index"),
        Index("idx_images_status", "status"),
        Index("idx_images_persona", "persona_id"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    persona_id = Column(
        Integer, ForeignKey("personas.id"), nullable=False
    )
    set_id = Column(
        Integer, ForeignKey("sets.id"), nullable=False
    )
    image_path = Column(String, nullable=False)
    source_image = Column(String, nullable=False)
    face_index = Column(Integer, default=1)
    status = Column(String, default="unreviewed")
    reviewed_at = Column(DateTime, nullable=True)

    # Geometry/identity distance metrics
    zg_distance = Column(Float, nullable=True)
    af_distance = Column(Float, nullable=True)

    # pgvector: native vector similarity for AuraFace embeddings
    auraface_embedding = Column(Vector(512), nullable=True)

    persona = relationship("Persona", back_populates="images")
    set = relationship("Set", back_populates="images")
