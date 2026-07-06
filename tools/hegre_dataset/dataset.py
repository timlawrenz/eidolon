"""HegreDataset — unified dataset access layer for hegre face data."""
import sqlite3
import functools
from pathlib import Path
from typing import Union

import numpy as np


class Artifact(np.ndarray):
    """A numpy array that remembers its source file path.

    Subclasses np.ndarray so all numpy operations (mean, dot, slice, etc.)
    work transparently.  The .path attribute carries the file this array was
    loaded from (or will be saved to).
    """

    def __new__(cls, data: np.ndarray, path: Path):
        obj = data.view(cls)
        obj.path = Path(path)
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.path = getattr(obj, "path", None)


class Persona:
    """A persona (identity) in the Hegre dataset."""

    def __init__(self, *, id: int, name: str, dataset: "HegreDataset | None" = None):
        self.id = id
        self.name = name
        self._dataset = dataset

    def __repr__(self):
        return f"Persona(id={self.id}, name={self.name!r})"

    def __eq__(self, other):
        if not isinstance(other, Persona):
            return NotImplemented
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)

    # ── per-persona aggregates ──

    def lda_average_path(self, dataset_root: Path) -> Path:
        """Path to this persona's LDA identity average."""
        return Path(dataset_root) / "averages" / f"{self.name}.lda.npy"

    def lda_average(self, dataset_root: Path) -> "Artifact":
        """Load this persona's LDA identity average from disk."""
        path = self.lda_average_path(dataset_root)
        if not path.exists():
            raise FileNotFoundError(f"LDA average not found: {path}")
        return Artifact(np.load(path), path)

    @functools.cached_property
    def photos(self) -> list["Photo"]:
        """All approved photos for this persona."""
        if self._dataset is None:
            raise RuntimeError("Persona is not bound to a dataset")
        rows = self._dataset.db.execute(
            "SELECT image_path FROM images "
            "WHERE persona_id = ? AND status = 'approved' "
            "ORDER BY image_path",
            (self.id,),
        ).fetchall()
        return [
            Photo(
                persona_name=self.name,
                image_path=r["image_path"],
                dataset_root=self._dataset.root,
            )
            for r in rows
        ]

    @property
    def photo_count(self) -> int:
        """Number of approved photos for this persona."""
        if self._dataset is None:
            raise RuntimeError("Persona is not bound to a dataset")
        row = self._dataset.db.execute(
            "SELECT COUNT(*) AS n FROM images "
            "WHERE persona_id = ? AND status = 'approved'",
            (self.id,),
        ).fetchone()
        return row["n"]

    @functools.cached_property
    def z_g_centroid(self) -> np.ndarray | None:
        """Approved-image centroid of z_g vectors, or None if no z_g available."""
        vectors = []
        for photo in self.photos:
            if photo.has_z_g:
                vectors.append(photo.z_g)
        if not vectors:
            return None
        return np.mean(np.stack(vectors), axis=0)

    @functools.cached_property
    def auraface_centroid(self) -> np.ndarray | None:
        """Approved-image centroid of AuraFace vectors, or None if no auraface."""
        vectors = []
        for photo in self.photos:
            if photo.has_auraface:
                vectors.append(photo.auraface)
        if not vectors:
            return None
        return np.mean(np.stack(vectors), axis=0)


class Photo:
    """A single approved image in the dataset."""

    def __init__(self, *, persona_name: str, image_path: str, dataset_root: Path):
        self.persona_name = persona_name
        self.image_path = image_path
        self.dataset_root = Path(dataset_root)

    def _artifact_path(self, prefix: str) -> Path:
        """Construct path to an eidolon artifact .npy file.

        Converts self.image_path (e.g. 'faces/p/shoot/img.jpg') to
        '{prefix}/faces/p/shoot/img.npy'.
        """
        rel = Path(self.image_path).with_suffix(".npy")
        return self.dataset_root / prefix / rel

    @property
    def auraface_path(self) -> Path:
        return self._artifact_path("auraface")

    @property
    def z_g_path(self) -> Path:
        return self._artifact_path("zg")

    @property
    def lda_path(self) -> Path:
        return self._artifact_path("lda")

    @property
    def has_auraface(self) -> bool:
        return self.auraface_path.exists()

    @property
    def has_z_g(self) -> bool:
        return self.z_g_path.exists()

    @property
    def has_lda(self) -> bool:
        return self.lda_path.exists()

    @property
    def is_complete(self) -> bool:
        """All expected eidolon artifacts exist."""
        return self.has_auraface and self.has_z_g and self.has_lda

    # ── lazy artifact loading ──

    @functools.cached_property
    def auraface(self) -> Artifact:
        path = self.auraface_path
        if not path.exists():
            raise FileNotFoundError(f"AuraFace artifact not found: {path}")
        return Artifact(np.load(path), path)

    @functools.cached_property
    def z_g(self) -> Artifact:
        path = self.z_g_path
        if not path.exists():
            raise FileNotFoundError(f"z_g artifact not found: {path}")
        return Artifact(np.load(path), path)

    @functools.cached_property
    def lda(self) -> Artifact:
        path = self.lda_path
        if not path.exists():
            raise FileNotFoundError(f"LDA artifact not found: {path}")
        return Artifact(np.load(path), path)


class HegreDataset:
    """Unified read-only access to a Hegre face dataset (v1 directory layout)."""

    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self._db: sqlite3.Connection | None = None
        self._personas: dict[str, Persona] | None = None

    @property
    def db(self) -> sqlite3.Connection:
        """Read-only connection to review.db with WAL mode enabled."""
        if self._db is None:
            db_path = self.root / "review.db"
            self._db = sqlite3.connect(
                f"file:{db_path}?mode=ro", uri=True
            )
            self._db.row_factory = sqlite3.Row
            try:
                self._db.execute("PRAGMA journal_mode=WAL")
            except sqlite3.OperationalError:
                pass  # already WAL (set by writer) or read-only
        return self._db

    @property
    def db_writable(self) -> sqlite3.Connection:
        """Writable connection to review.db (for mutations like approve/taint)."""
        db_path = self.root / "review.db"
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    @property
    def personas(self) -> dict[str, Persona]:
        """All personas in the dataset, keyed by name."""
        if self._personas is None:
            rows = self.db.execute("SELECT id, name FROM personas").fetchall()
            self._personas = {
                r["name"]: Persona(id=r["id"], name=r["name"], dataset=self)
                for r in rows
            }
        return self._personas

    @property
    def stratum_dir(self) -> Path:
        """Stratum output directory."""
        return self.root / "stratum"

    def persona(self, identifier: int | str) -> Persona:
        """Look up a persona by name or database ID."""
        if isinstance(identifier, int):
            row = self.db.execute(
                "SELECT id, name FROM personas WHERE id = ?", (identifier,)
            ).fetchone()
            if row is None:
                raise KeyError(f"No persona with id={identifier}")
            return Persona(id=row["id"], name=row["name"], dataset=self)
        else:
            if self._personas is None:
                _ = self.personas
            p = self._personas.get(identifier)
            if p is None:
                raise KeyError(f"No persona named '{identifier}'")
            return Persona(id=p.id, name=p.name, dataset=self)

    def photo(self, persona: str, image_path: str) -> Photo:
        """Look up a single approved photo by persona name and relative image path.

        Raises:
            ValueError: if the image is not approved for this persona.
        """
        row = self.db.execute(
            "SELECT status FROM images i "
            "JOIN personas p ON i.persona_id = p.id "
            "WHERE p.name = ? AND i.image_path = ?",
            (persona, image_path),
        ).fetchone()
        if row is None:
            raise ValueError(
                f"Image not found: persona={persona}, path={image_path}"
            )
        if row["status"] != "approved":
            raise ValueError(
                f"Image '{image_path}' is not approved (status={row['status']})"
            )
        return Photo(
            persona_name=persona, image_path=image_path, dataset_root=self.root
        )
