"""HegreDataset — unified dataset access layer for hegre face data."""
import functools
import re
from pathlib import Path
from typing import Union

import numpy as np

try:
    from sqlalchemy import create_engine, text as _sa_text
    HAS_SQLALCHEMY = True
except ImportError:
    HAS_SQLALCHEMY = False

    def _sa_text(sql):
        raise ImportError(
            "SQLAlchemy is required for database access. "
            "Install with: pip install sqlalchemy"
        )


class Artifact(np.ndarray):
    """A numpy array that remembers its source file path."""

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

    def lda_average_path(self, dataset_root: Path) -> Path:
        return Path(dataset_root) / "averages" / f"{self.name}.lda.npy"

    def lda_average(self, dataset_root: Path) -> "Artifact":
        path = self.lda_average_path(dataset_root)
        if not path.exists():
            raise FileNotFoundError(f"LDA average not found: {path}")
        return Artifact(np.load(path), path)

    @functools.cached_property
    def photos(self) -> list["Photo"]:
        if self._dataset is None:
            raise RuntimeError("Persona is not bound to a dataset")
        rows = self._dataset.db.execute(
            "SELECT image_path FROM images "
            "WHERE persona_id = ? AND status = 'approved' "
            "ORDER BY image_path",
            (self.id,),
        ).fetchall()
        return [
            Photo(persona_name=self.name, image_path=r["image_path"],
                  dataset_root=self._dataset.root)
            for r in rows
        ]

    @property
    def photo_count(self) -> int:
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
        vectors = []
        for photo in self.photos:
            if photo.has_z_g:
                vectors.append(photo.z_g)
        if not vectors:
            return None
        return np.mean(np.stack(vectors), axis=0)

    @functools.cached_property
    def auraface_centroid(self) -> np.ndarray | None:
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
        return self.has_auraface and self.has_z_g and self.has_lda

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


# ═══════════════════════════════════════════════════════════════════════
# DB connection adapter — makes SQLAlchemy quack like sqlite3.Connection
# ═══════════════════════════════════════════════════════════════════════

_QMARK_RE = re.compile(r"\?")


def _adapt_sql(sql: str, params: tuple | None = None) -> tuple:
    """Convert sqlite3-style ? placeholders to SQLAlchemy :pN named params.

    Returns (adapted_sql, adapted_params_dict).
    """
    if params is None:
        return sql, None

    if not isinstance(params, tuple):
        params = (params,)

    # Replace ? with :p0, :p1, ...
    counter = 0
    def _replace(_match):
        nonlocal counter
        name = f"p{counter}"
        counter += 1
        return f":{name}"

    adapted_sql = _QMARK_RE.sub(_replace, sql)
    adapted_params = {f"p{i}": v for i, v in enumerate(params)}
    return adapted_sql, adapted_params


class _RowAdapter:
    """Wraps SQLAlchemy Row to support both dict-like (row["col"])
    and integer-index (row[0]) access, matching sqlite3.Row behavior."""

    __slots__ = ("_row", "_keys")

    def __init__(self, row, keys: list[str]):
        self._row = row
        self._keys = keys

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._row[key]
        return self._row[self._keys.index(key)]

    def __repr__(self):
        return repr(dict(zip(self._keys, self._row)))


class _CursorWrapper:
    """Wraps SQLAlchemy CursorResult to quack like sqlite3.Cursor.

    Returns Row objects that support both integer (row[0]) and
    name-based (row["column"]) access.
    """

    def __init__(self, result):
        self._result = result
        self._keys = list(result.keys())
        self._rows = None

    def fetchone(self):
        if self._rows is None:
            self._rows = [_RowAdapter(r, self._keys) for r in self._result.all()]
        if self._rows:
            return self._rows.pop(0)
        return None

    def fetchall(self):
        if self._rows is None:
            self._rows = [_RowAdapter(r, self._keys) for r in self._result.all()]
        rows = self._rows
        self._rows = []
        return rows


class _DBConnection:
    """Wraps a SQLAlchemy Connection to quack like sqlite3.Connection.

    Supports: execute(sql, params), executemany(sql, seq), commit(), close().
    Row results are dict-like (support row["column"] access).
    Close must be called to release the connection back to the pool.
    """

    def __init__(self, engine):
        self._conn = engine.connect()
        self.row_factory = None

    def execute(self, sql: str, params=None):
        sql, params = _adapt_sql(sql, params)
        result = self._conn.execute(_sa_text(sql), params or {})
        return _CursorWrapper(result)

    def executemany(self, sql: str, seq):
        for params in seq:
            adapted_sql, adapted_params = _adapt_sql(sql, params)
            self._conn.execute(_sa_text(adapted_sql), adapted_params or {})

    def commit(self):
        self._conn.commit()

    def close(self):
        self._conn.close()


# ═══════════════════════════════════════════════════════════════════════
# HegreDataset
# ═══════════════════════════════════════════════════════════════════════

class HegreDataset:
    """Unified database access for a Hegre face dataset.

    Backed by PostgreSQL (via SQLAlchemy). Set EIDOLON_DB_URL in environment
    or configure config/database.yml with the connection string.
    """

    def __init__(self, root: Path, *, engine=None):
        self.root = Path(root).resolve()
        self._db: _DBConnection | None = None
        self._personas: dict[str, Persona] | None = None

        # Loud-failure guard: old SQLite file must be renamed before switching
        if (self.root / "review.db").exists():
            raise RuntimeError(
                "review.db still exists at dataset root. "
                "This project now uses PostgreSQL. Rename review.db to "
                "review.db.old if you have verified the migration, then restart."
            )

        if engine is not None:
            self._engine = engine
        else:
            if not HAS_SQLALCHEMY:
                raise ImportError(
                    "SQLAlchemy is required. Install with: pip install sqlalchemy"
                )
            from .config import database_url
            url = database_url()
            self._engine = create_engine(url, pool_size=5, max_overflow=10)

    @property
    def db(self) -> _DBConnection:
        """Database connection (reads and writes — PG handles concurrency).

        Connection is cached per HegreDataset instance. Call close() to
        release back to the pool, or let it live for the instance lifetime.
        """
        if self._db is None:
            self._db = _DBConnection(self._engine)
        return self._db

    @property
    def db_writable(self) -> _DBConnection:
        """Database connection for mutations. Same as db on PostgreSQL."""
        return self.db

    @property
    def personas(self) -> dict[str, Persona]:
        if self._personas is None:
            rows = self.db.execute("SELECT id, name FROM personas").fetchall()
            self._personas = {
                r["name"]: Persona(id=r["id"], name=r["name"], dataset=self)
                for r in rows
            }
        return self._personas

    @property
    def stratum_dir(self) -> Path:
        return self.root / "stratum"

    def persona(self, identifier: int | str) -> Persona:
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
