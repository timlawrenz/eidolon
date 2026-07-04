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

    def __init__(self, *, id: int, name: str):
        self.id = id
        self.name = name

    def __repr__(self):
        return f"Persona(id={self.id}, name={self.name!r})"

    def __eq__(self, other):
        if not isinstance(other, Persona):
            return NotImplemented
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)
