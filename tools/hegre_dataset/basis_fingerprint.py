"""
Basis fingerprint guard — refuse to run on silently-inconsistent LDA projections.

WHY THIS EXISTS
---------------
The AuraFace-LDA identity vector stored in every sample dir is meaningless
without the basis it was projected through. The basis was refit on 2026-07-23
(pre-refit artifacts preserved as `*.bak-20260720`). Any dataset directory
projected BEFORE that refit carries coordinates in a different basis
(different directions AND ~440x different magnitude).

This is not hypothetical: `ffhq/stratum` was left on the pre-refit basis, and
because `prx-tg/production/data_stratum.py` loads `auraface_lda.npy` directly
for the `eidolon` adapter, every Eidolon arm whose `stratum_dirs` included FFHQ
trained on a 64-d identity slot carrying two incompatible encodings. Nothing
errored. That is the failure mode this guard removes.

HOW IT WORKS
------------
A dataset directory is *stamped* with a `BASIS_FINGERPRINT.json` recording the
sha256 of the basis artifacts it was projected through, plus the projection
convention. Loaders call `assert_basis(dataset_dir)`; a stamp that is missing,
or that disagrees with the current basis, raises `BasisMismatch`.

Deliberately fail-loud: a missing stamp is an error, not a warning. Silent
degradation is what produced the mixed-basis corpus generation and the FFHQ
mixed-basis arms.

TWO DIFFERENT FINGERPRINTS — do not confuse them
  * `basis_fingerprint` (this module): sha256 over the BASIS ARTIFACTS
    (`auraface_lda.npz` + `auraface_preprocess.npz`). Describes the projection.
  * `lda_basis_fingerprint` in a corpus `_manifest.json`: sha256 over that
    corpus's `averages/*.lda.npy`. Describes one corpus's *content*.

Usage:
    from tools.hegre_dataset.basis_fingerprint import assert_basis, stamp, verify

    assert_basis(Path("/mnt/.../training-data/ffhq/stratum"))   # raises on mismatch
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path

_PROJ = Path(__file__).resolve().parent.parent.parent
DEFAULT_BASIS_DIR = _PROJ / "experiments" / "geometry_pca" / "output"
STAMP_NAME = "BASIS_FINGERPRINT.json"
BASIS_FILES = ("auraface_lda.npz", "auraface_preprocess.npz")


class BasisMismatch(RuntimeError):
    """Raised when a dataset's LDA basis fingerprint is missing or disagrees."""


def compute_basis_fingerprint(basis_dir: Path = DEFAULT_BASIS_DIR) -> str:
    """sha256 over the basis artifacts (name + bytes of each), truncated to 16 hex."""
    basis_dir = Path(basis_dir)
    h = hashlib.sha256()
    for name in BASIS_FILES:
        p = basis_dir / name
        if not p.is_file():
            raise BasisMismatch(f"basis artifact missing: {p}")
        h.update(name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def basis_meta(basis_dir: Path = DEFAULT_BASIS_DIR) -> dict:
    basis_dir = Path(basis_dir)
    return {
        "basis_fingerprint": compute_basis_fingerprint(basis_dir),
        "basis_dir": str(basis_dir),
        "basis_files": {n: hashlib.sha256((basis_dir / n).read_bytes()).hexdigest()[:16]
                        for n in BASIS_FILES},
    }


def stamp(dataset_dir: Path, basis_dir: Path = DEFAULT_BASIS_DIR, *,
          convention: str, sample_count: int | None = None,
          scope: str = "all samples") -> Path:
    """Write BASIS_FINGERPRINT.json into dataset_dir.

    convention: how vectors in this dir are encoded, e.g.
        "refit basis + L2-normalize (norm 1.0)"   <- hegre_corpus / ffhq after reprojection
        "refit basis, raw coords (norm ~153)"     <- hegre-faces/v1/lda per-image
    """
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.is_dir():
        raise BasisMismatch(f"dataset dir does not exist: {dataset_dir}")
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=_PROJ,
                                capture_output=True, text=True).stdout.strip()
    except Exception:
        commit = ""
    payload = {
        "stamped_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": commit,
        "dataset_dir": str(dataset_dir),
        "projection_convention": convention,
        "sample_count": sample_count,
        "scope": scope,
        **basis_meta(basis_dir),
    }
    out = dataset_dir / STAMP_NAME
    out.write_text(json.dumps(payload, indent=2))
    return out


def verify(dataset_dir: Path, basis_dir: Path = DEFAULT_BASIS_DIR) -> dict:
    """Return {'ok': bool, 'reason': str, ...}. Never raises."""
    dataset_dir = Path(dataset_dir)
    p = dataset_dir / STAMP_NAME
    if not p.is_file():
        return {"ok": False, "reason": f"no {STAMP_NAME} in {dataset_dir} (unstamped)",
                "expected": compute_basis_fingerprint(basis_dir)}
    try:
        d = json.loads(p.read_text())
    except Exception as e:
        return {"ok": False, "reason": f"unreadable stamp: {e}"}
    expected = compute_basis_fingerprint(basis_dir)
    got = d.get("basis_fingerprint")
    if got != expected:
        return {"ok": False, "reason": "basis fingerprint mismatch", "found": got,
                "expected": expected, "stamped_at": d.get("stamped_at"),
                "convention": d.get("projection_convention")}
    if not d.get("projection_convention"):
        return {"ok": False, "reason": "stamp has no projection_convention"}
    return {"ok": True, "fingerprint": got, "convention": d.get("projection_convention"),
            "stamped_at": d.get("stamped_at")}


def assert_basis(dataset_dir: Path, basis_dir: Path = DEFAULT_BASIS_DIR) -> dict:
    """Loader-side guard. Raises BasisMismatch on missing stamp or mismatch."""
    r = verify(dataset_dir, basis_dir)
    if not r["ok"]:
        raise BasisMismatch(
            f"LDA basis check FAILED for {dataset_dir}: {r['reason']}. "
            f"Refusing to load — a mixed-basis dataset produces silently "
            f"invalid experiments. Reproject with scripts/reproject_lda_ffhq.py "
            f"(or the hegre equivalent), then re-stamp."
        )
    return r