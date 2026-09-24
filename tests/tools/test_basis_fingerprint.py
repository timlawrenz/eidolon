"""Tests for the LDA basis fingerprint guard.

The guard exists to make a SILENT failure loud: a dataset projected through a
different basis than the one a loader expects. Its most important property is
therefore that it can FAIL — a guard that always passes is worse than none.
"""
from pathlib import Path

import pytest

from tools.hegre_dataset.basis_fingerprint import (
    BasisMismatch,
    assert_basis,
    compute_basis_fingerprint,
    stamp,
    verify,
)

CONV = "refit basis + L2-normalize (norm 1.0)"


def _fake_basis(d: Path, tag: bytes = b"a") -> Path:
    d.mkdir(parents=True, exist_ok=True)
    (d / "auraface_lda.npz").write_bytes(b"lda-" + tag)
    (d / "auraface_preprocess.npz").write_bytes(b"prep-" + tag)
    return d


def test_unstamped_dir_fails(tmp_path):
    basis = _fake_basis(tmp_path / "basis")
    ds = tmp_path / "ds"
    ds.mkdir()
    r = verify(ds, basis)
    assert r["ok"] is False
    assert "unstamped" in r["reason"]


def test_stamp_then_verify_ok(tmp_path):
    basis = _fake_basis(tmp_path / "basis")
    ds = tmp_path / "ds"
    ds.mkdir()
    stamp(ds, basis, convention=CONV, sample_count=3)
    r = verify(ds, basis)
    assert r["ok"] is True
    assert r["convention"] == CONV
    assert r["fingerprint"] == compute_basis_fingerprint(basis)


def test_different_basis_is_detected(tmp_path):
    """The whole point: stamp under basis A, verify against basis B -> FAIL."""
    basis_a = _fake_basis(tmp_path / "A", b"a")
    basis_b = _fake_basis(tmp_path / "B", b"b")
    ds = tmp_path / "ds"
    ds.mkdir()
    stamp(ds, basis_a, convention=CONV)
    assert verify(ds, basis_a)["ok"] is True
    r = verify(ds, basis_b)
    assert r["ok"] is False
    assert r["reason"] == "basis fingerprint mismatch"
    assert r["found"] != r["expected"]


def test_assert_basis_raises_on_mismatch(tmp_path):
    basis = _fake_basis(tmp_path / "basis")
    ds = tmp_path / "ds"
    ds.mkdir()
    with pytest.raises(BasisMismatch):
        assert_basis(ds, basis)


def test_missing_projection_convention_is_rejected(tmp_path):
    """A stamp without a convention record is not trustworthy."""
    import json

    basis = _fake_basis(tmp_path / "basis")
    ds = tmp_path / "ds"
    ds.mkdir()
    stamp(ds, basis, convention=CONV)
    p = ds / "BASIS_FINGERPRINT.json"
    d = json.loads(p.read_text())
    d["projection_convention"] = ""
    p.write_text(json.dumps(d))
    r = verify(ds, basis)
    assert r["ok"] is False
    assert "projection_convention" in r["reason"]


def test_fingerprint_stable_and_distinct(tmp_path):
    a = _fake_basis(tmp_path / "A", b"a")
    b = _fake_basis(tmp_path / "B", b"b")
    assert compute_basis_fingerprint(a) == compute_basis_fingerprint(a)
    assert compute_basis_fingerprint(a) != compute_basis_fingerprint(b)


def test_missing_basis_artifact_raises(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    with pytest.raises(BasisMismatch):
        compute_basis_fingerprint(d)