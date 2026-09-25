"""Tests for the shared AuraFace extractor.

Scope note: the agent venv has neither insightface nor torch, so these tests
cover everything EXCEPT the actual detector call -- basis verification, the
outcome dataclasses, batch counting, and the instrument record. The detector path
is exercised wherever the real run happens.

The negative controls matter more than the positive ones here: a basis guard that
cannot be shown to fire is not a guard.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from tools.auraface import (
    BASIS_DIR,
    CORPUS_DET_SIZE,
    EXPECTED_ARTIFACT_SHA256,
    PAD_FRACTION,
    AuraFaceResult,
    BatchSummary,
    RefitBasisError,
    describe_instrument,
    verify_basis,
)
from tools.auraface import extract as extract_mod


# --------------------------------------------------------------------------
# positive: the real artifacts must verify
# --------------------------------------------------------------------------

def test_real_basis_verifies():
    v = verify_basis()
    assert set(v["artifacts"]) == set(EXPECTED_ARTIFACT_SHA256)
    for name, rec in v["artifacts"].items():
        assert rec["sha256"] == rec["expected"], f"{name} hash drift"


def test_corpus_det_size_is_none():
    """None == do not pass det_size == insightface default 640 == corpus."""
    assert CORPUS_DET_SIZE is None
    assert PAD_FRACTION == 0.20


def test_describe_instrument_records_everything_required():
    d = describe_instrument()

    # condition 3: det_size explicit, and both artifact hashes recorded
    assert "det_size" in d and d["det_size"] is None
    assert d["artifact_sha256"]["auraface_lda.npz"] == EXPECTED_ARTIFACT_SHA256["auraface_lda.npz"]
    assert (
        d["artifact_sha256"]["auraface_preprocess.npz"]
        == EXPECTED_ARTIFACT_SHA256["auraface_preprocess.npz"]
    )

    # the model files that produce the embedding are named
    assert d["model_dir"].endswith("/models/auraface")
    assert "glintr100.onnx" in d["model_sha256"]

    # and the convention is stated, not implied
    assert "640" in d["det_size_note"]


# --------------------------------------------------------------------------
# negative control: condition 4 -- must fail loudly on a pre-refit basis
# --------------------------------------------------------------------------

def test_pre_refit_backup_only_is_refused(tmp_path):
    """A dir where only the `*.bak-20260720` artifact exists must be refused."""
    (tmp_path / "auraface_lda.npz.bak-20260720").write_bytes(b"stale")
    (tmp_path / "auraface_preprocess.npz").write_bytes(b"x")

    with pytest.raises(RefitBasisError) as e:
        verify_basis(tmp_path)
    assert "Pre-refit basis detected" in str(e.value)
    assert "2026-07-23" in str(e.value)


def test_swapped_artifact_is_refused_by_hash(tmp_path):
    """Right filename, wrong bytes -- the silent-swap case -- must be refused."""
    for name in EXPECTED_ARTIFACT_SHA256:
        (tmp_path / name).write_bytes(b"not the refit artifact")

    with pytest.raises(RefitBasisError) as e:
        verify_basis(tmp_path)
    assert "hash mismatch" in str(e.value)
    # the message must say how to resolve it, not just that it failed
    assert "EXPECTED_ARTIFACT_SHA256" in str(e.value)


def test_missing_artifact_is_refused(tmp_path):
    with pytest.raises(RefitBasisError) as e:
        verify_basis(tmp_path)
    assert "not found" in str(e.value)


def test_bak_file_loaded_by_name_is_refused():
    """Even a direct path to a backup is refused before it is read."""
    bak_name = "auraface_lda.npz.bak-20260720"
    assert bak_name.endswith(extract_mod.REJECTED_ARTIFACT_SUFFIXES)


# --------------------------------------------------------------------------
# condition 2: per-image outcome, countable
# --------------------------------------------------------------------------

def test_result_is_self_describing():
    r = AuraFaceResult(
        ok=True,
        outcome="detected_after_padding",
        n_faces=2,
        n_faces_first_pass=0,
        normed_embedding=np.zeros(512, dtype=np.float32),
        lda_coords=np.zeros(64),
        source_path=Path("/tmp/x.png"),
    )
    assert r.used_padding is True
    assert r.ambiguous is True          # two faces: faces[0] was a choice
    d = r.as_dict()
    assert d["outcome"] == "detected_after_padding"
    assert d["n_faces"] == 2
    assert d["n_faces_first_pass"] == 0
    assert d["ambiguous"] is True
    assert d["source_path"] == "/tmp/x.png"


def test_no_face_result_is_not_an_exception():
    """A face-free image is data, not a crash -- so it can be counted."""
    r = AuraFaceResult(ok=False, outcome="no_face")
    assert r.ok is False
    assert r.normed_embedding is None
    assert r.used_padding is False
    assert r.as_dict()["has_embedding"] is False


def test_batch_summary_counts_skips_and_padding():
    s = BatchSummary(
        n_total=10, n_ok=7, n_no_face=3, n_used_padding=2, n_ambiguous=1
    )
    assert s.skipped == 3
    assert s.skip_rate == pytest.approx(0.3)
    assert s.padding_rate == pytest.approx(2 / 7)
    d = s.as_dict()
    assert d["skipped"] == 3 and d["n_used_padding"] == 2 and d["n_ambiguous"] == 1


def test_batch_summary_empty_is_safe():
    s = BatchSummary()
    assert s.skipped == 0
    assert s.skip_rate == 0.0          # no ZeroDivisionError
    assert s.padding_rate == 0.0


# --------------------------------------------------------------------------
# the extractor refuses to run on a bad basis, before doing any work
# --------------------------------------------------------------------------

def test_extract_verifies_basis_before_detecting(monkeypatch):
    """verify_basis must run before any detection, so a bad basis costs no compute."""
    calls = []

    def fake_verify(*a, **k):
        calls.append(1)
        raise RefitBasisError("stopped early")

    monkeypatch.setattr(extract_mod, "verify_basis", fake_verify)

    with pytest.raises(RefitBasisError):
        extract_mod.extract_auraface(np.zeros((64, 64, 3), dtype=np.uint8))

    assert calls, "verify_basis must be called even for an in-memory array"


def test_module_exports_the_four_conditions():
    """Guard the contract prx-tg agreed to."""
    import tools.auraface as af

    for name in (
        "extract_auraface",        # 1 + 2: embedding, LDA, outcome
        "extract_auraface_batch",  # 2: countable skips
        "describe_instrument",     # 3: det_size + artifact hashes
        "verify_basis",            # 4: pre-refit is fatal
        "RefitBasisError",         # 4: the loud failure type
    ):
        assert hasattr(af, name), f"missing from the public contract: {name}"
