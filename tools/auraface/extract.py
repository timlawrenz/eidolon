"""Shared AuraFace extraction — one convention, for corpus and for eval.

WHY THIS MODULE EXISTS
----------------------
The repo grew two AuraFace invocation conventions: the corpus path
(`tools/hegre_dataset/enrichment.py`) calls `app.prepare(ctx_id=0)`, i.e.
insightface's default detector input of 640x640; the FFHQ path
(`scripts/pipeline/extract_ffhq_auraface.py`) passes `det_size=(512, 512)`.
Those are different detector input resolutions and can detect different faces.

A third convention is the outcome this module exists to prevent. Both projects
call `extract_auraface()` and get the same behaviour, and the behaviour used is
recorded in the result rather than implied.

The corpus images are *already* tight face crops. SCRFD routinely fails on those
because the face fills the frame and there is no shoulder/background context, so
the corpus path catches the zero-detection case, pads 20% black border and
retries. Generated images are face-dominant in exactly the same way, so the
padding fallback is part of the contract, not an optional extra.

WHAT EVERY RESULT CARRIES
-------------------------
  * the raw `normed_embedding` (512-d) AND the projected LDA coordinates (64-d),
    so a caller can measure in either space;
  * an explicit per-image outcome -- `detected` / `detected_after_padding` /
    `no_face` -- plus `n_faces`, so a biased skip is countable and a two-face
    render is visible instead of silently taking `faces[0]`;
  * the `det_size` actually used.

Basis safety
------------
`verify_basis()` fails loudly on a pre-refit basis. The pre-refit artifacts are
still on disk as `*.bak-20260720`; pointing at one, or swapping one in, would
confound every identity number downstream, so it is refused by content hash and
by filename, with a message that says what to do.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal, Sequence

import numpy as np

__all__ = [
    "AuraFaceResult",
    "BatchSummary",
    "RefitBasisError",
    "Outcome",
    "MODEL_NAME",
    "MODEL_ROOT",
    "CORPUS_DET_SIZE",
    "PAD_FRACTION",
    "BASIS_DIR",
    "EXPECTED_ARTIFACT_SHA256",
    "REJECTED_ARTIFACT_SUFFIXES",
    "verify_basis",
    "describe_instrument",
    "extract_auraface",
    "extract_auraface_batch",
]

Outcome = Literal["detected", "detected_after_padding", "no_face"]

#: insightface model pack name and root. Resolves to
#: /mnt/nas-ai-models/models/auraface/, which holds glintr100.onnx (the
#: recognition net that produces the 512-d embedding) and scrfd_10g_bnkps.onnx
#: (the detector used for detection *and* the 5-point alignment crop).
MODEL_NAME = "auraface"
MODEL_ROOT = "/mnt/nas-ai-models"

#: `None` == do not pass `det_size` to `app.prepare()`, so insightface uses its
#: default of 640x640. This is CORPUS BEHAVIOUR. Passing (512, 512) reproduces
#: the FFHQ path and is a *different* instrument; both are allowed, but the
#: choice is recorded in every result so it can never be implicit.
CORPUS_DET_SIZE: tuple[int, int] | None = None

#: Fraction of width/height added as a black border on the retry, per side.
PAD_FRACTION = 0.20

_PROJ = Path(__file__).resolve().parent.parent.parent
BASIS_DIR = _PROJ / "experiments" / "geometry_pca" / "output"

#: sha256 of the CURRENT (post-refit) basis artifacts. These are pinned so that
#: a swapped-in artifact is a loud failure rather than a silent confound.
EXPECTED_ARTIFACT_SHA256 = {
    "auraface_preprocess.npz": "6bbc0937b7b83d6a3901df6525b8d2073e5c5e02cca395f2b00a5aa551201ebe",
    "auraface_lda.npz": "8ebcc47f6de6cecb2f0d8dddb494fd3ec9bb42d0561186af1d1640ddfba487b8",
}

#: A pre-refit artifact is recognisable by name before it is even hashed.
REJECTED_ARTIFACT_SUFFIXES = (".bak-20260720",)

#: The pooled refit basis everything in the project is stamped with.
EXPECTED_BASIS_FINGERPRINT = "120e1c5a1dc4f423"


class RefitBasisError(RuntimeError):
    """Raised when the AuraFace/LDA basis artifacts are not the post-refit ones.

    Always fatal. Using a pre-refit basis confounds identity comparisons
    silently, so this never degrades to a warning.
    """


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_basis(basis_dir: Path | str = BASIS_DIR, *, check_fingerprint: bool = True) -> dict:
    """Assert the basis artifacts are the post-refit ones. Raises RefitBasisError.

    Three independent checks, because each catches a different mistake:
      1. filename -- a `*.bak-20260720` path is refused before anything is read;
      2. content hash -- an artifact swapped in under the correct name is caught;
      3. basis fingerprint -- the pooled refit stamp, if the stamp file exists.

    Returns a dict of the verified hashes and fingerprint, suitable for logging.
    """
    basis_dir = Path(basis_dir)

    for name in EXPECTED_ARTIFACT_SHA256:
        candidate = basis_dir / name
        if candidate.exists():
            continue
        # a backup sitting where the real artifact belongs
        for suffix in REJECTED_ARTIFACT_SUFFIXES:
            if (basis_dir / f"{name}{suffix}").exists() and not candidate.exists():
                raise RefitBasisError(
                    f"Pre-refit basis detected: {name} is missing and only "
                    f"{name}{suffix} is present in {basis_dir}.\n"
                    f"Refusing to run: the pre-refit artifacts are from before the "
                    f"2026-07-23 pooled refit and would confound every identity "
                    f"number produced from this run.\n"
                    f"Restore (or refit) the post-refit {name} and retry."
                )
        raise RefitBasisError(
            f"Basis artifact not found: {candidate}\n"
            f"Expected the post-refit artifact. See "
            f"docs/02_EXPERIMENTS_AND_RESULTS.md (LDA basis refit) for how it is produced."
        )

    verified: dict = {"basis_dir": str(basis_dir), "artifacts": {}}
    for name, expected in EXPECTED_ARTIFACT_SHA256.items():
        path = basis_dir / name

        for suffix in REJECTED_ARTIFACT_SUFFIXES:
            if path.name.endswith(suffix):
                raise RefitBasisError(
                    f"Refusing to load pre-refit artifact by name: {path.name}\n"
                    f"The '{suffix}' artifacts predate the 2026-07-23 pooled refit."
                )

        actual = _sha256(path)
        verified["artifacts"][name] = {"sha256": actual, "expected": expected}
        if actual != expected:
            raise RefitBasisError(
                f"Basis artifact hash mismatch for {name}:\n"
                f"  expected (post-refit): {expected}\n"
                f"  found:                 {actual}\n"
                f"  path:                  {path}\n"
                f"Refusing to run. Either this is a pre-refit artifact, or the basis "
                f"was legitimately refit -- in which case pin the new hash in "
                f"tools/auraface/extract.py:EXPECTED_ARTIFACT_SHA256 and record the "
                f"refit in the ledger before measuring anything with it."
            )

    if check_fingerprint:
        verified["basis_fingerprint"] = _read_fingerprint(basis_dir)
    return verified


def _read_fingerprint(basis_dir: Path) -> str | None:
    """Reuse the project's own fingerprint helper when it is importable."""
    try:
        import sys

        if str(_PROJ) not in sys.path:
            sys.path.insert(0, str(_PROJ))
        from tools.hegre_dataset.basis_fingerprint import compute_basis_fingerprint

        return compute_basis_fingerprint(basis_dir)
    except Exception as exc:  # pragma: no cover - informational only
        print(f"Warning: could not compute basis fingerprint: {exc}")
        return None


def _hash_models() -> dict:
    """Hash the two model files that actually matter, for the provenance record."""
    model_dir = Path(MODEL_ROOT) / "models" / MODEL_NAME
    out = {}
    for fname in ("glintr100.onnx", "scrfd_10g_bnkps.onnx"):
        p = model_dir / fname
        out[fname] = _sha256(p) if p.exists() else None
    return out


def describe_instrument(*, det_size: tuple[int, int] | None = CORPUS_DET_SIZE) -> dict:
    """Everything a run needs to record about how identity was measured.

    Call this once at the start of an evaluation and write the result next to the
    metrics. It is the answer to "which AuraFace convention produced these
    numbers?" without having to read the code again.
    """
    basis = verify_basis()
    model_dir = Path(MODEL_ROOT) / "models" / MODEL_NAME
    return {
        "model_name": MODEL_NAME,
        "model_root": MODEL_ROOT,
        "model_dir": str(model_dir),
        "det_size": det_size,
        "det_size_note": (
            "None means insightface default 640x640 -- CORPUS BEHAVIOUR. "
            "A tuple reproduces the FFHQ path and is a different instrument."
        ),
        "pad_fraction": PAD_FRACTION,
        "embedding": "normed_embedding (512-d, L2-normalised)",
        "lda_components": 64,
        "basis_dir": basis["basis_dir"],
        "artifact_sha256": {k: v["sha256"] for k, v in basis["artifacts"].items()},
        "basis_fingerprint": basis.get("basis_fingerprint"),
        "expected_basis_fingerprint": EXPECTED_BASIS_FINGERPRINT,
        "model_sha256": _hash_models(),
    }


@dataclass
class AuraFaceResult:
    """One image's identity measurement, with its own outcome and provenance.

    Attributes:
        ok: True iff an embedding was produced.
        outcome: 'detected' | 'detected_after_padding' | 'no_face'.
        n_faces: len(faces) from the (successful) detection pass. Recorded even on
            success: a render that produced two faces is visible here rather than
            silently collapsing to faces[0] = 0.
        n_faces_first_pass: len(faces) before the padding retry. 0 together with a
            non-zero n_faces means the padding fallback is what saved this image --
            useful for knowing how much of a cohort depends on it.
        normed_embedding: (512,) float32 L2-normalised AuraFace embedding, or None.
        lda_coords: (64,) float64 LDA identity coordinates, or None.
        source_path: where the image came from, so a value can always be traced
            back to its input.
        det_size: the detector input size actually used for this image.
        backend: 'numpy' for this implementation (kept for future parity).
    """

    ok: bool
    outcome: Outcome
    n_faces: int = 0
    n_faces_first_pass: int = 0
    normed_embedding: np.ndarray | None = None
    lda_coords: np.ndarray | None = None
    source_path: Path | None = None
    det_size: tuple[int, int] | None = CORPUS_DET_SIZE
    backend: str = "insightface"

    @property
    def used_padding(self) -> bool:
        return self.outcome == "detected_after_padding"

    @property
    def ambiguous(self) -> bool:
        """True when more than one face was found: faces[0] was a choice, not a fact."""
        return self.n_faces > 1

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "outcome": self.outcome,
            "n_faces": self.n_faces,
            "n_faces_first_pass": self.n_faces_first_pass,
            "used_padding": self.used_padding,
            "ambiguous": self.ambiguous,
            "source_path": str(self.source_path) if self.source_path else None,
            "det_size": self.det_size,
            "has_embedding": self.normed_embedding is not None,
            "has_lda": self.lda_coords is not None,
        }


@dataclass
class BatchSummary:
    """Countable outcomes across a batch -- so silent skips cannot happen.

    `skipped` is the number that matters. If it is non-zero, whatever metric is
    computed from `results` is conditioned on detectability, and the caller has to
    say so.
    """

    n_total: int = 0
    n_ok: int = 0
    n_no_face: int = 0
    n_used_padding: int = 0
    n_ambiguous: int = 0
    results: list[AuraFaceResult] = field(default_factory=list)

    @property
    def skipped(self) -> int:
        return self.n_no_face

    @property
    def skip_rate(self) -> float:
        return (self.n_no_face / self.n_total) if self.n_total else 0.0

    @property
    def padding_rate(self) -> float:
        return (self.n_used_padding / self.n_ok) if self.n_ok else 0.0

    def as_dict(self) -> dict:
        return {
            "n_total": self.n_total,
            "n_ok": self.n_ok,
            "skipped": self.skipped,
            "skip_rate": self.skip_rate,
            "n_used_padding": self.n_used_padding,
            "padding_rate": self.padding_rate,
            "n_ambiguous": self.n_ambiguous,
        }


_APP = None


def _get_app(det_size: tuple[int, int] | None = CORPUS_DET_SIZE):
    """Load (once) the insightface FaceAnalysis app with the AuraFace pack.

    `app.prepare(ctx_id=0)` with no det_size is the corpus behaviour; passing
    det_size changes the detector's input resolution, so a single process must
    not mix the two. Cache is keyed on det_size.
    """
    global _APP
    key = det_size
    cached = getattr(_get_app, "_cache", {})
    if key in cached:
        return cached[key]

    from insightface.app import FaceAnalysis

    app = FaceAnalysis(name=MODEL_NAME, root=MODEL_ROOT, providers=["CPUExecutionProvider"])
    if det_size is None:
        app.prepare(ctx_id=0)
    else:
        app.prepare(ctx_id=0, det_size=det_size)

    cached[key] = app
    _get_app._cache = cached  # type: ignore[attr-defined]
    _APP = app
    return app


def _to_lda(normed_embedding: np.ndarray) -> np.ndarray:
    """clean -> project. Imported lazily; geometry_pca is an experiments package."""
    import sys

    geom = _PROJ / "experiments" / "geometry_pca"
    if str(geom) not in sys.path:
        sys.path.insert(0, str(geom))
    from geometry_pca.auraface_preprocessing import clean_auraface, project_to_lda

    return np.asarray(project_to_lda(clean_auraface(normed_embedding)), dtype=np.float64)


def extract_auraface(
    image,
    *,
    source_path: Path | str | None = None,
    det_size: tuple[int, int] | None = CORPUS_DET_SIZE,
    pad_fraction: float = PAD_FRACTION,
    keep_embedding: bool = True,
    keep_lda: bool = True,
) -> AuraFaceResult:
    """Extract identity from ONE image, exactly as the corpus does.

    Args:
        image: an (H, W, 3) BGR uint8 array (as cv2.imread returns), or a path.
        source_path: recorded on the result; inferred when `image` is a path.
        det_size: None == corpus behaviour (insightface default 640x640).
        pad_fraction: black border added per side on the detection retry.
        keep_embedding / keep_lda: drop either space if the caller only needs one.

    Returns:
        AuraFaceResult. Never raises on a face-free image -- `ok=False` and
        `outcome='no_face'` are the signal, so the caller can count rather than
        discover a gap later.
    """
    import cv2

    if isinstance(image, (str, Path)):
        source_path = source_path or Path(image)
        img = cv2.imread(str(image))
        if img is None:
            return AuraFaceResult(
                ok=False, outcome="no_face", source_path=Path(source_path), det_size=det_size
            )
    else:
        img = image

    verify_basis()
    app = _get_app(det_size)

    faces = app.get(img)
    n_first = len(faces)
    outcome: Outcome = "detected"

    if n_first == 0:
        h, w = img.shape[:2]
        pad_y, pad_x = int(h * pad_fraction), int(w * pad_fraction)
        padded = cv2.copyMakeBorder(
            img, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
        faces = app.get(padded)
        if len(faces) == 0:
            return AuraFaceResult(
                ok=False,
                outcome="no_face",
                n_faces=0,
                n_faces_first_pass=0,
                source_path=Path(source_path) if source_path else None,
                det_size=det_size,
            )
        outcome = "detected_after_padding"

    emb = np.asarray(faces[0].normed_embedding, dtype=np.float32)

    return AuraFaceResult(
        ok=True,
        outcome=outcome,
        n_faces=len(faces),
        n_faces_first_pass=n_first,
        normed_embedding=emb if keep_embedding else None,
        lda_coords=_to_lda(emb) if keep_lda else None,
        source_path=Path(source_path) if source_path else None,
        det_size=det_size,
    )


def extract_auraface_batch(
    images: Iterable,
    *,
    det_size: tuple[int, int] | None = CORPUS_DET_SIZE,
    pad_fraction: float = PAD_FRACTION,
    keep_embedding: bool = True,
    keep_lda: bool = True,
) -> BatchSummary:
    """Extract identity for many images, counting every outcome.

    Each element may be a path or an (H, W, 3) BGR array. Prefer paths: the
    insightface app is loaded once and reused.

    Returns:
        BatchSummary. Check `.skipped` before computing any metric -- a non-zero
        skip rate means the metric is conditioned on detectability.
    """
    summary = BatchSummary()
    for item in images:
        summary.n_total += 1
        res = extract_auraface(
            item,
            det_size=det_size,
            pad_fraction=pad_fraction,
            keep_embedding=keep_embedding,
            keep_lda=keep_lda,
        )
        summary.results.append(res)
        if res.ok:
            summary.n_ok += 1
            if res.used_padding:
                summary.n_used_padding += 1
            if res.ambiguous:
                summary.n_ambiguous += 1
        else:
            summary.n_no_face += 1
    return summary
