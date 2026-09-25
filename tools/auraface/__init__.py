"""Shared AuraFace identity extraction for eidolon and prx-tg.

One implementation, one convention. Import the function, not a convention:

    from tools.auraface import extract_auraface, describe_instrument

    # record once, next to your metrics
    provenance = describe_instrument()          # det_size, artifact hashes, fingerprint

    res = extract_auraface(path)
    res.ok            # False -> count it, do not drop it silently
    res.outcome       # 'detected' | 'detected_after_padding' | 'no_face'
    res.n_faces       # > 1 -> faces[0] was a choice; res.ambiguous is True
    res.normed_embedding   # (512,) raw AuraFace, L2-normalised
    res.lda_coords         # (64,) LDA identity coordinates

    batch = extract_auraface_batch(paths)
    batch.skipped     # non-zero -> your metric is conditioned on detectability

`verify_basis()` raises RefitBasisError on a pre-refit (`*.bak-20260720`) basis.
It is called automatically before the first extraction; call it explicitly at the
start of a run so a basis problem fails before any compute is spent.
"""

from .extract import (
    BASIS_DIR,
    CORPUS_DET_SIZE,
    EXPECTED_ARTIFACT_SHA256,
    EXPECTED_BASIS_FINGERPRINT,
    MODEL_NAME,
    MODEL_ROOT,
    PAD_FRACTION,
    AuraFaceResult,
    BatchSummary,
    Outcome,
    RefitBasisError,
    describe_instrument,
    extract_auraface,
    extract_auraface_batch,
    verify_basis,
)

__all__ = [
    "AuraFaceResult",
    "BatchSummary",
    "Outcome",
    "RefitBasisError",
    "BASIS_DIR",
    "CORPUS_DET_SIZE",
    "EXPECTED_ARTIFACT_SHA256",
    "EXPECTED_BASIS_FINGERPRINT",
    "MODEL_NAME",
    "MODEL_ROOT",
    "PAD_FRACTION",
    "describe_instrument",
    "extract_auraface",
    "extract_auraface_batch",
    "verify_basis",
]
