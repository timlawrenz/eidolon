#!/usr/bin/env python3
"""
FFHQ stratum AuraFace-LDA reprojection onto the refit basis (2026-07-20).

WHY: /mnt/.../ffhq/stratum/{id}/auraface_lda.npy was written 2026-06-30, i.e.
BEFORE the pooled LDA basis refit (2026-07-23). Verified bit-exact: the stored
files are `project_to_lda` output under the PRE-refit basis (`*.bak-20260720`).
The refit basis gives coords ~153 in norm where the old basis gave ~0.35.

Consequence: any Eidolon-adapter run whose `stratum_dirs` includes
ffhq/stratum received a 64-d identity conditioning slot carrying two
incompatible encodings (FFHQ old basis + hegre_corpus refit basis).
`prx-tg/production/data_stratum.py` loads `auraface_lda.npy` directly.

TARGET CONVENTION: refit basis + L2-normalize (norm == 1.0).
Rationale: hegre_corpus (the shipped training data) stores L2-normalized
vectors (norm exactly 1.000000). L2 normalization provably does not change
cosine geometry (verified: between-image cosine identical, 0.9945 +/- 0.0012),
so this removes the magnitude mismatch at zero information cost. The
alternative (raw refit coords, norm ~153) matches the per-image retrieval
convention in hegre-faces/v1/lda/ and is NOT what the DiT consumes.

Usage:
    python scripts/reproject_lda_ffhq.py --dry-run     # validate + report, no writes
    python scripts/reproject_lda_ffhq.py --apply       # backup, reproject, write manifest
    python scripts/reproject_lda_ffhq.py --verify      # confirm stored == recomputed (refit basis)
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import numpy as np

_PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJ))
sys.path.insert(0, str(_PROJ / "experiments" / "geometry_pca"))
from geometry_pca.auraface_preprocessing import clean_auraface, project_to_lda  # noqa: E402

FFHQ_ROOT = Path("/mnt/nas-ai-models/training-data/ffhq")
STRATUM = FFHQ_ROOT / "stratum"
RAW = FFHQ_ROOT / "auraface"
OUTPUT = _PROJ / "experiments" / "geometry_pca" / "output"
BASIS_LDA = OUTPUT / "auraface_lda.npz"
BASIS_PREP = OUTPUT / "auraface_preprocess.npz"
OLD_LDA = OUTPUT / "auraface_lda.npz.bak-20260720"
OLD_PREP = OUTPUT / "auraface_preprocess.npz.bak-20260720"
BACKUP = STRATUM / "_auraface_lda.oldbasis-backup.tar.gz"
MANIFEST = STRATUM / "_auraface_lda.reproject_manifest.json"

EIDOLON = Path("/mnt/nas-ai-models/training-data/eidolon")
STAMP_TARGETS = [
    (STRATUM, "refit basis + L2-normalize (norm 1.0)"),
    (EIDOLON / "hegre_corpus", "refit basis + L2-normalize (norm 1.0)"),
    (EIDOLON / "hegre-faces/v1/lda", "refit basis, raw coords (norm ~153)"),
]


def basis_fingerprint():
    """sha256 over the two refit-basis artifacts. Identifies the projection."""
    h = hashlib.sha256()
    for p in (BASIS_LDA, BASIS_PREP):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def _set_module_basis(prep_path, lda_path):
    from geometry_pca import auraface_preprocessing as ap
    ap._REF = None
    ap._LDA = None
    ap._REF_PATH = prep_path
    ap._LDA_PATH = lda_path


def project(raw, normalize=True, new_basis=True):
    _set_module_basis(BASIS_PREP if new_basis else OLD_PREP,
                      BASIS_LDA if new_basis else OLD_LDA)
    c = np.asarray(project_to_lda(clean_auraface(raw)), dtype=np.float64).ravel()
    if normalize:
        c = c / (np.linalg.norm(c) + 1e-12)
    return c


def discover():
    """Return [(id, raw_path|None, lda_path|None, dir_exists)] over real stratum dirs."""
    rows = []
    for d in sorted(STRATUM.iterdir()):
        if not d.is_dir() or d.name.startswith("@") or d.name.startswith("_"):
            continue
        sid = d.name
        rows.append((sid, RAW / f"{sid}.npy", d / "auraface_lda.npy"))
    return rows


def dry_run(rows):
    have_raw = [(s, r, l) for s, r, l in rows if r.exists()]
    have_lda = [(s, r, l) for s, r, l in rows if l.exists()]
    print(f"stratum dirs (real)          : {len(rows)}")
    print(f"  with raw AuraFace          : {len(have_raw)}")
    print(f"  with auraface_lda          : {len(have_lda)}")
    print(f"  raw but no lda  (creatable): {len(set(s for s, _, _ in have_raw) - set(s for s, _, _ in have_lda))}")
    print(f"  lda but no raw  (UNFIXABLE): {len(set(s for s, _, _ in have_lda) - set(s for s, _, _ in have_raw))}")
    print(f"  basis fingerprint          : {basis_fingerprint()}")

    print("\nvalidating that stored files are on the OLD basis (expect ~0.0 vs old, ~150 vs new):")
    d_old, d_new = [], []
    for s, r, l in have_lda[:250]:
        if not r.exists():
            continue
        stored = np.load(l).astype(np.float64).ravel()
        d_old.append(float(np.linalg.norm(stored - project(np.load(r), normalize=False, new_basis=False))))
        d_new.append(float(np.linalg.norm(stored - project(np.load(r), normalize=False, new_basis=True))))
    if d_old:
        print(f"  ||stored - OLD|| mean={np.mean(d_old):.8f} max={np.max(d_old):.8f}")
        print(f"  ||stored - NEW|| mean={np.mean(d_new):.8f} max={np.max(d_new):.8f}")
        print(f"  => {'CONFIRMED old basis' if np.max(d_old) < 1e-6 else 'UNEXPECTED — investigate before applying'}")

    print("\npreview (first 3):")
    for s, r, l in have_lda[:3]:
        if not r.exists():
            continue
        v = project(np.load(r))
        print(f"  {s}: new+norm norm={np.linalg.norm(v):.6f}  sample={np.round(v[:4], 6)}")


def _already_done(lda_path, tol=1e-6):
    """True if the file is already refit-basis + L2-normalized (norm == 1.0).

    Refit-basis raw coords are ~153 and pre-refit coords are ~0.35, so unit norm
    is an unambiguous marker of 'this file has already been reprojected'.
    Makes --apply resumable after an interruption.
    """
    try:
        v = np.load(lda_path)
        return v.shape == (64,) and abs(float(np.linalg.norm(v)) - 1.0) < tol
    except Exception:
        return False


def apply(rows, force=False):
    all_pairs = [(s, r, l) for s, r, l in rows if r.exists() and l.exists()]
    # The per-file "already done" check costs one NAS read per file. It is only
    # needed on a RESUME: apply() always creates the backup before reprojecting,
    # so if no backup exists, nothing has been reprojected yet.
    if force or not BACKUP.exists():
        pairs = all_pairs
    else:
        pairs = [(s, r, l) for s, r, l in all_pairs if not _already_done(l)]
    print(f"targets: {len(all_pairs)}  to reproject: {len(pairs)}  "
          f"already done (skipped): {len(all_pairs) - len(pairs)}", flush=True)

    # Never overwrite an existing backup: on a re-run it would capture
    # already-reprojected files and destroy the only copy of the old basis.
    if BACKUP.exists():
        print(f"backup already exists, keeping it: {BACKUP.name} "
              f"({BACKUP.stat().st_size / 1e6:.1f} MB)")
    else:
        print(f"backing up {len(pairs)} old-basis files -> {BACKUP.name}")
        n = 0
        with tarfile.open(BACKUP, "w:gz") as tar:
            for _, _, l in pairs:
                tar.add(l, arcname=str(l.relative_to(STRATUM)))
                n += 1
                if n % 10000 == 0:
                    print(f"  backed up {n}/{len(pairs)}", flush=True)
        print(f"  backup written: {BACKUP.stat().st_size / 1e6:.1f} MB, {n} entries")
        if n != len(pairs):
            print(f"  ABORT: backup entry count {n} != {len(pairs)}")
            return 1

    t0 = time.time()
    written = errors = 0
    for i, (s, r, l) in enumerate(pairs, 1):
        try:
            v = project(np.load(r)).astype(np.float64)
            # NOTE: np.save(path) APPENDS '.npy' when the path lacks that suffix,
            # which silently breaks an atomic os.replace. Write through a file
            # handle so numpy does no name munging.
            tmp = l.with_name(l.name + ".tmp")
            with open(tmp, "wb") as fh:
                np.save(fh, v)
            os.replace(tmp, l)
            written += 1
        except Exception as e:
            errors += 1
            print(f"  ERROR {s}: {e}")
            try:
                l.with_name(l.name + ".tmp").unlink(missing_ok=True)
            except Exception:
                pass
        if i % 5000 == 0:
            print(f"  {i}/{len(pairs)}  ({time.time() - t0:.0f}s)", flush=True)

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=_PROJ,
                            capture_output=True, text=True).stdout.strip()
    MANIFEST.write_text(json.dumps({
        "reprojected_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": commit,
        "dataset_dir": str(STRATUM),
        "source_raw": str(RAW),
        "basis_fingerprint": basis_fingerprint(),
        "basis_files": [str(BASIS_LDA), str(BASIS_PREP)],
        "convention": "refit basis + L2-normalize (norm 1.0)",
        "counts": {"targets_total": len(all_pairs), "reprojected": len(pairs),
                   "skipped_already_done": len(all_pairs) - len(pairs),
                   "written": written, "errors": errors},
        "backup": str(BACKUP),
    }, indent=2))
    print(f"\nwritten {written}, errors {errors}, {time.time() - t0:.0f}s")
    print(f"manifest: {MANIFEST}")

    # Stamp every CONSUMED dir, not just the repaired one: the guard fails loud
    # on a missing stamp, and hegre_corpus (correct basis, no stamp) must be
    # stamped too or a loader enforcing the guard would reject valid data.
    print("\nstamping consumed dataset dirs (basis guard)...")
    sys.path.insert(0, str(_PROJ))
    from tools.hegre_dataset.basis_fingerprint import stamp
    for d, conv in STAMP_TARGETS:
        if d.is_dir():
            try:
                p = stamp(d, OUTPUT, convention=conv)
                print(f"  stamped  {d}  -> {p.name}  [{conv}]")
            except Exception as e:
                print(f"  STAMP FAILED {d}: {e}")
        else:
            print(f"  skipped (absent) {d}")


def clean_litter(rows):
    """Remove stray *.tmp / *.tmp.npy files left in sample dirs by a failed write."""
    removed = 0
    for s, r, l in rows:
        for cand in (l.with_name(l.name + ".tmp"), l.with_name(l.name + ".tmp.npy")):
            if cand.exists():
                try:
                    cand.unlink()
                    removed += 1
                except Exception as e:
                    print(f"  could not remove {cand}: {e}")
    print(f"removed {removed} stray tmp files")


def verify(rows):
    targets = [(s, r, l) for s, r, l in rows if r.exists() and l.exists()]
    bad, checked = [], 0
    for s, r, l in targets[:3000]:
        stored = np.load(l).astype(np.float64).ravel()
        if abs(float(np.linalg.norm(stored)) - 1.0) > 1e-6:
            bad.append((s, "norm", float(np.linalg.norm(stored))))
            continue
        if float(np.linalg.norm(stored - project(np.load(r)))) > 1e-9:
            bad.append((s, "value", None))
        checked += 1
    print(f"checked {checked}; mismatches {len(bad)}")
    for b in bad[:10]:
        print(f"  {b}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--apply", action="store_true")
    g.add_argument("--verify", action="store_true")
    g.add_argument("--clean-litter", action="store_true",
                   help="Remove stray *.tmp/*.tmp.npy files left by a failed write")
    ap.add_argument("--force", action="store_true",
                    help="Reproject even files already on the refit basis (overwrites)")
    a = ap.parse_args()
    rows = discover()
    if a.dry_run:
        dry_run(rows)
    elif a.clean_litter:
        clean_litter(rows)
    elif a.apply:
        rc = apply(rows, force=a.force)
        sys.exit(rc or 0)
    else:
        verify(rows)