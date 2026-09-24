"""Verify whether existing per-image LDA files match the CURRENT refitted basis."""
import os
import random
import sys
from pathlib import Path
import numpy as np

REPO = Path("/home/tim/source/activity/eidolon")
ROOT = Path("/mnt/nas-ai-models/training-data/eidolon/hegre-faces/v1")
sys.path.insert(0, str(REPO / "experiments" / "geometry_pca"))

for art in ["auraface_preprocess.npz", "auraface_lda.npz"]:
    for cand in [
        REPO / "experiments" / "geometry_pca" / "output" / art,
        REPO / "output" / art,
    ]:
        if cand.exists():
            st = cand.stat()
            import datetime
            print(f"{art}: {cand}  mtime={datetime.datetime.fromtimestamp(st.st_mtime)}  size={st.st_size}")
            break
    else:
        print(f"{art}: NOT FOUND in expected locations")

# sample mtimes of lda npy files
lda_dir = ROOT / "lda"
sample_mtimes = []
count = 0
for dirpath, dirnames, filenames in os.walk(lda_dir):
    for fn in filenames:
        if fn.endswith(".npy"):
            count += 1
            if random.random() < 0.001:
                sample_mtimes.append(os.path.getmtime(os.path.join(dirpath, fn)))
sample_mtimes.sort()
import datetime
if sample_mtimes:
    print(f"lda npy sample (n={len(sample_mtimes)} of {count}):")
    print("  oldest:", datetime.datetime.fromtimestamp(sample_mtimes[0]))
    print("  median:", datetime.datetime.fromtimestamp(sample_mtimes[len(sample_mtimes) // 2]))
    print("  newest:", datetime.datetime.fromtimestamp(sample_mtimes[-1]))

# Re-projection equivalence check on random existing pairs
from geometry_pca.auraface_preprocessing import clean_auraface, project_to_lda  # noqa: E402

print("---- basis match check ----")
checked = 0
mismatch = 0
for dirpath, dirnames, filenames in os.walk(lda_dir):
    for fn in filenames:
        if not fn.endswith(".npy"):
            continue
        if random.random() > 0.00008:
            continue
        lda_p = Path(dirpath) / fn
        rel = lda_p.relative_to(lda_dir)
        af_p = ROOT / "auraface" / rel
        if not af_p.exists():
            continue
        af = np.load(af_p)
        existing = np.load(lda_p)
        v_clean = clean_auraface(af)
        new_lda = project_to_lda(v_clean)
        ok = np.allclose(existing, new_lda, atol=1e-5)
        checked += 1
        if not ok:
            mismatch += 1
        print(f"  {'MATCH' if ok else 'STALE'}  {rel}  maxdiff={np.max(np.abs(np.asarray(existing, dtype=np.float64) - np.asarray(new_lda, dtype=np.float64))):.3e}")
        if checked >= 8:
            break
    if checked >= 8:
        break
print(f"checked={checked} mismatch={mismatch}")
