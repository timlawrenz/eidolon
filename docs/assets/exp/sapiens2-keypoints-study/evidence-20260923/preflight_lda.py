"""Pre-flight for Hegre LDA reprojection: exact missing-LDA count among approved images.

Avoids full-tree rglob where possible: uses os.walk for the auraface/lda trees
(directory traversal, not per-file stats over SMB).
"""
import os
import sys
from pathlib import Path

ROOT = Path("/mnt/nas-ai-models/training-data/eidolon/hegre-faces/v1")

sys.path.insert(0, "/home/tim/source/activity/eidolon")
from tools.hegre_dataset.dataset import HegreDataset  # noqa: E402

ds = HegreDataset(ROOT)
rows = ds.db.execute(
    "SELECT image_path FROM images WHERE status = 'approved'"
).fetchall()
approved = [r["image_path"] for r in rows]
print(f"approved images (PG): {len(approved)}")

af_dir = ROOT / "auraface"
lda_dir = ROOT / "lda"


def collect(base):
    out = set()
    n = 0
    for dirpath, _dirnames, filenames in os.walk(base):
        for fn in filenames:
            if fn.endswith(".npy"):
                full = Path(dirpath) / fn
                out.add(str(full.relative_to(base)))
                n += 1
    return out, n


af_rel, af_n = collect(af_dir)
print(f"auraface .npy files on disk: {af_n}")
lda_rel, lda_n = collect(lda_dir)
print(f"lda .npy files on disk: {lda_n}")

# approved images (image_path like faces/persona/set/img.jpg) -> .npy rel path
appr_npy = {str(Path(p).with_suffix(".npy")) for p in approved}

missing_af = appr_npy - af_rel
missing_lda = (appr_npy & af_rel) - lda_rel
extra_lda = lda_rel - af_rel

print(f"approved missing auraface: {len(missing_af)}")
print(f"approved missing lda (has af): {len(missing_lda)}")
print(f"lda files with no auraface counterpart: {len(extra_lda)}")

# any lda file whose auraface exists but image not approved?
lda_nonapproved = lda_rel - appr_npy
print(f"lda files not corresponding to an approved image: {len(lda_nonapproved)}")

for name, s in (("missing_af", missing_af), ("missing_lda", missing_lda)):
    print(f"--- sample {name} ---")
    for x in list(sorted(s))[:10]:
        print("   ", x)
