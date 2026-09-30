"""Missing-artifact scan over approved images. Read-only, NFS-heavy."""
import sys
from pathlib import Path
import numpy as np

ROOT = Path("/home/tim/source/activity/eidolon")
DATASET = Path("/mnt/nas-ai-models/training-data/eidolon/hegre-faces/v1")
sys.path.insert(0, str(ROOT))
from tools.hegre_dataset.dataset import HegreDataset

ds = HegreDataset(DATASET)
paths = ds.db.execute("SELECT image_path FROM images WHERE status='approved'").fetchall()
n_app = len(paths)
print(f"approved images: {n_app}", flush=True)

stratum = DATASET / "stratum"
auraface = DATASET / "auraface"
lda = DATASET / "lda"
zg = DATASET / "zg"

miss_af = miss_lda = miss_lda_no_af = miss_zg = miss_pose = 0
norm_samples = []
for i, r in enumerate(paths):
    rel = Path(r["image_path"])
    af_f = auraface / rel.with_suffix(".npy")
    lda_f = lda / rel.with_suffix(".npy")
    has_af = af_f.exists()
    if not has_af:
        miss_af += 1
    if not lda_f.exists():
        miss_lda += 1
        if not has_af:
            miss_lda_no_af += 1
    elif len(norm_samples) < 5:
        v = np.load(lda_f)
        norm_samples.append((str(rel), v.shape, v.dtype.str, float(np.linalg.norm(v))))
    if not (zg / rel.with_suffix(".npy")).exists():
        miss_zg += 1
    if not (stratum / rel.parent / rel.stem / "pose.npy").exists():
        miss_pose += 1
    if i and i % 20000 == 0:
        print(f"  scanned {i}/{n_app} ...", flush=True)

print(f"missing auraface .npy:   {miss_af}", flush=True)
print(f"missing LDA .npy:        {miss_lda}", flush=True)
print(f"  of those no auraface:  {miss_lda_no_af}  (not projectable)", flush=True)
print(f"missing z_g .npy:        {miss_zg}", flush=True)
print(f"missing pose.npy:        {miss_pose}", flush=True)
print("existing LDA samples:", flush=True)
for s in norm_samples:
    print("   ", s, flush=True)
print("SCAN_DONE", flush=True)
