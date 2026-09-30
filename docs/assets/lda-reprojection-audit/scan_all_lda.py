"""Full scan of the per-image lda dir: norm distribution + outlier identification.
Read-only. Slow (NFS)."""
import sys
from pathlib import Path
import numpy as np
import time

DATASET = Path("/mnt/nas-ai-models/training-data/eidolon/hegre-faces/v1")
lda = DATASET / "lda"

t0 = time.time()
files = list(lda.rglob("*.npy"))
print(f"files: {len(files)}  (rglob {time.time()-t0:.0f}s)", flush=True)

lo, hi = [], []
norm_min, norm_max = 1e9, -1e9
n = 0
bad = 0
shape_counts = {}
for i, f in enumerate(files):
    try:
        v = np.load(f)
        nrm = float(np.linalg.norm(v))
    except Exception:
        bad += 1
        continue
    shape_counts[v.shape] = shape_counts.get(v.shape, 0) + 1
    n += 1
    if nrm < norm_min: norm_min = nrm
    if nrm > norm_max: norm_max = nrm
    if nrm < 50.0:
        lo.append((str(f.relative_to(lda)), nrm, f.stat().st_mtime))
    if nrm > 400.0:
        hi.append((str(f.relative_to(lda)), nrm, f.stat().st_mtime))
    if i and i % 50000 == 0:
        print(f"  scanned {i}/{len(files)}  (elapsed {time.time()-t0:.0f}s)", flush=True)

print(f"loaded {n}, errors {bad}", flush=True)
print("shapes:", shape_counts, flush=True)
print(f"norm min {norm_min:.4f}  max {norm_max:.4f}", flush=True)
print(f"norm<50  count = {len(lo)}  ({100.0*len(lo)/max(n,1):.3f}%)", flush=True)
print(f"norm>400 count = {len(hi)}", flush=True)
import datetime
for rel, nrm, mt in lo[:60]:
    print(f"   LOW {nrm:10.4f}  {datetime.datetime.fromtimestamp(mt):%Y-%m-%d %H:%M}  {rel}", flush=True)
if len(lo) > 60:
    print(f"   ... and {len(lo)-60} more", flush=True)
# mtime histogram of low-norm files
if lo:
    mt_days = {}
    for _, _, mt in lo:
        d = datetime.datetime.fromtimestamp(mt).strftime("%Y-%m-%d")
        mt_days[d] = mt_days.get(d, 0) + 1
    print("low-norm mtime days:", dict(sorted(mt_days.items())), flush=True)
print("DONE", flush=True)
