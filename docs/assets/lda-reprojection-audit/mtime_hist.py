"""When was the refit-basis reprojection written? mtime histogram of high-norm files."""
import random, datetime
from pathlib import Path
from collections import Counter
import numpy as np

DATASET = Path("/mnt/nas-ai-models/training-data/eidolon/hegre-faces/v1")
lda = DATASET / "lda"
files = list(lda.rglob("*.npy"))
random.seed(7)
sample = random.sample(files, 3000)
days = Counter()
for f in sample:
    try:
        nrm = float(np.linalg.norm(np.load(f)))
    except Exception:
        continue
    d = datetime.datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d")
    days[("HIGH" if nrm >= 50 else "LOW", d)] += 1
for k in sorted(days, key=lambda x: (x[0], x[1])):
    print(f"  {k[0]:4} {k[1]}  {days[k]}")
print("DONE")
