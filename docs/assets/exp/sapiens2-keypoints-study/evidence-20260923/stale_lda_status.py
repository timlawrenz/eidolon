#!/usr/bin/env python3
"""Are the 2,220 stale per-image LDA files in the approved (consumed) set?"""
import os, sys
import numpy as np
from pathlib import Path

sys.path.insert(0, '/home/tim/source/activity/eidolon')
os.environ.setdefault('EIDOLON_SKIP_REVIEWDB_GUARD', '1')
from tools.hegre_dataset.dataset import HegreDataset

D = Path('/mnt/nas-ai-models/training-data/eidolon/hegre-faces/v1')
LDA = D / 'lda'

stale = []
for p in LDA.rglob('*.npy'):
    try:
        if float(np.linalg.norm(np.load(p))) < 10:
            stale.append(p)
    except Exception:
        pass
print(f'stale per-image lda files: {len(stale)}')

# map to DB image_path: lda/faces/{persona}/{set}/{stem}.npy -> faces/{persona}/{set}/{stem}.jpg
rel = []
for p in stale:
    r = p.relative_to(LDA)                      # faces/.../x.npy
    rel.append(str(r.with_suffix('.jpg')))
    assert rel[-1].startswith('faces/'), rel[-1]

ds = HegreDataset(D)
ph = ','.join('?' * len(rel))
rows = ds.db.execute(
    f"SELECT image_path, status FROM images WHERE image_path IN ({ph})", rel
).fetchall()
from collections import Counter
c = Counter(s for _, s in rows)
print(f'  found in DB: {len(rows)} / {len(rel)}')
print(f'  status breakdown: {dict(c)}')
print(f'  => {"IN the approved/consumed path" if c.get(chr(97)+"pproved") else "NOT approved"}')