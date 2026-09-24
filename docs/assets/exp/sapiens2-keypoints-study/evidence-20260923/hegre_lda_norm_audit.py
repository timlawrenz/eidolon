#!/usr/bin/env python3
"""Are there stale/normalized leftovers in the hegre per-image lda tree (expect norm ~153)?"""
import numpy as np
from pathlib import Path

D = Path('/mnt/nas-ai-models/training-data')
LDA = D / 'eidolon/hegre-faces/v1/lda'

files = sorted(LDA.rglob('*.npy'))
print(f'total per-image lda files: {len(files)}')
lo, mid, hi = [], [], []
for i, p in enumerate(files):
    if i % 25000 == 0:
        print(f'  scanned {i}...', flush=True)
    try:
        n = float(np.linalg.norm(np.load(p)))
    except Exception:
        continue
    if n < 10:
        lo.append((n, str(p)))
    elif n < 100:
        mid.append((n, str(p)))
    else:
        hi.append(n)

print(f'\n  norm ~150+ (correct, new basis): {len(hi)}')
print(f'  norm <10  (SUSPECT: old basis or zero): {len(lo)}')
print(f'  norm 10-100 (SUSPECT: intermediate):    {len(mid)}')
if hi:
    print(f'  hi norms: mean={np.mean(hi):.2f} min={np.min(hi):.2f} max={np.max(hi):.2f}')
for n, p in lo[:10]:
    print(f'    low {n:.6f}  {p.replace(str(D), "...")}')