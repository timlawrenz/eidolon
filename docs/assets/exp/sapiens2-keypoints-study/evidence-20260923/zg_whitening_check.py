#!/usr/bin/env python3
"""Is z_g whitened (per-dim std ~1.0) in both datasets? If not, they're different encodings."""
import numpy as np
from pathlib import Path

FFHQ = Path('/mnt/nas-ai-models/training-data/ffhq/stratum')
CORP = Path('/mnt/nas-ai-models/training-data/eidolon/hegre_corpus')
SRC = Path('/mnt/nas-ai-models/training-data/eidolon/hegre-faces/v1')

def load_many(dirs, n=None):
    ds = [d for d in sorted(dirs) if d.is_dir()]
    if n:
        ds = ds[:n]
    Z = []
    for d in ds:
        p = d / 'z_g.npy'
        if p.exists():
            Z.append(np.load(p))
    return np.stack(Z) if Z else None

ff = load_many([FFHQ / f'{i:05d}' for i in range(600)])
co = load_many([d for d in CORP.iterdir()])
print(f'FFHQ      n={len(ff)}  shape={ff.shape}')
print(f'hegre_corpus n={len(co)}  shape={co.shape}')

for name, Z in [('FFHQ stratum', ff), ('hegre_corpus', co)]:
    osd = Z.std(axis=0)              # per-dim std
    print(f'\n{name}:')
    print(f'  global std       = {Z.std():.4f}')
    print(f'  mean per-dim std = {osd.mean():.4f}   (min {osd.min():.4f}, max {osd.max():.4f})')
    print(f'  dim means |mean| = {np.abs(Z.mean(axis=0)).mean():.4f}')
    print(f'  per-vector norm  = {np.linalg.norm(Z,axis=1).mean():.3f}')
    print(f'  expected if whitened: per-dim std ~ 1.0')

print('\n=== source hegre-faces/v1 z_g (where is it?) ===')
for sub in ['zg', 'stratum']:
    p = SRC / sub
    if p.exists():
        kids = [d for d in p.iterdir() if d.is_dir()][:3]
        print(f'  {p}: {len([d for d in p.iterdir()])} entries; sample {[d.name for d in kids]}')
        for k in kids[:2]:
            for f in list(k.rglob('z_g.npy'))[:1]:
                print(f'    {f}  size={f.stat().st_size} mtime={f.stat().st_mtime}')