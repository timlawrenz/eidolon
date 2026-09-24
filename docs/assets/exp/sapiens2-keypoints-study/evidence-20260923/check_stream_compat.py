#!/usr/bin/env python3
"""Are FFHQ and hegre-corpus conditioning streams compatible?
  (a) z_g 50-d scale/shape comparison
  (b) identity 64-d magnitude mismatch (the proven basis split)"""
import json
import numpy as np
from pathlib import Path

FFHQ = Path('/mnt/nas-ai-models/training-data/ffhq/stratum')
CORP = Path('/mnt/nas-ai-models/training-data/eidolon/hegre_corpus')

def sample(dirpath, n=200, pattern='[0-9]*'):
    ids = sorted(d.name for d in dirpath.iterdir() if d.is_dir())[:n]
    return ids

print('=== z_g (50-d) ===')
ff_ids = [f'{i:05d}' for i in range(200)]
ff_zg = np.stack([np.load(FFHQ / i / 'z_g.npy') for i in ff_ids])
corp_dirs = [d for d in sorted(CORP.iterdir()) if d.is_dir()][:200]
co_zg = np.stack([np.load(d / 'z_g.npy') for d in corp_dirs])
for name, Z in [('FFHQ ', ff_zg), ('hegre', co_zg)]:
    print(f'  {name}: shape={Z.shape} dtype={Z.dtype} global_mean={Z.mean():+.5f} '
          f'global_std={Z.std():.5f} per_vec_norm_mean={np.linalg.norm(Z,axis=1).mean():.4f}')
print(f'  per-dim mean correlation: {np.corrcoef(ff_zg.mean(0), co_zg.mean(0))[0,1]:.4f}')
print()

print('=== identity auraface_lda (64-d) magnitude ===')
ff_af = np.stack([np.load(FFHQ / i / 'auraface_lda.npy').astype(np.float64) for i in ff_ids[:200]])
co_af = np.stack([np.load(d / 'auraface_lda.npy').astype(np.float64) for d in corp_dirs])
for name, A in [('FFHQ ', ff_af), ('hegre', co_af)]:
    n = np.linalg.norm(A, axis=1)
    print(f'  {name}: norm mean={n.mean():.6f} std={n.std():.6f} min={n.min():.6f} max={n.max():.6f}')
# scale ratio
r = np.linalg.norm(co_af, axis=1).mean() / np.linalg.norm(ff_af, axis=1).mean()
print(f'  => hegre/FFHQ identity magnitude ratio: {r:.3f}x')
print()
# value ranges
print(f'  FFHQ  value range: [{ff_af.min():+.3f}, {ff_af.max():+.3f}]')
print(f'  hegre value range: [{co_af.min():+.3f}, {co_af.max():+.3f}]')
print()
print('  interpretation: same conditioning slot (identity_dim=64) receiving two')
print('  different distributions = the model sees a domain/task switch, not identity.')