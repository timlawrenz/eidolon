#!/usr/bin/env python3
"""ADVERSARIAL audit of the FFHQ reprojection — full coverage, not a sample.

Tries to falsify the PASS:
  1. full scan of all 69,960 files (verify() only sampled the first 3,000)
  2. NaN / Inf / wrong-shape / degenerate (near-zero) detection
  3. independent recomputation from raw on a RANDOM sample (not the first N)
  4. mtime check: every file must have been rewritten today
  5. does the new encoding actually match hegre_corpus's convention?
"""
import random
import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, '/home/tim/source/activity/eidolon')
sys.path.insert(0, '/home/tim/source/activity/eidolon/experiments/geometry_pca')
from geometry_pca.auraface_preprocessing import clean_auraface, project_to_lda  # noqa

D = Path('/mnt/nas-ai-models/training-data')
STRATUM = D / 'ffhq/stratum'
RAW = D / 'ffhq/auraface'
CORP = D / 'eidolon/hegre_corpus'

dirs = sorted(d for d in STRATUM.iterdir() if d.is_dir() and not d.name.startswith(('@', '_')))
print(f'scanning {len(dirs)} dirs (FULL coverage)')
t0 = time.time()

n_ok = n_norm_bad = n_nan = n_shape = n_zero = n_missing = 0
norms = []
mtimes = set()
for i, d in enumerate(dirs, 1):
    p = d / 'auraface_lda.npy'
    if not p.exists():
        n_missing += 1
        continue
    try:
        v = np.load(p)
    except Exception:
        n_shape += 1
        continue
    if v.shape != (64,):
        n_shape += 1
        continue
    if not np.isfinite(v).all():
        n_nan += 1
        continue
    nv = float(np.linalg.norm(v))
    norms.append(nv)
    if nv < 1e-6:
        n_zero += 1
    elif abs(nv - 1.0) > 1e-6:
        n_norm_bad += 1
    else:
        n_ok += 1
    mtimes.add(int(p.stat().st_mtime) // 86400)
    if i % 20000 == 0:
        print(f'  {i}/{len(dirs)}  ({time.time()-t0:.0f}s)', flush=True)

norms = np.array(norms)
print(f'\n=== 1. FULL SCAN ({time.time()-t0:.0f}s) ===')
print(f'  unit norm (correct)     : {n_ok}')
print(f'  norm != 1              : {n_norm_bad}')
print(f'  near-zero / degenerate  : {n_zero}')
print(f'  NaN / Inf               : {n_nan}')
print(f'  wrong shape / unreadable: {n_shape}')
print(f'  missing file            : {n_missing}')
print(f'  norm min={norms.min():.9f} max={norms.max():.9f} mean={norms.mean():.9f}')
print(f'  distinct mtime days: {sorted(mtimes)}')

print('\n=== 2. independent recomputation, RANDOM sample (not the first N) ===')
random.seed(1234)
sample = random.sample(dirs, 250)
bad = 0
for d in sample:
    sid = d.name
    rp = RAW / f'{sid}.npy'
    if not rp.exists():
        continue
    v = np.load(d / 'auraface_lda.npy').astype(np.float64)
    exp = project_to_lda(clean_auraface(np.load(rp))).astype(np.float64)
    exp = exp / (np.linalg.norm(exp) + 1e-12)
    if np.linalg.norm(v - exp) > 1e-9:
        bad += 1
print(f'  random 250: mismatches vs recomputed = {bad}')

print('\n=== 3. does the new encoding match hegre_corpus convention? ===')
cdirs = [d for d in sorted(CORP.iterdir()) if d.is_dir()][:800]
cn = np.array([np.linalg.norm(np.load(d / 'auraface_lda.npy')) for d in cdirs])
print(f'  ffhq  norm mean={norms.mean():.9f}')
print(f'  hegre norm mean={cn.mean():.9f}')
print(f'  => {"CONSISTENT" if abs(norms.mean()-cn.mean()) < 1e-6 else "STILL MISMATCHED"}')