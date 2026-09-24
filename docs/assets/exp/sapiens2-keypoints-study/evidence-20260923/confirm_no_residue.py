#!/usr/bin/env python3
"""Confirm the killed first attempt (proc_7c0f085e801f) left no residue."""
import random
import numpy as np
from pathlib import Path

S = Path('/mnt/nas-ai-models/training-data/ffhq/stratum')
dirs = sorted(d for d in S.iterdir() if d.is_dir() and not d.name.startswith(('@', '_')))
print(f'dirs: {len(dirs)}')

# 1. litter scan across ALL dirs (stat only, cheap)
litter = 0
for d in dirs:
    for name in ('auraface_lda.npy.tmp', 'auraface_lda.npy.tmp.npy'):
        if (d / name).exists():
            litter += 1
print(f'stray tmp files remaining: {litter}')

# 2. norms on a random sample
random.seed(7)
sample = random.sample(dirs, 400)
bad = n = 0
for d in sample:
    p = d / 'auraface_lda.npy'
    if not p.exists():
        continue
    v = np.load(p)
    n += 1
    if v.shape != (64,) or abs(float(np.linalg.norm(v)) - 1.0) > 1e-6:
        bad += 1
print(f'random {n}: off-unit-norm or wrong shape = {bad}')

# 3. any old-basis survivors anywhere? (norm ~0.35)
old = 0
for d in random.sample(dirs, 1500):
    p = d / 'auraface_lda.npy'
    if p.exists():
        nv = float(np.linalg.norm(np.load(p)))
        if 0.1 < nv < 0.9:
            old += 1
print(f'old-basis survivors (norm 0.1-0.9) in 1500 random: {old}')

# 4. stamps
import json
for name, p in [('ffhq/stratum', S / 'BASIS_FINGERPRINT.json'),
                ('hegre_corpus', Path('/mnt/nas-ai-models/training-data/eidolon/hegre_corpus/BASIS_FINGERPRINT.json'))]:
    print(f'{name}: {"stamped " + json.loads(p.read_text())["basis_fingerprint"] if p.exists() else "NO STAMP"}')