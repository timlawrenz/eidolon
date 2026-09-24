#!/usr/bin/env python3
"""z_g validity audit: is hegre's 1.9x std a degenerate-projection tail, or a different encoding?"""
import numpy as np
from pathlib import Path

FFHQ_STRATUM = Path('/mnt/nas-ai-models/training-data/ffhq/stratum')
FFHQ_ZG = Path('/mnt/nas-ai-models/training-data/ffhq/zg')
HEGRE_ZG = Path('/mnt/nas-ai-models/training-data/eidolon/hegre-faces/v1/zg')
CORP = Path('/mnt/nas-ai-models/training-data/eidolon/hegre_corpus')

def stats(name, Z):
    n = np.linalg.norm(Z, axis=1)
    osd = Z.std(axis=0)
    print(f'\n{name}  (n={len(Z)})')
    print(f'  per-vector norm: p1={np.percentile(n,1):.2f} p50={np.percentile(n,50):.2f} '
          f'p90={np.percentile(n,90):.2f} p99={np.percentile(n,99):.2f} max={n.max():.2f}')
    print(f'  mean per-dim std: {osd.mean():.4f}')
    for t in (15, 20, 25, 40):
        print(f'    norm > {t}: {int((n>t).sum())} ({(n>t).mean()*100:.2f}%)')

# FFHQ stratum z_g
ff = np.stack([np.load(FFHQ_STRATUM / f'{i:05d}' / 'z_g.npy') for i in range(3000)])
stats('ffhq/stratum/{i}/z_g.npy', ff)

# FFHQ zg/ tree (from extract_zg_and_averages.py)
if FFHQ_ZG.exists():
    kids = sorted(FFHQ_ZG.iterdir())[:3000]
    Z = [np.load(k / 'zg.npy') for k in kids if (k / 'zg.npy').exists()]
    if Z:
        stats('ffhq/zg/{i}/zg.npy', np.stack(Z))
else:
    print('\nffhq/zg/ does not exist')

# hegre zg/ source tree
files = sorted(HEGRE_ZG.rglob('*.npy'))[:8000]
Z = []
for f in files:
    try:
        Z.append(np.load(f))
    except Exception:
        pass
hz = np.stack(Z)
stats('hegre-faces/v1/zg/faces/**/*.npy', hz)

# hegre corpus
cdirs = [d for d in sorted(CORP.iterdir()) if d.is_dir()][:8000]
cz = np.stack([np.load(d / 'z_g.npy') for d in cdirs])
stats('hegre_corpus/*/z_g.npy', cz)

print('\n=== does the corpus match the zg/ source? ===')
# compare same-name samples
import random
random.seed(0)
matched = 0
diffs = []
for d in [d for d in sorted(CORP.iterdir()) if d.is_dir()][:400]:
    persona, stem = d.name.split('--', 1)
    # zg path: zg/faces/{persona}/{set}/{stem}.npy -- find by stem
    cands = list((HEGRE_ZG / 'faces' / persona).rglob(f'{stem}.npy'))
    if cands:
        a = np.load(d / 'z_g.npy')
        b = np.load(cands[0])
        diffs.append(float(np.linalg.norm(a - b)))
        matched += 1
print(f'  matched {matched} samples; ||corpus - zg_source||: '
      f'mean={np.mean(diffs):.6f} max={np.max(diffs):.6f}' if diffs else '  no matches found')