#!/usr/bin/env python3
"""z_g validity: degenerate-projection tail vs a different encoding?"""
import os
import numpy as np
from pathlib import Path

FFHQ = Path('/mnt/nas-ai-models/training-data/ffhq/stratum')
HZG = Path('/mnt/nas-ai-models/training-data/eidolon/hegre-faces/v1/zg')
CORP = Path('/mnt/nas-ai-models/training-data/eidolon/hegre_corpus')

def load_zg(root, key='z_g.npy', n=None):
    out = []
    with os.scandir(root) as it:
        for e in it:
            if not e.is_dir(follow_symlinks=False):
                continue
            p = os.path.join(e.path, key)
            if os.path.exists(p):
                try:
                    out.append((e.name, np.load(p)))
                except Exception:
                    pass
            if n and len(out) >= n:
                break
    return out

def stats(name, Z):
    n = np.linalg.norm(Z, axis=1)
    osd = Z.std(axis=0)
    print(f'\n{name}  (n={len(Z)})')
    print(f'  per-vector norm: p1={np.percentile(n,1):6.2f} p25={np.percentile(n,25):6.2f} '
          f'p50={np.percentile(n,50):6.2f} p75={np.percentile(n,75):6.2f} '
          f'p95={np.percentile(n,95):6.2f} p99={np.percentile(n,99):6.2f} max={n.max():9.2f}')
    print(f'  mean per-dim std: {osd.mean():.4f}  (min {osd.min():.3f} max {osd.max():.3f})')
    for t in (15, 20, 25, 40, 100):
        print(f'    norm > {t:3d}: {int((n>t).sum()):6d} ({(n>t).mean()*100:6.2f}%)')
    # robust: drop the top 5% by norm and recompute
    keep = n <= np.percentile(n, 95)
    print(f'  excluding top 5% by norm: mean per-dim std = {Z[keep].std(axis=0).mean():.4f}')

ff = load_zg(FFHQ, 'z_g.npy')
stats('ffhq/stratum/{i}/z_g.npy', np.stack([v for _, v in ff]))

co = load_zg(CORP, 'z_g.npy')
stats('hegre_corpus/{sample}/z_g.npy', np.stack([v for _, v in co]))

# hegre zg/ source tree (nested)
hz = []
for f in sorted(HZG.rglob('*.npy'))[:8000]:
    try:
        hz.append(np.load(f))
    except Exception:
        pass
stats('hegre-faces/v1/zg/**/*.npy', np.stack(hz))

# corpus == zg source?
print('\n=== corpus z_g vs zg/ source (same file?) ===')
diffs, matched = [], 0
for name, v in co[:600]:
    persona, stem = name.split('--', 1)
    cands = list((HZG / 'faces' / persona).rglob(f'{stem}.npy'))
    if cands:
        diffs.append(float(np.linalg.norm(v - np.load(cands[0]))))
        matched += 1
if diffs:
    print(f'  matched {matched}; ||corpus - source|| mean={np.mean(diffs):.8f} max={np.max(diffs):.8f}')
else:
    print('  no matches found')