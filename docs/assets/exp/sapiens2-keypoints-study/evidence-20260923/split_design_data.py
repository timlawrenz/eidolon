#!/usr/bin/env python3
"""(a) Do the two FFHQ z_g locations agree?  (b) corpus persona/set distribution for split design."""
import json
import os
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict

FFHQ_S = Path('/mnt/nas-ai-models/training-data/ffhq/stratum')
FFHQ_ZG = Path('/mnt/nas-ai-models/training-data/ffhq/zg')
CORP = Path('/mnt/nas-ai-models/training-data/eidolon/hegre_corpus')
MAN = CORP / '_manifest.json'

print('=== (a) ffhq/zg/{i}/zg.npy  vs  ffhq/stratum/{i}/z_g.npy ===')
print(f'  ffhq/zg exists: {FFHQ_ZG.exists()}')
if FFHQ_ZG.exists():
    n = 0
    d = []
    for i in range(400):
        sid = f'{i:05d}'
        a = FFHQ_ZG / sid / 'zg.npy'
        b = FFHQ_S / sid / 'z_g.npy'
        if a.exists() and b.exists():
            x, y = np.load(a), np.load(b)
            d.append(float(np.linalg.norm(x - y)))
            n += 1
    if d:
        print(f'  compared {n}: ||zg − z_g|| mean={np.mean(d):.8f} max={np.max(d):.8f}')
        print('  => SAME' if max(d) < 1e-6 else '  => DIFFERENT!')
    else:
        print('  no overlapping samples')

print('\n=== (b) manifest structure ===')
m = json.loads(MAN.read_text())
print(f'  keys: {list(m.keys())}')
for k, v in m.items():
    if isinstance(v, list):
        print(f'    {k}: list[{len(v)}] first={v[0] if v else None}')
    elif isinstance(v, dict):
        print(f'    {k}: dict keys={list(v.keys())[:8]}')
    else:
        print(f'    {k}: {v}')

print('\n=== (c) persona / set distribution ===')
keys = None
for k, v in m.items():
    if isinstance(v, list) and v and isinstance(v[0], str) and '--' in v[0]:
        keys = v
samples = keys or [d.name for d in CORP.iterdir() if d.is_dir()]
print(f'  using {len(samples)} samples')

personas = Counter()
sets_per_persona = defaultdict(set)
for s in samples:
    persona, stem = s.split('--', 1)
    personas[persona] += 1
    # set = stem minus trailing "-NN-RES_faceK"
    parts = stem.rsplit('-', 2)
    setname = parts[0] if len(parts) >= 3 else stem
    sets_per_persona[persona].add(setname)

imgs = np.array([personas[p] for p in personas])
nsets = np.array([len(sets_per_persona[p]) for p in personas])
print(f'  personas: {len(personas)}')
print(f'  images/persona: min={imgs.min()} p25={np.percentile(imgs,25):.0f} '
      f'median={np.median(imgs):.0f} p75={np.percentile(imgs,75):.0f} max={imgs.max()} total={imgs.sum()}')
print(f'  sets/persona:   min={nsets.min()} median={np.median(nsets):.0f} max={nsets.max()}')
for t in (1, 2, 3, 5, 10):
    print(f'    personas with >= {t} sets: {int((nsets>=t).sum())} '
          f'({(nsets>=t).mean()*100:.1f}%)  covering {int(imgs[nsets>=t].sum())} images')
print('\n  images-per-persona histogram (buckets):')
buckets = Counter()
for v in imgs:
    b = '1-9' if v < 10 else '10-24' if v < 25 else '25-49' if v < 50 else '50-99' if v < 100 else '100+'
    buckets[b] += 1
for b in ['1-9', '10-24', '25-49', '50-99', '100+']:
    print(f'    {b:>7}: {buckets[b]:4d} personas')