#!/usr/bin/env python3
"""Is the corpus identity target degenerate? Compare within- vs between-persona
separation for PER-IMAGE LDA vectors vs PERSONA AVERAGES."""
import numpy as np
from pathlib import Path
import itertools, random

DATASET = Path('/mnt/nas-ai-models/training-data/eidolon/hegre-faces/v1')
CORPUS = Path('/mnt/nas-ai-models/training-data/eidolon/hegre_corpus')
AVG = DATASET / 'averages'

def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

# --- 0. sanity: corpus file == averages/ file?
ad = sorted(d for d in CORPUS.iterdir() if d.is_dir() and d.name.startswith('adriana--'))[0]
c = np.load(ad / 'auraface_lda.npy')
a = np.load(AVG / 'adriana.lda.npy')
print(f'corpus==averages for adriana: {np.allclose(c, a, atol=1e-6)}')
print(f'  corpus dtype={c.dtype}  avg dtype={a.dtype}')
print()

# --- 1. persona averages across many personas
avg_files = sorted(AVG.glob('*.lda.npy'))
names = [f.stem.replace('.lda', '') for f in avg_files]
random.seed(0)
sel = random.sample(list(zip(names, avg_files)), 30)
vecs = {n: np.load(p) for n, p in sel}
print(f'=== persona AVERAGES: norms ===')
norms = [np.linalg.norm(v) for v in vecs.values()]
print(f'  min={min(norms):.4f} max={max(norms):.4f}')
ks = [n for n, _ in sel]
sims = [cos(np.load(AVG/f'{a}.lda.npy'), np.load(AVG/f'{b}.lda.npy')) for a, b in itertools.combinations(ks, 2)]
print(f'=== persona AVERAGES: pairwise cosine (n={len(sims)}) ===')
print(f'  mean={np.mean(sims):.4f}  min={np.min(sims):.4f}  max={np.max(sims):.4f}')
print()

# --- 2. per-image LDA: within vs between persona
print('=== PER-IMAGE LDA (from dataset lda/ tree) ===')
personas = ['adriana', 'alba', 'aleksandra', 'alena', 'ariel']
samples = {}
for p in personas:
    # find that persona's images via corpus metadata (has persona name in dir)
    dirs = sorted(d for d in CORPUS.iterdir() if d.is_dir() and d.name.startswith(p + '--'))[:6]
    vs = []
    for d in dirs:
        stem = d.name.split('--', 1)[1]
        setname = None
        import json
        meta = json.loads((d / 'metadata.json').read_text())
        setname = meta['set']
        lda_path = DATASET / 'lda' / 'faces' / p / setname / f'{stem}.npy'
        if lda_path.exists():
            vs.append(np.load(lda_path))
    samples[p] = vs
    print(f'  {p}: {len(vs)} per-image LDA vectors loaded')

within = []
for p, vs in samples.items():
    within += [cos(a, b) for a, b in itertools.combinations(vs, 2) if len(vs) > 1]
between = []
for (p1, v1), (p2, v2) in itertools.combinations(samples.items(), 2):
    for a in v1[:3]:
        for b in v2[:3]:
            between.append(cos(a, b))
if within:
    print(f'  WITHIN-persona  cosine: mean={np.mean(within):.4f}  min={np.min(within):.4f}  max={np.max(within):.4f}')
print(f'  BETWEEN-persona cosine: mean={np.mean(between):.4f}  min={np.min(between):.4f}  max={np.max(between):.4f}')
print()
print('  (separation exists if WITHIN >> BETWEEN)')