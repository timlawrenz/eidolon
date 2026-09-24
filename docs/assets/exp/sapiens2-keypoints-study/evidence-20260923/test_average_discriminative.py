#!/usr/bin/env python3
"""Decisive test: is the PERSONA-AVERAGE identity vector (what the corpus ships)
discriminative? Query = a held-out image's per-image LDA; index = 324 persona averages.
Compare against the same query with a per-image index (the G2-ceiling setting)."""
import json
import random
import numpy as np
from pathlib import Path
from collections import defaultdict

DATASET = Path('/mnt/nas-ai-models/training-data/eidolon/hegre-faces/v1')
CORPUS = Path('/mnt/nas-ai-models/training-data/eidolon/hegre_corpus')
AVG = DATASET / 'averages'

def l2n(v):
    return v / (np.linalg.norm(v) + 1e-12)

# --- index: persona averages
avg_files = sorted(AVG.glob('*.lda.npy'))
personas = [f.stem.replace('.lda', '') for f in avg_files]
A = l2n(np.stack([np.load(f).astype(np.float64) for f in avg_files]))  # (324,64)
pidx = {p: i for i, p in enumerate(personas)}
print(f'persona averages: {A.shape}')

# --- build a per-image sample set (one image per persona, from corpus metadata)
random.seed(42)
buckets = defaultdict(list)
for d in CORPUS.iterdir():
    if not d.is_dir():
        continue
    p = d.name.split('--', 1)[0]
    buckets[p].append(d)

queries = []   # (persona, per-image LDA vec)
for p, ds in buckets.items():
    if p not in pidx:
        continue
    random.shuffle(ds)
    for d in ds:
        meta = json.loads((d / 'metadata.json').read_text())
        lp = DATASET / 'lda' / 'faces' / p / meta['set'] / f"{meta['image_id']}.npy"
        if lp.exists():
            v = np.load(lp).astype(np.float64)
            if v.shape == (64,):
                queries.append((p, v))
                break   # one query image per persona
print(f'queries: {len(queries)}')

Q = l2n(np.stack([v for _, v in queries]))
truth = np.array([pidx[p] for p, _ in queries])

# --- Euclidean nearest-neighbour retrieval
D = np.linalg.norm(Q[:, None, :] - A[None, :, :], axis=2)      # (N,324)
rank = np.argsort(D, axis=1)
for k in (1, 5, 10):
    hit = np.mean([truth[i] in rank[i, :k] for i in range(len(Q))])
    print(f'  persona-AVERAGE index, Euclidean R@{k}: {hit:.4f}')

# --- cosine
S = Q @ A.T
rankc = np.argsort(-S, axis=1)
for k in (1, 5, 10):
    hit = np.mean([truth[i] in rankc[i, :k] for i in range(len(Q))])
    print(f'  persona-AVERAGE index, cosine    R@{k}: {hit:.4f}')

print(f'  (chance R@1 = {1/len(personas):.4f})')

# --- control: margin — distance to own average vs nearest other
own = D[np.arange(len(Q)), truth]
Dm = D.copy(); Dm[np.arange(len(Q)), truth] = np.inf
nearest_other = Dm.min(axis=1)
print(f'\n  own-avg distance : mean={own.mean():.4f}')
print(f'  nearest-other    : mean={nearest_other.mean():.4f}')
print(f'  margin (other-own) mean={np.mean(nearest_other-own):.4f}  '
      f'frac(own<other)={np.mean(own<nearest_other):.4f}')