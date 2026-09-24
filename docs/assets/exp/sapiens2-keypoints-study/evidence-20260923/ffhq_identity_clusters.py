#!/usr/bin/env python3
"""Does FFHQ contain repeat identities (persona structure), or ~1 image per identity?
Uses RAW AuraFace (512-d) so the LDA compression doesn't hide duplicates."""
import numpy as np
from pathlib import Path
import itertools

RAW = Path('/mnt/nas-ai-models/training-data/ffhq/auraface')
CACHE = Path('/home/tim/.hermes/profiles/eidolon/cache/scratch/ffhq_raw_sample.npz')

N = 3000
if CACHE.exists():
    Z = np.load(CACHE)['Z']
    print(f'loaded cached {Z.shape}')
else:
    ids = sorted(p.stem for p in RAW.glob('*.npy'))[:N]
    Z = np.stack([np.load(RAW / f'{i}.npy').astype(np.float64) for i in ids])
    Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    np.savez(CACHE, Z=Z)
    print(f'computed {Z.shape}')

S = Z @ Z.T
np.fill_diagonal(S, -np.inf)
# distribution of each vector's nearest neighbour
nn = S.max(axis=1)
print(f'FFHQ raw-AuraFace NN cosine (n={len(Z)}):')
print(f'  mean={nn.mean():.4f}  p50={np.median(nn):.4f}  p95={np.percentile(nn,95):.4f}  max={nn.max():.4f}')

# count pairs above identity thresholds
for t in (0.5, 0.6, 0.7):
    n_pairs = int((S > t).sum() // 2)
    involved = len(set(np.where(S > t)[0].tolist()) | set(np.where(S > t)[1].tolist()))
    print(f'  pairs with cosine > {t}: {n_pairs}  ({involved} vectors involved, '
          f'{100.0*involved/len(Z):.1f}% of the sample)')

print()
print('interpretation: if pairs>0.6 are ~0, FFHQ has essentially one image per identity')
print('=> it CANNOT provide a persona-level identity target (hegre can: 100 imgs/persona).')