#!/usr/bin/env python3
"""Pin the target convention for FFHQ's auraface_lda.npy:
   compare norms across hegre per-image lda tree / hegre corpus / ffhq (old basis),
   and preview what FFHQ looks like reprojected to the new basis (raw + normalized)."""
import sys, os
import numpy as np
from pathlib import Path

sys.path.insert(0, '/home/tim/source/activity/eidolon/experiments/geometry_pca')
from geometry_pca import auraface_preprocessing as ap

OUT = Path('/home/tim/source/activity/eidolon/experiments/geometry_pca/output')
D = Path('/mnt/nas-ai-models/training-data')

def norms_of(paths, label, limit=1500):
    ns, vals = [], []
    for p in paths[:limit]:
        try:
            v = np.load(p).astype(np.float64).ravel()
        except Exception:
            continue
        ns.append(np.linalg.norm(v)); vals.append(v)
    if not ns:
        print(f'  {label}: none'); return None, None
    ns = np.array(ns)
    print(f'  {label:38s} n={len(ns):5d}  norm mean={ns.mean():.6f} min={ns.min():.6f} max={ns.max():.6f}')
    return ns, np.stack(vals)

print('=== stored auraface_lda.npy norms by location ===')
# hegre per-image lda tree (reprojected 2026-09-22, NEW basis)
hegre_lda = sorted((D / 'eidolon/hegre-faces/v1/lda').rglob('*.npy'))
norms_of(hegre_lda, 'hegre lda/faces/** (per-image)')

# hegre corpus (persona average)
corp = sorted(d / 'auraface_lda.npy' for d in (D / 'eidolon/hegre_corpus').iterdir() if d.is_dir())
norms_of(corp, 'hegre_corpus/* (persona avg)')

# ffhq stratum (OLD basis, per-image)
ffhq = sorted((D / 'ffhq/stratum').glob('*/auraface_lda.npy'))
norms_of(ffhq, 'ffhq/stratum/* (per-image, OLD)')

print('\n=== preview: FFHQ reprojected to NEW basis ===')
RAW = D / 'ffhq/auraface'
ids = [f'{i:05d}' for i in range(250)]
raws, olds = [], []
for i in ids:
    rp = RAW / f'{i}.npy'
    sp = D / 'ffhq/stratum' / i / 'auraface_lda.npy'
    if rp.exists() and sp.exists():
        raws.append(np.load(rp).astype(np.float64))
        olds.append(np.load(sp).astype(np.float64))
raws = np.stack(raws); olds = np.stack(olds)

new_coords = np.stack([ap.project_to_lda(ap.clean_auraface(r)) for r in raws])
nn = np.linalg.norm(new_coords, axis=1)
print(f'  reprojected (raw, unnormalized): norm mean={nn.mean():.6f} '
      f'min={nn.min():.6f} max={nn.max():.6f}')
print(f'  reprojected + L2-normalized    : norm = 1.000000 (by construction)')
print()
print(f'  OLD stored norms (same 250)    : mean={np.linalg.norm(olds,axis=1).mean():.6f}')
print()
# does normalizing preserve identity structure? check pairwise cosine spread
def cos(a, b): return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
raw_pairs = [cos(new_coords[i], new_coords[j]) for i in range(60) for j in range(i+1, 60)]
nrm = new_coords / np.linalg.norm(new_coords, axis=1, keepdims=True)
nrm_pairs = [cos(nrm[i], nrm[j]) for i in range(60) for j in range(i+1, 60)]
print(f'  between-image cosine, unnormalized: mean={np.mean(raw_pairs):.4f} std={np.std(raw_pairs):.4f}')
print(f'  between-image cosine, normalized  : mean={np.mean(nrm_pairs):.4f} std={np.std(nrm_pairs):.4f}')
print('  (identical by construction — L2 normalization does not change cosine geometry)')