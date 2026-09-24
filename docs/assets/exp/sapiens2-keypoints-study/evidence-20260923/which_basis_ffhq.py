#!/usr/bin/env python3
"""DECISIVE: are FFHQ stratum's stored auraface_lda.npy files on the OLD or NEW LDA basis?"""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, '/home/tim/source/activity/eidolon/experiments/geometry_pca')
from geometry_pca import auraface_preprocessing as ap

OUT = Path('/home/tim/source/activity/eidolon/experiments/geometry_pca/output')
FFHQ_RAW = Path('/mnt/nas-ai-models/training-data/ffhq/auraface')
FFHQ_STRATUM = Path('/mnt/nas-ai-models/training-data/ffhq/stratum')

NEW_PREP, NEW_LDA = OUT / 'auraface_preprocess.npz', OUT / 'auraface_lda.npz'
OLD_PREP = OUT / 'auraface_preprocess.npz.bak-20260720'
OLD_LDA = OUT / 'auraface_lda.npz.bak-20260720'

def recompute(raw, prep_path, lda_path):
    ap._REF = None
    ap._LDA = None
    ap._REF_PATH = prep_path
    ap._LDA_PATH = lda_path
    return ap.project_to_lda(ap.clean_auraface(raw))

ids = ['00000', '00001', '00002', '00003', '12345', '40000']
print(f'{"id":>7} | {"stored‖v‖":>9} | {"‖stored−NEW‖":>12} | {"‖stored−OLD‖":>12} | verdict')
print('-' * 78)
verdicts = []
for i in ids:
    raw = np.load(FFHQ_RAW / f'{i}.npy')
    stored = np.load(FFHQ_STRATUM / i / 'auraface_lda.npy').astype(np.float64)
    v_new = np.asarray(recompute(raw, NEW_PREP, NEW_LDA), dtype=np.float64).ravel()
    v_old = np.asarray(recompute(raw, OLD_PREP, OLD_LDA), dtype=np.float64).ravel()
    d_new = np.linalg.norm(stored - v_new)
    d_old = np.linalg.norm(stored - v_old)
    verdict = 'NEW' if d_new < d_old else 'OLD'
    verdicts.append(verdict)
    print(f'{i:>7} | {np.linalg.norm(stored):9.6f} | {d_new:12.8f} | {d_old:12.8f} | {verdict}')

print()
print(f'verdicts: {verdicts}')
print(f'=> FFHQ stratum identity vectors are on the {max(set(verdicts), key=verdicts.count)} basis')
print()
# sanity: do the two bases actually differ?
print('sanity — basis files differ?')
for name, p in [('new', NEW_LDA), ('old', OLD_LDA)]:
    d = np.load(p)
    print(f'  {name}: lda_basis {d["lda_basis"].shape}, evals[:3]={np.asarray(d["lda_eigenvalues"]).ravel()[:3]}')
a = np.load(NEW_LDA)['lda_basis']
b = np.load(OLD_LDA)['lda_basis']
print(f'  basis subspace overlap |W_newᵀ W_old| mean diag: {np.mean(np.abs(np.sum(a*b, axis=0))):.4f}')