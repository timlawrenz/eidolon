#!/usr/bin/env python3
"""Compare auraface_lda semantics: FFHQ (per-image?) vs hegre corpus (persona-averaged?)."""
import numpy as np
from pathlib import Path

FFHQ = Path('/mnt/nas-ai-models/training-data/ffhq/stratum')
HEGRE = Path('/mnt/nas-ai-models/training-data/eidolon/hegre_corpus')

print('=== FFHQ: auraface_lda across 4 different images ===')
ff = []
for name in ['00000', '00001', '00002', '00003']:
    v = np.load(FFHQ / name / 'auraface_lda.npy')
    ff.append(v)
    print(f'  {name}: shape={v.shape} dtype={v.dtype} norm={np.linalg.norm(v):.4f}')
print(f'  pairwise cosine (0↔1): {np.dot(ff[0], ff[1]) / (np.linalg.norm(ff[0])*np.linalg.norm(ff[1])):.4f}')
print(f'  identical bytes 0 vs 1: {np.array_equal(ff[0], ff[1])}')

print()
print('=== HEGRE corpus: auraface_lda within one persona ===')
# find a persona with several samples
dirs = sorted(d.name for d in HEGRE.iterdir() if d.is_dir())
persona_counts = {}
for d in dirs:
    p = d.split('--')[0]
    persona_counts.setdefault(p, []).append(d)
# pick the persona with the most samples
top = sorted(persona_counts.items(), key=lambda kv: -len(kv[1]))[0]
pname, samples = top
print(f'  persona {pname!r}: {len(samples)} samples')
vecs = []
for d in samples[:4]:
    v = np.load(HEGRE / d / 'auraface_lda.npy')
    vecs.append(v)
    print(f'    {d}: norm={np.linalg.norm(v):.4f}')
print(f'  identical across samples? {all(np.array_equal(vecs[0], v) for v in vecs)}')

print()
print('=== HEGRE: do DIFFERENT personas differ? ===')
other = [p for p in persona_counts if p != pname][:3]
for p in other:
    d = persona_counts[p][0]
    v = np.load(HEGRE / d / 'auraface_lda.npy')
    cos = np.dot(vecs[0], v) / (np.linalg.norm(vecs[0]) * np.linalg.norm(v))
    print(f'    {p} vs {pname}: cosine={cos:.4f}')