#!/usr/bin/env python3
"""Verify the rebuilt hegre corpus: manifest vs old dir list + sample integrity."""
import json
import random
from pathlib import Path

CORPUS = Path('/mnt/nas-ai-models/training-data/eidolon/hegre_corpus')
OLD_LIST = Path('/home/tim/.hermes/profiles/eidolon/cache/scratch/corpus_old_dirs.txt')
SCRATCH = Path('/home/tim/.hermes/profiles/eidolon/cache/scratch')

man = json.loads((CORPUS / '_manifest.json').read_text())
new = set(man['samples'])
old = set(OLD_LIST.read_text().split('\n')) - {''}

print('=== MANIFEST ===')
print('built_at          :', man['built_at'])
print('basis fingerprint :', man['lda_basis_fingerprint'])
print('params            :', man['params'])
print('counts            :', json.dumps(man['counts']))
print()

print('=== SAMPLE SET COMPARISON ===')
print('new (manifest)               :', len(new))
print('old (on disk, pre-rebuild)   :', len(old))
stale = old - new
missing = new - old
print('STALE  (in old, not in new)  :', len(stale))
print('MISSING (in new, not in old) :', len(missing))
print()

print('=== INTEGRITY: 4 required files, 500-sample check ===')
REQ = ('pixel.npy', 'auraface_lda.npy', 'z_g.npy', 'metadata.json')
random.seed(1)
bad = []
for name in random.sample(sorted(new), 500):
    d = CORPUS / name
    if not all((d / f).exists() for f in REQ):
        bad.append(name)
print('incomplete in 500-sample check:', len(bad))
if bad:
    print('examples:', bad[:5])
print()

# on-disk dir count vs manifest
on_disk = {d.name for d in CORPUS.iterdir() if d.is_dir()}
print('on-disk sample dirs          :', len(on_disk))
print('manifest samples             :', len(new))
print('dirs on disk not in manifest :', len(on_disk - new))
print('manifest samples not on disk :', len(new - on_disk))
print()

SCRATCH.joinpath('corpus_stale_dirs.txt').write_text('\n'.join(sorted(stale)))
print('stale list written ->', SCRATCH / 'corpus_stale_dirs.txt')
print('sample of stale dirs:', sorted(stale)[:5])
