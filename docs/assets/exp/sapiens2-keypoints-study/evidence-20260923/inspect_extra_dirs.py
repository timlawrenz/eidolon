#!/usr/bin/env python3
"""Inspect corpus dirs on disk that are absent from the manifest."""
import json
from pathlib import Path

CORPUS = Path('/mnt/nas-ai-models/training-data/eidolon/hegre_corpus')
man = json.loads((CORPUS / '_manifest.json').read_text())
new = set(man['samples'])
on_disk = {d.name for d in CORPUS.iterdir() if d.is_dir()}
extra = sorted(on_disk - new)
print('dirs on disk not in manifest:', len(extra))
for name in extra:
    d = CORPUS / name
    files = sorted(f.name for f in d.iterdir())
    sizes = {f: (d / f).stat().st_size for f in files}
    print(f'  {name}: files={files} sizes={sizes}')
