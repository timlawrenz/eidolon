#!/usr/bin/env python3
"""Why are 4 complete sample dirs absent from the manifest? Check eligibility."""
import json
import os
from pathlib import Path

os.environ.setdefault('EIDOLON_SKIP_REVIEWDB_GUARD', '1')
import sys
sys.path.insert(0, '/home/tim/source/activity/eidolon')

from tools.hegre_dataset.dataset import HegreDataset

DATASET = Path('/mnt/nas-ai-models/training-data/eidolon/hegre-faces/v1')
CORPUS = Path('/mnt/nas-ai-models/training-data/eidolon/hegre_corpus')

man = json.loads((CORPUS / '_manifest.json').read_text())
new = set(man['samples'])
on_disk = {d.name for d in CORPUS.iterdir() if d.is_dir()}
extra = sorted(on_disk - new)

ds = HegreDataset(DATASET)
name_to_persona = {p.name.lower(): p for p in ds.personas.values()}

for dir_name in extra:
    pname, _, stem = dir_name.partition('--')
    p = name_to_persona.get(pname.lower())
    print(f'--- {dir_name}')
    if p is None:
        print('    persona NOT FOUND in DB')
        continue
    rows = ds.db.execute(
        "SELECT image_path, status FROM images WHERE persona_id = ? AND image_path LIKE ?",
        (p.id, f'%/{stem}.jpg')
    ).fetchall()
    if not rows:
        print(f'    persona={p.name} (id={p.id}) — NO image row matching stem {stem!r}')
        continue
    for image_path, status in rows:
        zg = DATASET / 'zg' / image_path.replace('.jpg', '.npy')
        af = DATASET / 'auraface' / image_path.replace('.jpg', '.npy')
        print(f'    persona={p.name} status={status}')
        print(f'      image_path={image_path}')
        print(f'      z_g exists={zg.exists()}   auraface exists={af.exists()}')
