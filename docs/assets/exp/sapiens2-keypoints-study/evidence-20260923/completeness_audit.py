#!/usr/bin/env python3
"""Full completeness audit: per-stream file presence across both datasets."""
import json
import os
import sys
import numpy as np
from pathlib import Path
from collections import Counter

CORP = Path('/mnt/nas-ai-models/training-data/eidolon/hegre_corpus')
FFHQ = Path('/mnt/nas-ai-models/training-data/ffhq/stratum')
OUT = Path('/home/tim/.hermes/profiles/eidolon/cache/scratch/completeness.json')

CORP_STREAMS = ['pixel.npy', 'auraface_lda.npy', 'z_g.npy', 'metadata.json']
FFHQ_STREAMS = ['pixel.npy', 'pose.npy', 'z_g.npy', 'auraface_lda.npy',
                'dinov3_patches.npy', 't5_hidden.npy', 'caption.txt', 'flux_latent.npy']

def scan(root, streams, label, limit=None):
    present = Counter()
    total = 0
    missing_examples = {s: [] for s in streams}
    with os.scandir(root) as it:
        for e in it:
            if not e.is_dir(follow_symlinks=False):
                continue
            total += 1
            if limit and total > limit:
                break
            for s in streams:
                if os.path.exists(os.path.join(e.path, s)):
                    present[s] += 1
                elif len(missing_examples[s]) < 5:
                    missing_examples[s].append(e.name)
            if total % 10000 == 0:
                print(f'  [{label}] scanned {total}...', flush=True)
    res = {'total_dirs': total,
           'present': {s: present[s] for s in streams},
           'missing': {s: total - present[s] for s in streams},
           'missing_pct': {s: round(100.0 * (total - present[s]) / total, 2) for s in streams},
           'missing_examples': missing_examples}
    return res

allres = {}
for root, streams, label, lim in [(CORP, CORP_STREAMS, 'hegre_corpus', None),
                                  (FFHQ, FFHQ_STREAMS, 'ffhq_stratum', None)]:
    print(f'\n=== {label} ===', flush=True)
    r = scan(root, streams, label, lim)
    allres[label] = r
    print(f'  total dirs: {r["total_dirs"]}')
    for s in streams:
        print(f'    {s:26s} present {r["present"][s]:6d}  missing {r["missing"][s]:6d} '
              f'({r["missing_pct"][s]:5.2f}%)  e.g. {r["missing_examples"][s][:3]}', flush=True)

OUT.write_text(json.dumps(allres, indent=2))
print(f'\nwrote {OUT}')