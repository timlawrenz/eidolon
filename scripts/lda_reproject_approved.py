"""
Run just the AuraFace-LDA projection step for approved images.
Bypasses the early return in run_stratum_enrichment (insightface check)
to reach the LDA projection code that already exists in enrichment.py.
"""
import sys
import os
import time
import numpy as np
from pathlib import Path

# Ensure project root is on sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

dataset_dir = Path("/mnt/nas-ai-models/training-data/eidolon/hegre-faces/v1")

# Add geometry_pca to path for auraface_preprocessing
geom_pca = Path("/home/tim/source/activity/eidolon/experiments/geometry_pca")
sys.path.insert(0, str(geom_pca))

from geometry_pca.auraface_preprocessing import clean_auraface, project_to_lda

# Query approved images
from tools.hegre_dataset.dataset import HegreDataset
os.environ.setdefault('EIDOLON_SKIP_REVIEWDB_GUARD', '1')
ds = HegreDataset(dataset_dir)

rows = ds.db.execute(
    "SELECT image_path FROM images WHERE status = 'approved' ORDER BY persona_id, image_path"
).fetchall()

auraface_out = dataset_dir / "auraface"
lda_dir = dataset_dir / "lda"
lda_dir.mkdir(parents=True, exist_ok=True)

# Find images that have AuraFace but missing LDA
missing_lda = []
missing_af = 0
for row in rows:
    rel_p = Path(row["image_path"])
    lda_file = lda_dir / rel_p.with_suffix(".npy")
    if not lda_file.exists():
        af_file = auraface_out / rel_p.with_suffix(".npy")
        if af_file.exists():
            missing_lda.append((rel_p, af_file, lda_file))
        else:
            missing_af += 1

print(f"Approved images: {len(rows)}")
print(f"Missing AuraFace-LDA: {len(missing_lda)}")
print(f"Missing AuraFace raw (no .npy): {missing_af}")

if not missing_lda:
    print("All approved images already have AuraFace-LDA data.")
    sys.exit(0)

t0_lda = time.time()
n_lda = 0
n_lda_skip = 0
batch_size = 256
batch_af = []
batch_lda_paths = []

for i, (rel_p, af_file, lda_file) in enumerate(missing_lda):
    try:
        v_raw = np.load(af_file).astype(np.float64)
        if v_raw.shape != (512,):
            n_lda_skip += 1
            continue
        batch_af.append(v_raw)
        batch_lda_paths.append(lda_file)
    except Exception:
        n_lda_skip += 1
        continue

    if len(batch_af) >= batch_size or i == len(missing_lda) - 1:
        if batch_af:
            stacked = np.stack(batch_af)
            cleaned = clean_auraface(stacked)
            lda_coords = project_to_lda(cleaned)
            # Ensure 2D even for single-element batches (clean_auraface may squeeze)
            lda_coords = np.atleast_2d(lda_coords)
            for j, lda_path in enumerate(batch_lda_paths):
                lda_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(lda_path, lda_coords[j].astype(np.float32))
            n_lda += len(batch_af)
        batch_af = []
        batch_lda_paths = []

    if (i + 1) % 500 == 0:
        elapsed = time.time() - t0_lda
        rate = (i + 1) / elapsed
        eta = (len(missing_lda) - i - 1) / rate if rate > 0 else 0
        print(f"  [{i+1}/{len(missing_lda)}] {rate:.1f} img/s, ETA: {eta/60:.0f}m", flush=True)

elapsed = time.time() - t0_lda
print(f"AuraFace-LDA projection complete in {elapsed:.0f}s. Projected {n_lda}, skipped {n_lda_skip}.")
print(f"Rate: {n_lda/elapsed:.1f} img/s")
