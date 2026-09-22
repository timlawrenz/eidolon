#!/usr/bin/env python3
"""Full per-image LDA reprojection — deletes old files, batch-reprojects all approved images.

Uses the new LDA basis (auraface_lda.npz + auraface_preprocess.npz, refitted 2026-07-20).
Batch-loads per persona, projects, saves per-image. No GPU needed — AuraFace files already exist.
"""
import os, sys, time, numpy as np
from pathlib import Path
from collections import defaultdict

# Setup
_PROJ = Path(__file__).resolve().parent.parent  # project root
sys.path.insert(0, str(_PROJ))
sys.path.insert(0, str(_PROJ / "experiments" / "geometry_pca"))

os.environ.setdefault('EIDOLON_SKIP_REVIEWDB_GUARD', '1')

from tools.hegre_dataset.dataset import HegreDataset
from geometry_pca.auraface_preprocessing import clean_auraface, project_to_lda

DATASET = Path("/mnt/nas-ai-models/training-data/eidolon/hegre-faces/v1")


def main():
    ds = HegreDataset(DATASET)
    t0 = time.time()

    # Query approved images grouped by persona
    rows = ds.db.execute("""
        SELECT persona_id, image_path
        FROM images WHERE status = 'approved'
        ORDER BY persona_id, image_path
    """).fetchall()

    persona_images = defaultdict(list)
    for pid, img_path in rows:
        af_path = DATASET / "auraface" / img_path.replace(".jpg", ".npy")
        lda_path = DATASET / "lda" / img_path.replace(".jpg", ".npy")
        persona_images[pid].append((img_path, af_path, lda_path))

    total_personas = len(persona_images)
    total_images = sum(len(v) for v in persona_images.values())
    print(f"{total_images} images across {total_personas} personas")

    n_deleted = 0
    n_projected = 0
    n_errors = 0
    n_skipped_af = 0

    for pid, images in persona_images.items():
        persona_obj = ds.persona(pid)
        pname = persona_obj.name if persona_obj else f"persona_{pid}"

        # Batch-load all AuraFace vectors for this persona
        raw_vecs = []
        valid_entries = []
        for img_path, af_path, lda_path in images:
            try:
                v = np.load(af_path).astype(np.float64)
                if v.shape == (512,):
                    raw_vecs.append(v)
                    valid_entries.append((img_path, lda_path))
            except Exception:
                n_skipped_af += 1
                continue

        if not raw_vecs:
            continue

        # Batch clean + project
        raw_stack = np.stack(raw_vecs)
        cleaned = clean_auraface(raw_stack)
        lda_all = project_to_lda(cleaned)  # (N, 64)

        # Save per-image
        for (img_path, lda_path), lda_vec in zip(valid_entries, lda_all):
            try:
                lda_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(lda_path, lda_vec.astype(np.float32))
                n_projected += 1
            except Exception as e:
                n_errors += 1
                if n_errors <= 5:
                    print(f"  Error {img_path}: {e}")

        if len(persona_images) and n_projected % 1000 == 0:
            elapsed = time.time() - t0
            rate = n_projected / elapsed
            eta = (total_images - n_projected) / rate if rate > 0 else 0
            print(f"  [{n_projected}/{total_images}] {rate:.1f} img/s, ETA: {eta/60:.0f}m")

    elapsed = time.time() - t0
    print(f"\nReprojection complete in {elapsed/60:.1f}m")
    print(f"  Projected: {n_projected}")
    print(f"  Skipped (missing AF): {n_skipped_af}")
    print(f"  Errors: {n_errors}")


if __name__ == "__main__":
    main()