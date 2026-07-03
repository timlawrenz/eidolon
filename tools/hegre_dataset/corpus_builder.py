"""Corpus builder: assemble hegre training samples into stratum-style directories.

Each output directory contains:
    pixel.npy        (3, 1024, 1024) float16  — face crop
    auraface_lda.npy (64,)           float64  — persona-averaged identity vector
    z_g.npy          (50,)           float32  — per-image geometry vector
    metadata.json    {"persona": ..., "set": ..., "image_id": ...}
"""

import sqlite3
import json
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image


def build_corpus(
    dataset_root: Path,
    output_dir: Path,
    min_images_per_persona: int = 5,
    max_images_per_persona: int | None = None,
    resolution: int = 1024,
    dry_run: bool = False,
) -> int:
    """Build stratum-style corpus from hegre dataset.

    Args:
        dataset_root: Path to hegre-faces/v1 dataset.
        output_dir: Where to create numbered sample directories.
        min_images_per_persona: Skip personas with fewer approved images.
        max_images_per_persona: Cap images per persona (None = no cap).
        resolution: Target pixel resolution (default 1024).
        dry_run: If True, only count samples without writing files.

    Returns:
        0 on success, 1 on error.
    """
    db_path = dataset_root / "review.db"
    if not db_path.exists():
        print(f"Error: review.db not found at {db_path}")
        return 1

    avg_dir = dataset_root / "averages"
    faces_dir = dataset_root  # image_path is relative to dataset root

    conn = sqlite3.connect(f"file:{db_path.resolve()}?nolock=1", uri=True)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    # ── Contamination-free personas ───────────────────────────────────
    c.execute("SELECT DISTINCT persona_id FROM images WHERE status='tainted:contamination'")
    contaminated_pids = {r[0] for r in c.fetchall()}

    c.execute("SELECT id, name FROM personas ORDER BY name")
    all_personas = c.fetchall()

    # ── Gather eligible images ────────────────────────────────────────
    eligible = []  # list of (persona_name, image_path) tuples
    skipped_missing_avg = 0
    skipped_missing_zg = 0
    skipped_too_few = 0
    skipped_contaminated = 0

    for p in all_personas:
        pid = p["id"]
        pname = p["name"]

        if pid in contaminated_pids:
            skipped_contaminated += 1
            continue

        # Check persona average LDA exists
        avg_lda_path = avg_dir / f"{pname}.lda.npy"
        if not avg_lda_path.exists():
            skipped_missing_avg += 1
            continue

        approved = c.execute(
            "SELECT image_path FROM images WHERE persona_id = ? AND status = 'approved'",
            (pid,)
        ).fetchall()

        if len(approved) < min_images_per_persona:
            skipped_too_few += 1
            continue

        # Filter: only images with z_g available
        persona_eligible = []
        for img in approved:
            img_path = img["image_path"]
            zg_path = dataset_root / "zg" / img_path.replace('.jpg', '.npy')
            if zg_path.exists():
                persona_eligible.append(img_path)
            else:
                skipped_missing_zg += 1

        if max_images_per_persona and len(persona_eligible) > max_images_per_persona:
            # Take a random subset (deterministic by sorting)
            persona_eligible = sorted(persona_eligible)[:max_images_per_persona]

        eligible.extend((pname, ip) for ip in persona_eligible)

    conn.close()

    print(f"Corpus build summary:")
    print(f"  Personas scanned:           {len(all_personas)}")
    print(f"  Contaminated (skipped):     {skipped_contaminated}")
    print(f"  Missing LDA average:        {skipped_missing_avg}")
    print(f"  Too few images (<{min_images_per_persona}):  {skipped_too_few}")
    print(f"  Missing z_g (skipped):      {skipped_missing_zg}")
    print(f"  Eligible images:            {len(eligible)}")
    print(f"  Unique personas:            {len(set(name for name, _ in eligible))}")

    if dry_run:
        print("\nDry run — no files written.")
        return 0

    # ── Build corpus ──────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    written = 0
    errors = 0

    for i, (pname, img_path) in enumerate(eligible):
        sample_dir = output_dir / f"{i:05d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Pixels: prefer stratum's pixel.npy if enrichment ran pixel pass
            img_path_p = Path(img_path)
            stratum_pixel = dataset_root / "stratum" / img_path_p.parent / img_path_p.stem / "pixel.npy"
            if stratum_pixel.exists():
                import shutil
                shutil.copy2(stratum_pixel, sample_dir / "pixel.npy")
            else:
                # Fallback: JPEG → float16 numpy
                jpg_path = faces_dir / img_path
                img = Image.open(jpg_path).convert("RGB")
                if img.size != (resolution, resolution):
                    img = img.resize((resolution, resolution), Image.LANCZOS)
                pixel = np.array(img, dtype=np.float16) / 255.0  # (H, W, 3)
                pixel = np.transpose(pixel, (2, 0, 1))  # (3, H, W)
                np.save(sample_dir / "pixel.npy", pixel)

            # AuraFace-LDA: persona average (same for all images of this persona)
            avg_lda = np.load(avg_dir / f"{pname}.lda.npy")
            np.save(sample_dir / "auraface_lda.npy", avg_lda)

            # z_g: per-image geometry
            zg_src = dataset_root / "zg" / img_path.replace('.jpg', '.npy')
            zg = np.load(zg_src)
            np.save(sample_dir / "z_g.npy", zg.astype(np.float32))

            # Metadata
            set_slug = Path(img_path).parent.name
            stem = Path(img_path).stem
            meta = {
                "persona": pname,
                "set": set_slug,
                "image_id": stem,
            }
            with open(sample_dir / "metadata.json", "w") as f:
                json.dump(meta, f)

            written += 1

        except Exception as e:
            print(f"\n  Error on {sample_dir.name} ({pname}/{img_path}): {e}")
            errors += 1

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(eligible) - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{len(eligible)}] {rate:.1f} samples/s, "
                  f"ETA: {eta/60:.0f}m")

    elapsed = time.time() - t0
    print(f"\nCorpus built in {elapsed/60:.1f}m")
    print(f"  Written:  {written}")
    print(f"  Errors:   {errors}")
    print(f"  Output:   {output_dir}")

    return 0
