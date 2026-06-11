#!/usr/bin/env python3
"""
Phase 3 Correction: Face-cropped DINOv3 extractor.

The original DINOv3 stratum pass ran on full 14000px editorial scenes. DINO
did 'scene matching' rather than facial recognition, and the FFHQ-fit bridge (W)
failed because it was trained on face-crops but fed scene embeddings.
This script extracts face crops from the approved hegre images (matching the z_a
bbox logic) and passes them through DINOv3.

Output: dinov3_cls_face.npy inside each enriched_dir.
"""
import os, sys, time, sqlite3
import numpy as np
from PIL import Image
import torch

# Mute PIL DecompressionBombWarning for 14000px images
Image.MAX_IMAGE_PIXELS = None

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from geometry_pca.constants import FACE_SLICE
from geometry_pca.depth_encoder import face_bbox_px

# Insert stratum-hq to path so we can use its DINOv3 loader
STRATUM_PATH = "/home/tim/source/activity/stratum-hq"
if STRATUM_PATH not in sys.path:
    sys.path.insert(0, STRATUM_PATH)

try:
    from stratum.pipeline.dinov3 import load_dinov3, compute_dinov3_both
except ImportError as e:
    print(f"FATAL: Could not import stratum pipeline. {e}")
    sys.exit(1)


def get_hegre_images():
    """READ-ONLY query of review.db to get image paths and enriched dirs."""
    db = sqlite3.connect("file:data/review.db?mode=ro", uri=True)
    c = db.cursor()
    c.execute("""
        SELECT i.image_path, i.enriched_dir
        FROM images i JOIN personas p ON i.persona_id = p.id
        WHERE i.status = 'approved'
          AND i.persona_id NOT IN (
              SELECT persona_id FROM images WHERE status = 'tainted:contamination'
          )
        ORDER BY p.name, i.id
    """)
    rows = c.fetchall()
    db.close()
    return rows


def main():
    print("=" * 60)
    print("  Extracting Face-Cropped DINOv3 Features (Phase 3 Correction)")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading DINOv3 on {device}...")
    dino_model = load_dinov3(device)
    print("Done.\n")

    images = get_hegre_images()
    print(f"Found {len(images)} approved gate images in review.db.")

    n_ok = n_skip = 0
    t0 = time.time()

    for idx, (img_path, ed_rel) in enumerate(images):
        # Resolve paths
        # enriched_dir is relative (data/hegre_enriched/...)
        ed_abs = os.path.join(
            "/mnt/nas-ai-models/training-data/eidolon/hegre_enriched",
            ed_rel.split("hegre_enriched/", 1)[1]
        )
        out_path = os.path.join(ed_abs, "dinov3_cls_face.npy")

        # Idempotency
        if os.path.exists(out_path):
            n_ok += 1
            continue

        try:
            # 1. Load pose to get face bbox
            pose_path = os.path.join(ed_abs, "pose.npy")
            if not os.path.exists(pose_path):
                n_skip += 1; continue
            pose = np.load(pose_path).astype(np.float32)
            face = pose[FACE_SLICE]

            # 2. Load full image to get dimensions
            if not os.path.exists(img_path):
                n_skip += 1; continue
            with Image.open(img_path) as img:
                w, h = img.size
                
                # 3. Compute crop (same logic as z_a normal crop)
                x0, y0, x1, y1 = face_bbox_px(face, h, w)
                
                # 4. Crop image
                face_crop = img.crop((x0, y0, x1, y1))
                
                # 5. Run DINOv3
                cls_list, _ = compute_dinov3_both(face_crop, dino_model)
                cls_token = cls_list[0].astype(np.float16)  # Match original dtype

            # 6. Save
            np.save(out_path, cls_token)
            n_ok += 1
            
            if n_ok % 50 == 0:
                print(f"  Processed {n_ok} / {len(images)}  ({(time.time()-t0)/n_ok:.1f}s/img)")

        except Exception as e:
            print(f"Error on {img_path}: {e}")
            n_skip += 1
            continue

    print(f"\nDone. Processed {n_ok} OK, skipped/failed {n_skip}.")
    print(f"Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
