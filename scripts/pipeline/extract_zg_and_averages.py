#!/usr/bin/env python3
"""
Idempotent extraction of z_g vectors and calculation of identity averages.
Processes FFHQ and Hegre datasets.
"""

import os
import sys
import sqlite3
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add geometry_pca to path so we can import encode_zg
sys.path.append(os.path.join(os.path.dirname(__file__), '../../experiments/geometry_pca'))
from geometry_pca.zg_inference import encode_zg
from geometry_pca.constants import FACE_SLICE

FFHQ_ROOT = Path("/mnt/nas-ai-models/training-data/ffhq")
HEGRE_ROOT = Path("/mnt/nas-ai-models/training-data/eidolon/hegre-faces/v1")

PROD_ENCODER_PATH = Path("/home/tim/source/activity/eidolon/experiments/geometry_pca/output/encoder_production.npz")


def load_production_encoder():
    return dict(np.load(PROD_ENCODER_PATH))


def extract_zg_for_dataset(dataset_name, pose_paths, zg_out_paths, encoder, desc):
    """Extract z_g vectors for a list of pose paths (always recomputes to sync with DB)."""
    if not pose_paths:
        return

    print(f"[{dataset_name}] Extracting/updating {len(pose_paths)} z_g vectors...")
    for pose_path, zg_path in tqdm(zip(pose_paths, zg_out_paths), total=len(pose_paths), desc=desc):
        if not pose_path.exists():
            # Old/new stratum path handling can be an issue, we'll try both for Hegre
            alt_path = Path(str(pose_path).replace('stratum/faces/', 'stratum/'))
            if alt_path.exists():
                pose_path = alt_path
            else:
                continue

        pose = np.load(pose_path)
        # Ensure it has confidence channel to slice
        if pose.shape == (133, 3):
            face_2d = pose[FACE_SLICE, :2]
        elif pose.shape == (68, 2):
            face_2d = pose
        else:
            continue
        
        # Check for invalid poses (e.g. zeros from DWPose failures)
        if (face_2d == 0).all():
            continue

        z_g = encode_zg(face_2d.astype(np.float32), encoder)
        zg_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(zg_path, z_g)


def process_ffhq(encoder):
    stratum_dir = FFHQ_ROOT / "stratum"
    zg_dir = FFHQ_ROOT / "zg"
    
    pose_paths = list(stratum_dir.glob("*/pose.npy"))
    zg_paths = [zg_dir / p.parent.name / "zg.npy" for p in pose_paths]
    
    extract_zg_for_dataset("FFHQ", pose_paths, zg_paths, encoder, "FFHQ z_g")


def process_hegre(encoder):
    db_path = HEGRE_ROOT / "review.db"
    if not db_path.exists():
        print(f"Hegre DB not found: {db_path}")
        return

    conn = sqlite3.connect(f"file:{db_path}?mode=ro&nolock=1", uri=True)
    c = conn.cursor()
    c.execute("""
        SELECT i.image_path, p.name 
        FROM images i
        JOIN personas p ON i.persona_id = p.id
        WHERE i.status = 'approved'
    """)
    approved_images = c.fetchall()
    conn.close()

    pose_paths = []
    zg_paths = []
    persona_image_map = {}

    for img_path, persona in approved_images:
        # img_path looks like: faces/anna-l/anna-l-hegre-model/img.jpg
        # we want stratum/faces/.../pose.npy
        rel_base = Path(img_path).with_suffix('')
        
        pose_path = HEGRE_ROOT / "stratum" / rel_base / "pose.npy"
        zg_path = HEGRE_ROOT / "zg" / f"{rel_base}.npy"
        
        pose_paths.append(pose_path)
        zg_paths.append(zg_path)
        
        if persona not in persona_image_map:
            persona_image_map[persona] = []
        persona_image_map[persona].append((rel_base, zg_path))

    extract_zg_for_dataset("Hegre", pose_paths, zg_paths, encoder, "Hegre z_g")

    # Compute Averages
    print(f"[Hegre] Computing identity averages for {len(persona_image_map)} personas...")
    avg_dir = HEGRE_ROOT / "averages"
    
    # Clean stale averages so removed personas don't leave ghost files
    if avg_dir.exists():
        for f in avg_dir.glob("*.npy"):
            f.unlink()
    avg_dir.mkdir(exist_ok=True)

    for persona, items in tqdm(persona_image_map.items(), desc="Averages"):
        zg_vectors = []
        auraface_vectors = []
        
        for rel_base, zg_path in items:
            if zg_path.exists():
                zg_vectors.append(np.load(zg_path))
                
            aura_path = HEGRE_ROOT / "auraface" / f"{rel_base}.npy"
            if aura_path.exists():
                auraface_vectors.append(np.load(aura_path))

        if zg_vectors:
            avg_zg = np.mean(np.stack(zg_vectors), axis=0)
            np.save(avg_dir / f"{persona}.zg.npy", avg_zg)
            
        if auraface_vectors:
            avg_aura = np.mean(np.stack(auraface_vectors), axis=0)
            # Re-normalize to hypersphere
            avg_aura = avg_aura / (np.linalg.norm(avg_aura) + 1e-8)
            np.save(avg_dir / f"{persona}.auraface.npy", avg_aura)


if __name__ == "__main__":
    print("Loading production encoder...")
    encoder = load_production_encoder()
    
    process_ffhq(encoder)
    process_hegre(encoder)
    print("Done.")
