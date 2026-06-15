import numpy as np
import sqlite3
import sys
import os
from pathlib import Path

# Add project root to sys.path so we can import geometry_pca
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../experiments/geometry_pca")))

from geometry_pca.zg_inference import encode_zg
from geometry_pca.fit import load_encoder

def compute_zg_distances(db_path: Path, stratum_dir: Path, encoder_path: str):
    db = sqlite3.connect(str(db_path))
    db.row_factory = sqlite3.Row
    
    # 1. Add zg_distance column if it doesn't exist
    try:
        db.execute("ALTER TABLE images ADD COLUMN zg_distance REAL")
        db.commit()
        print("Added zg_distance column to images table.")
    except sqlite3.OperationalError:
        pass # Column already exists
        
    try:
        encoder = load_encoder(encoder_path)
    except FileNotFoundError:
        print(f"Error: Encoder not found at {encoder_path}")
        return 1

    personas = db.execute("SELECT id, name FROM personas").fetchall()
    total_updated = 0
    
    for p in personas:
        pid, pname = p["id"], p["name"]
        images = db.execute("SELECT id, image_path FROM images WHERE persona_id = ?", (pid,)).fetchall()
        
        vectors = []
        img_ids = []
        
        for img in images:
            # image_path is relative to 'faces', e.g. 'anna-l/anna-l-shoot/img_face1.jpg'
            # stratum_dir has 'anna-l/anna-l-shoot/img_face1/pose.npy'
            rel_path = Path(img["image_path"])
            pose_path = stratum_dir / rel_path.parent / rel_path.stem / "pose.npy"
            
            if pose_path.exists():
                try:
                    pose = np.load(pose_path).astype(np.float32)
                    face_2d = pose[23:91, :2]
                    zg = encode_zg(face_2d, encoder)
                    vectors.append(zg)
                    img_ids.append(img["id"])
                except Exception:
                    pass
        
        if vectors:
            vectors = np.array(vectors)
            centroid = np.mean(vectors, axis=0)
            distances = np.linalg.norm(vectors - centroid, axis=1)
            
            for iid, dist in zip(img_ids, distances):
                db.execute("UPDATE images SET zg_distance = ? WHERE id = ?", (float(dist), iid))
            db.commit()
            total_updated += len(vectors)
            print(f"Computed distances for {len(vectors)} images of {pname}")
            
    db.close()
    print(f"\nDone. Updated zg_distance for {total_updated} total images.")
    return 0
