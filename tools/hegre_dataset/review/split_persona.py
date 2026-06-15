import argparse
import sys
import os
import sqlite3
import numpy as np
from pathlib import Path
from sklearn.cluster import DBSCAN

# Add project root to sys.path so we can import geometry_pca
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../experiments/geometry_pca")))

from geometry_pca.zg_inference import encode_zg
from geometry_pca.fit import load_encoder

def cmd_review_split_persona(args):
    dataset = Path(args.dataset)
    db_path = dataset / "review.db"
    stratum_dir = dataset / "stratum"
    
    if not stratum_dir.exists():
        print(f"Error: Stratum directory {stratum_dir} not found. Run enrichment first.")
        return 1

    db = sqlite3.connect(str(db_path))
    db.row_factory = sqlite3.Row
    
    # Load encoder
    try:
        encoder = load_encoder(args.encoder)
    except FileNotFoundError:
        print(f"Error: Encoder not found at {args.encoder}")
        return 1

    # Find the target persona
    persona_row = db.execute("SELECT id, name FROM personas WHERE name = ?", (args.persona,)).fetchone()
    if not persona_row:
        print(f"Error: Persona '{args.persona}' not found in the database.")
        return 1
        
    pid = persona_row["id"]
    pname = persona_row["name"]

    # Get all unreviewed images for this persona
    images = db.execute("SELECT id, image_path FROM images WHERE persona_id = ? AND status = 'unreviewed'", (pid,)).fetchall()
    if not images:
        print(f"No unreviewed images found for '{pname}'.")
        return 0

    print(f"Loading pose data for {len(images)} images of '{pname}'...")
    vectors = []
    img_ids = []
    
    for img in images:
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

    if len(vectors) < 10:
        print("Not enough pose data to perform clustering (need at least 10).")
        return 1

    vectors = np.array(vectors)
    print(f"Extracted {len(vectors)} zg vectors. Running DBSCAN clustering...")

    # Run DBSCAN (eps threshold determines how far apart vectors can be to be considered the same cluster)
    # eps=40 is a reasonable starting point given our spike showed standard deviations of ~42 for Anna's contaminated set
    clustering = DBSCAN(eps=args.eps, min_samples=5).fit(vectors)
    labels = clustering.labels_
    
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
    
    if n_clusters <= 1:
        print("No distinct sub-personas found. The current persona appears unified.")
        return 0
        
    print(f"Found {n_clusters} distinct clusters (and some noise). Splitting database...")
    
    for cluster_id in unique_labels:
        if cluster_id == -1:
            # Noise points (outliers). Leave them in the original persona to be reviewed/tainted manually.
            noise_count = sum(1 for l in labels if l == -1)
            print(f"  - Left {noise_count} noisy outliers in original '{pname}'")
            continue
            
        new_pname = f"{pname}_cluster_{cluster_id + 1}"
        db.execute("INSERT OR IGNORE INTO personas (name) VALUES (?)", (new_pname,))
        new_pid = db.execute("SELECT id FROM personas WHERE name = ?", (new_pname,)).fetchone()["id"]
        
        # Get image IDs for this cluster
        cluster_img_ids = [img_ids[i] for i in range(len(labels)) if labels[i] == cluster_id]
        
        # We also need to copy the 'sets' association over to the new persona so the images don't break foreign key constraints
        for iid in cluster_img_ids:
            # Find which set this image belonged to originally
            old_set_id = db.execute("SELECT set_id FROM images WHERE id = ?", (iid,)).fetchone()["set_id"]
            set_slug = db.execute("SELECT slug FROM sets WHERE id = ?", (old_set_id,)).fetchone()["slug"]
            
            # Ensure the set exists under the new persona
            db.execute("INSERT OR IGNORE INTO sets (persona_id, slug) VALUES (?, ?)", (new_pid, set_slug))
            new_set_id = db.execute("SELECT id FROM sets WHERE persona_id = ? AND slug = ?", (new_pid, set_slug)).fetchone()["id"]
            
            # Move the image to the new persona and set
            db.execute("UPDATE images SET persona_id = ?, set_id = ? WHERE id = ?", (new_pid, new_set_id, iid))
            
        print(f"  - Created '{new_pname}' with {len(cluster_img_ids)} images.")
        
    db.commit()
    db.close()
    print("Database split successfully. Run `review compute-geometry` again to recalculate the centroids for the new clusters.")
    return 0
