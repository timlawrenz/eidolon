import numpy as np
import sqlite3
import sys
import os
from pathlib import Path
from PIL import Image, ImageDraw

# Add project root to sys.path so we can import geometry_pca
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../experiments/geometry_pca")))

from geometry_pca.zg_inference import encode_zg
from geometry_pca.fit import load_encoder
try:
    from tools.hegre_dataset.review.procrustes import generate_pixel_average
    from tools.hegre_dataset.review.flame_projector import extract_canonical_shape, generate_textured_mesh, render_spin_gif
except ImportError:
    generate_pixel_average = None
    extract_canonical_shape = None
    generate_textured_mesh = None
    render_spin_gif = None

def decode_zg(zg, encoder):
    comps = encoder["components"]
    pmean = encoder["pca_mean"]
    wmu = encoder["whiten_mu"]
    wsig = encoder["whiten_sigma"]
    
    raw = (zg * wsig) + wmu
    aligned_flat = (raw @ comps) + pmean
    return aligned_flat.reshape(68, 2)
def normalize_face_geometry(face_2d):
    """
    Returns a normalized face geometry where:
    - the eyes are horizontal (unrotated)
    """
    left_eye = np.mean(face_2d[36:42], axis=0)
    right_eye = np.mean(face_2d[42:48], axis=0)
    angle = np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])
    cos_a, sin_a = np.cos(-angle), np.sin(-angle)
    rot_mat = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    
    nose_tip = face_2d[30]
    shifted = face_2d - nose_tip
    rotated_face = shifted @ rot_mat.T
    rotated_face += nose_tip
    return rotated_face

def compute_zg_distances(db_path: Path, stratum_dir: Path, encoder_path: str, persona: str | None = None, skip_3d: bool = False):
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

    if persona is not None:
        # Check if the user passed an integer ID or a string name
        if str(persona).isdigit():
            personas = db.execute("SELECT id, name FROM personas WHERE id = ?", (int(persona),)).fetchall()
        else:
            personas = db.execute("SELECT id, name FROM personas WHERE name = ?", (persona,)).fetchall()
    else:
        personas = db.execute("SELECT id, name FROM personas").fetchall()
    total_updated = 0
    total_images_processed = 0
    
    print(f"Starting geometry compute for {len(personas)} personas...")
    
    for p in personas:
        pid, pname = p["id"], p["name"]
        
        # When DBSCAN splits personas, the new directory names in `stratum/` don't change!
        # If the persona is 'anna_cluster_1', the files are still inside `stratum/anna/`.
        # We need to map the persona name back to the original directory name by stripping '_cluster_X'.
        base_pname = pname.split("_cluster_")[0]
        
        # Only compute geometry for images that are either unreviewed or approved.
        # Images tainted as non-face or unusable shouldn't skew the true centroid.
        images = db.execute("SELECT id, image_path, status FROM images WHERE persona_id = ? AND status IN ('unreviewed', 'approved')", (pid,)).fetchall()
        
        vectors = []
        img_ids = []
        bad_geo_ids = []
        image_paths = []
        face_2ds = []
        dataset_root = stratum_dir.parent
        
        for img in images:
            # We know the specific subdirectory structure Stratum uses!
            # Instead of a slow rglob across the whole tree, we can just glob the persona dir directly.
            # Stratum outputs: stratum_dir / persona_name / ... / img_stem / pose.npy
            pose_path = None
            persona_dir = stratum_dir / base_pname
            if persona_dir.exists():
                for pth in persona_dir.rglob(f"{Path(img['image_path']).stem}/pose.npy"):
                    pose_path = pth
                    break
            
            if pose_path and pose_path.exists():
                try:
                    pose = np.load(pose_path).astype(np.float32)
                    face_2d = pose[23:91, :2]
                    
                    # 1. Catch DWPose False Negatives (All Zeros)
                    if np.all(face_2d == 0):
                        bad_geo_ids.append((img["id"],))
                        continue
                        
                    zg = encode_zg(face_2d, encoder)
                    vectors.append(zg)
                    img_ids.append(img["id"])
                    
                    if img["status"] == "approved":
                        image_paths.append(dataset_root / img["image_path"])
                        face_2ds.append(face_2d)
                        
                    total_images_processed += 1
                except Exception:
                    pass
        
        # Auto-label bad geometry so they don't skew the centroid
        if bad_geo_ids:
            db.executemany("UPDATE images SET status = 'tainted:approved_bad_geometry', reviewed_at = datetime('now') WHERE id = ?", bad_geo_ids)
            db.commit()
            print(f"  -> Auto-labeled {len(bad_geo_ids)} DWPose failures as 'Bad Geometry'")
        
        if vectors:
            vectors = np.array(vectors)
            centroid = np.mean(vectors, axis=0)
            distances = np.linalg.norm(vectors - centroid, axis=1)
            
            # Draw the 'Ghost' Average (Inverse PCA)
            face_2d = decode_zg(centroid, encoder)
            img = Image.new("RGB", (300, 300), (24, 24, 27)) # Zinc-950
            draw = ImageDraw.Draw(img)
            
            # 1. Rotate the face so eyes are horizontal
            rotated_face = normalize_face_geometry(face_2d)
            nose_tip = rotated_face[30]
            
            min_c = rotated_face.min(axis=0)
            max_c = rotated_face.max(axis=0)
            size_c = max_c - min_c
            scale = 220.0 / max(size_c) # Fit within 220px
            
            # Center the ghost on the nose tip
            for x, y in rotated_face:
                px = 150 + (x - nose_tip[0]) * scale
                py = 150 + (y - nose_tip[1]) * scale
                draw.ellipse([px-3, py-3, px+3, py+3], fill=(52, 211, 153)) # Emerald 400
            
            ghost_path = stratum_dir / base_pname / f"ghost_{pname}.png"
            if stratum_dir.exists() and (stratum_dir / base_pname).exists():
                img.save(ghost_path)
            
            # Draw the 'Pixel' Average (Procrustes Warping)
            if generate_pixel_average is not None and len(image_paths) > 0:
                pixel_path = stratum_dir / base_pname / f"pixel_{pname}.jpg"
                try:
                    # Limit to 50 random approved images to keep it fast
                    if len(image_paths) > 50:
                        idx = np.random.choice(len(image_paths), 50, replace=False)
                        sel_paths = [image_paths[i] for i in idx]
                        sel_marks = [face_2ds[i] for i in idx]
                    else:
                        sel_paths = image_paths
                        sel_marks = face_2ds
                        
                    # Pass the rotated_face to Procrustes instead of the tilted face_2d
                    pixel_img = generate_pixel_average(sel_paths, sel_marks, rotated_face) 
                    if pixel_img is not None:
                        import cv2
                        # cv2.imread loads in BGR. procrustes averages in BGR.
                        # cv2.imwrite expects BGR. 
                        # We were doing cv2.cvtColor(BGR2BGR) basically, which flipped the channels!
                        cv2.imwrite(str(pixel_path), pixel_img)
                except Exception as e:
                    print(f"Error generating pixel average for {pname}: {e}")
                    
            # 3. Spin Rendering (FLAME + Pixel Average)
            if not skip_3d and extract_canonical_shape is not None and len(image_paths) > 0:
                print(f"  -> Generating 3D FLAME Mesh for {pname}...")
                try:
                    # Phase 1: Extract mean skull geometry from DB via SMIRK
                    avg_shape = extract_canonical_shape(db_path, dataset_root, pname)
                    
                    # Phase 2: Project Pixel Average onto 3D mesh
                    pixel_path = stratum_dir / base_pname / f"pixel_{pname}.jpg"
                    if pixel_path.exists() and generate_textured_mesh is not None and render_spin_gif is not None:
                        mesh = generate_textured_mesh(avg_shape, pixel_path)
                        
                        # Phase 3: Render rotation GIF
                        gif_path = stratum_dir / base_pname / f"3d_{pname}.gif"
                        render_spin_gif(mesh, gif_path)
                        print(f"  -> Saved 3D asset: 3d_{pname}.gif")
                except Exception as e:
                    print(f"  -> Skipping 3D generation for {pname} due to error: {e}")
            
            dist_updates = []
            nonface_ids = []
            
            for iid, dist in zip(img_ids, distances):
                float_dist = float(dist)
                
                # 2. Catch extreme geometric outliers (Not a face)
                if float_dist > 100.0:
                    nonface_ids.append((iid,))
                else:
                    dist_updates.append((float_dist, iid))
                    
            db.executemany("UPDATE images SET zg_distance = ? WHERE id = ?", dist_updates)
            
            if nonface_ids:
                db.executemany("UPDATE images SET status = 'tainted:extraction_nonface', reviewed_at = datetime('now') WHERE id = ?", nonface_ids)
                print(f"  -> Auto-labeled {len(nonface_ids)} extreme outliers (dist > 100) as 'Non-face'")
                
            db.commit()
            total_updated += len(vectors)
            print(f"[{total_images_processed} poses loaded] Computed distances for {len(vectors)} images of {pname}")
        else:
            print(f"Skipped {pname} (No valid pose.npy files found)")
            
    db.close()
    print(f"\nDone. Updated zg_distance for {total_updated} total images.")
    return 0
