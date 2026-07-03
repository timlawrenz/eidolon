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

def compute_af_distances(db_path: Path, dataset_root: Path, persona: str | None = None) -> int:
    """Compute AuraFace cosine distances from the approved-image centroid for each persona.

    Loads deterministic AuraFace .npy files from auraface/faces/<persona>/<set>/<img>.npy,
    computes the centroid of all valid (unreviewed + approved) embeddings, and stores
    the cosine distance (1 - cosine_similarity) as af_distance in the images table.

    Args:
        db_path: Path to review.db.
        dataset_root: Dataset root containing auraface/ subdirectory.
        persona: Optional persona name or ID to limit computation.

    Returns:
        0 on success, 1 on error.
    """
    auraface_dir = dataset_root / "auraface"
    if not auraface_dir.exists():
        print(f"AuraFace directory not found: {auraface_dir}")
        print("Run 'enrich' first to extract AuraFace embeddings.")
        return 1

    db = sqlite3.connect(f"file:{db_path.resolve()}?nolock=1", uri=True)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")

    # Ensure af_distance column exists
    try:
        db.execute("ALTER TABLE images ADD COLUMN af_distance REAL")
        db.commit()
    except sqlite3.OperationalError:
        pass

    if persona is not None:
        if str(persona).isdigit():
            personas = db.execute("SELECT id, name FROM personas WHERE id = ?", (int(persona),)).fetchall()
        else:
            personas = db.execute("SELECT id, name FROM personas WHERE name = ?", (persona,)).fetchall()
    else:
        personas = db.execute("SELECT id, name FROM personas").fetchall()

    total_updated = 0

    print(f"Starting AuraFace distance compute for {len(personas)} personas...")

    for p in personas:
        pid, pname = p["id"], p["name"]

        # Load approved images for centroid; all images for distance computation
        approved_images = db.execute(
            "SELECT id, image_path FROM images "
            "WHERE persona_id = ? AND status = 'approved'",
            (pid,)
        ).fetchall()

        all_images = db.execute(
            "SELECT id, image_path FROM images "
            "WHERE persona_id = ? AND status IN ('unreviewed', 'approved')",
            (pid,)
        ).fetchall()

        # Approved embeddings → centroid
        approved_embeddings = []
        for img in approved_images:
            rel_path = Path(img["image_path"])
            af_path = auraface_dir / rel_path.with_suffix(".npy")
            if af_path.exists():
                try:
                    emb = np.load(af_path)
                    if emb.ndim == 1:
                        approved_embeddings.append(emb)
                except Exception:
                    continue

        if not approved_embeddings:
            print(f"[{pname}] Skipped (no approved AuraFace .npy files for centroid)")
            continue

        approved_embeddings = np.array(approved_embeddings)
        centroid = np.mean(approved_embeddings, axis=0)

        # All images → distances from approved centroid
        all_embeddings = []
        all_img_ids = []
        for img in all_images:
            rel_path = Path(img["image_path"])
            af_path = auraface_dir / rel_path.with_suffix(".npy")
            if af_path.exists():
                try:
                    emb = np.load(af_path)
                    if emb.ndim == 1:
                        all_embeddings.append(emb)
                        all_img_ids.append(img["id"])
                except Exception:
                    continue

        if all_embeddings:
            all_embeddings = np.array(all_embeddings)
            # Cosine similarity via dot product (embeddings are L2-normalized)
            similarities = all_embeddings @ centroid  # (N,)
            distances = 1.0 - similarities  # cosine distance in [0, 2]

            updates = [(float(d), iid) for d, iid in zip(distances, all_img_ids)]
            db.executemany("UPDATE images SET af_distance = ? WHERE id = ?", updates)
            db.commit()
            total_updated += len(updates)

            mean_dist = float(np.mean(distances))
            n_approved = len(approved_embeddings)
            print(f"[{pname}] Centroid from {n_approved} approved images; "
                  f"Computed af_distance for {len(updates)} images "
                  f"(mean cosine dist: {mean_dist:.4f})")
        else:
            print(f"[{pname}] Skipped (no valid AuraFace .npy files found)")

    db.close()
    print(f"\nDone. Updated af_distance for {total_updated} total images.")
    return 0


def compute_zg_distances(db_path: Path, stratum_dir: Path, encoder_path: str, persona: str | None = None, skip_3d: bool = False, metric: str = "both", zg_max_distance: float = 100.0):
    db = sqlite3.connect(f"file:{db_path.resolve()}?nolock=1", uri=True)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    
    # 1. Add zg_distance column if it doesn't exist
    try:
        db.execute("ALTER TABLE images ADD COLUMN zg_distance REAL")
        db.commit()
        print("Added zg_distance column to images table.")
    except sqlite3.OperationalError:
        pass # Column already exists

    # 1b. Add af_distance column if it doesn't exist
    try:
        db.execute("ALTER TABLE images ADD COLUMN af_distance REAL")
        db.commit()
        print("Added af_distance column to images table.")
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

    skip_zg = metric == "af"

    for p in personas:
        pid, pname = p["id"], p["name"]

        # When DBSCAN splits personas, the new directory names in `stratum/` don't change!
        # If the persona is 'anna_cluster_1', the files are still inside `stratum/anna/`.
        # We need to map the persona name back to the original directory name by stripping '_cluster_X'.
        base_pname = pname.split("_cluster_")[0]

        # Only compute geometry for images that are either unreviewed or approved.
        # Images tainted as non-face or unusable shouldn't skew the true centroid.
        images = db.execute("SELECT id, image_path, status FROM images WHERE persona_id = ? AND status IN ('unreviewed', 'approved')", (pid,)).fetchall()

        if skip_zg:
            total_images_processed += len(images)
            continue
        
        vectors = []
        img_ids = []
        approved_vectors = []  # Only approved images for centroid
        bad_geo_ids = []
        image_paths = []
        face_2ds = []
        dataset_root = stratum_dir.parent
        
        for img in images:
            # We know the specific subdirectory structure Stratum uses!
            # Instead of a slow rglob across the whole tree, we can just glob the persona dir directly.
            # Stratum outputs: stratum_dir / faces/ / persona_name / ... / img_stem / pose.npy
            # (or stratum_dir / persona_name / ... in older Phase 5a tree)
            # Using rglob to find regardless of prefix variation.
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
                        approved_vectors.append(zg)
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
        
        if vectors and approved_vectors:
            approved_vectors = np.array(approved_vectors)
            centroid = np.mean(approved_vectors, axis=0)
            
            # Compute distances for ALL images from the approved centroid
            vectors = np.array(vectors)
            distances = np.linalg.norm(vectors - centroid, axis=1)

            # Filter approved images for visual anchor generation: only use
            # those with zg_distance < 20 to keep ghost/pixel/volume tight.
            approved_dists = np.linalg.norm(approved_vectors - centroid, axis=1)
            anchor_mask = approved_dists < 20.0
            anchor_paths = [image_paths[i] for i in range(len(image_paths)) if anchor_mask[i]]
            anchor_marks = [face_2ds[i] for i in range(len(face_2ds)) if anchor_mask[i]]

            if len(image_paths) > 0:
                n_filtered = len(image_paths) - len(anchor_paths)
                if n_filtered > 0:
                    print(f"  -> Excluded {n_filtered} approved images (zg >= 20) from visual anchors")
            
            # Draw the 'Ghost' Average (Inverse PCA) from approved centroid
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
            if generate_pixel_average is not None and len(anchor_paths) > 0:
                pixel_path = stratum_dir / base_pname / f"pixel_{pname}.jpg"
                try:
                    # Only use approved images with zg_distance < 20 for the pixel average.
                    sel_paths = anchor_paths
                    sel_marks = anchor_marks
                        
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
            if not skip_3d and extract_canonical_shape is not None and len(anchor_paths) > 0:
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
                if float_dist > zg_max_distance:
                    nonface_ids.append((iid,))
                else:
                    dist_updates.append((float_dist, iid))
                    
            db.executemany("UPDATE images SET zg_distance = ? WHERE id = ?", dist_updates)
            
            if nonface_ids:
                db.executemany("UPDATE images SET status = 'tainted:extraction_nonface', reviewed_at = datetime('now') WHERE id = ?", nonface_ids)
                print(f"  -> Auto-labeled {len(nonface_ids)} extreme outliers (dist > {zg_max_distance}) as 'Non-face'")
                
            db.commit()
            total_updated += len(vectors)
            print(f"[{total_images_processed} poses loaded] Centroid from {len(approved_vectors)} approved; "
                  f"computed distances for {len(vectors)} images of {pname}")
        elif vectors:
            print(f"Skipped {pname} (poses found but no approved images for centroid)")
        else:
            print(f"Skipped {pname} (No valid pose.npy files found)")
            
    db.close()
    print(f"\nDone. Updated zg_distance for {total_updated} total images.")
    return 0


def compute_lda_vectors(db_path: Path, dataset_root: Path, persona: str | None = None) -> int:
    """Compute AuraFace-LDA vectors and per-persona identity averages.

    For each approved image from a contamination-free persona:
      1. Load raw AuraFace .npy
      2. clean_auraface() → remove PC1 + yaw nuisances
      3. project_to_lda() → 64-d identity basis
      4. Save as lda/{image_path}.npy

    After per-image extraction, computes per-persona averages
    of the LDA vectors → averages/{persona}.lda.npy.

    Idempotent: skips images where the output file already exists.

    Args:
        db_path: Path to review.db.
        dataset_root: Dataset root containing auraface/ and output targets.
        persona: Optional persona name to limit computation.

    Returns:
        0 on success, 1 on error.
    """
    import time

    auraface_dir = dataset_root / "auraface"
    if not auraface_dir.exists():
        print(f"AuraFace directory not found: {auraface_dir}")
        print("Run 'enrich' first to extract AuraFace embeddings.")
        return 1

    db = sqlite3.connect(f"file:{db_path.resolve()}?nolock=1", uri=True)
    db.row_factory = sqlite3.Row

    # ── Import AuraFace preprocessing (geometry_pca dependency) ──────
    try:
        import sys, os
        _geom_pca = Path(__file__).resolve().parent.parent.parent.parent / "experiments" / "geometry_pca"
        sys.path.insert(0, str(_geom_pca))
        from geometry_pca.auraface_preprocessing import clean_auraface, project_to_lda
    except ImportError as e:
        print(f"Error: Cannot import geometry_pca.auraface_preprocessing: {e}")
        print("Ensure geometry_pca is at experiments/geometry_pca/")
        return 1

    # ── Load persona list ────────────────────────────────────────────
    if persona is not None:
        if str(persona).isdigit():
            personas = db.execute("SELECT id, name FROM personas WHERE id = ?", (int(persona),)).fetchall()
        else:
            personas = db.execute("SELECT id, name FROM personas WHERE name = ?", (persona,)).fetchall()
    else:
        personas = db.execute("SELECT id, name FROM personas").fetchall()

    total_extracted = 0
    total_skipped = 0
    avg_new = 0
    avg_skipped = 0

    print(f"Starting LDA extraction for {len(personas)} personas...")
    t0 = time.time()

    for p in personas:
        pid, pname = p["id"], p["name"]

        # Skip contaminated personas
        contaminated = db.execute(
            "SELECT 1 FROM images WHERE persona_id = ? AND status = 'tainted:contamination' LIMIT 1",
            (pid,)
        ).fetchone()
        if contaminated:
            continue

        approved = db.execute(
            "SELECT image_path FROM images WHERE persona_id = ? AND status = 'approved'",
            (pid,)
        ).fetchall()

        if not approved:
            continue

        persona_extracted = 0
        af_missing = 0

        for img in approved:
            img_path = img["image_path"]
            lda_out = dataset_root / "lda" / img_path.replace('.jpg', '.npy')

            if lda_out.exists():
                total_skipped += 1
                persona_extracted += 1  # count as present for averaging
                continue

            af_in = auraface_dir / img_path.replace('.jpg', '.npy')
            if not af_in.exists():
                af_missing += 1
                continue

            try:
                v_raw = np.load(af_in)
                v_clean = clean_auraface(v_raw)
                lda_coords = project_to_lda(v_clean)

                lda_out.parent.mkdir(parents=True, exist_ok=True)
                np.save(lda_out, lda_coords)
                total_extracted += 1
                persona_extracted += 1
            except Exception as e:
                print(f"\n  Error on {pname}/{img_path}: {e}")

        # ── Per-persona average ──────────────────────────────────────
        avg_path = dataset_root / "averages" / f"{pname}.lda.npy"
        if avg_path.exists():
            avg_skipped += 1
        elif persona_extracted > 0:
            lda_vectors = []
            for img in approved:
                lda_p = dataset_root / "lda" / img["image_path"].replace('.jpg', '.npy')
                if lda_p.exists():
                    lda_vectors.append(np.load(lda_p))
            if lda_vectors:
                avg_vec = np.mean(np.stack(lda_vectors), axis=0)
                avg_vec = avg_vec / (np.linalg.norm(avg_vec) + 1e-8)
                avg_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(avg_path, avg_vec)
                avg_new += 1

        if persona_extracted > 0 or af_missing > 0:
            elapsed = time.time() - t0
            rate = total_extracted / elapsed if elapsed > 0 else 0
            print(f"  [{pname}] {persona_extracted} LDA vectors  "
                  f"(missing AF: {af_missing})  "
                  f"[{total_extracted} total, {rate:.1f}/s]")

    db.close()
    elapsed = time.time() - t0
    print(f"\nLDA extraction complete in {elapsed/60:.1f}m")
    print(f"  Per-image extracted: {total_extracted}")
    print(f"  Per-image skipped:   {total_skipped}")
    print(f"  New averages:        {avg_new}")
    print(f"  Averages skipped:    {avg_skipped}")
    return 0
