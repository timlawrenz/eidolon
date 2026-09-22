import numpy as np
import sys
import os
from pathlib import Path

# Add project root to sys.path so we can import geometry_pca
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../experiments/geometry_pca")))

from geometry_pca.zg_inference import encode_zg
from geometry_pca.fit import load_encoder
from ..dataset import HegreDataset, Photo
try:
    from tools.hegre_dataset.review.procrustes import generate_pixel_average
except ImportError:
    generate_pixel_average = None


def compute_af_distances(db_path: Path, dataset_root: Path, persona: str | None = None) -> int:
    """Compute AuraFace cosine distances from the approved-image centroid for each persona.

    Loads AuraFace .npy files via HegreDataset.Photo, computes the centroid of
    approved images using cosine similarity, then assigns af_distance for every
    image that has an AuraFace embedding.
    """
    import time

    ds = HegreDataset(dataset_root)
    avg_dir = dataset_root / "averages"

    if persona is not None:
        if str(persona).isdigit():
            personas = ds.db.execute("SELECT id, name FROM personas WHERE id = ?", (int(persona),)).fetchall()
        else:
            personas = ds.db.execute("SELECT id, name FROM personas WHERE name = ?", (persona,)).fetchall()
        if not personas:
            print(f"Persona '{persona}' not found.")
            return 1
    else:
        personas = ds.db.execute("SELECT id, name FROM personas").fetchall()

    db = ds.db_writable

    total_updated = 0
    for p in personas:
        pname = p["name"]
        pid = p["id"]

        # Get all images with AuraFace embeddings for this persona
        approved_images = ds.db.execute(
            "SELECT image_path FROM images WHERE persona_id = ? AND status = 'approved'",
            (pid,)
        ).fetchall()
        all_images = ds.db.execute(
            "SELECT image_path FROM images WHERE persona_id = ?",
            (pid,)
        ).fetchall()

        if not approved_images:
            print(f"[{pname}] Skipped (no approved images)")
            continue

        # Build centroid from approved images
        approved_vectors = []
        for img in approved_images:
            photo = Photo(persona_name=pname, image_path=img["image_path"], dataset_root=dataset_root)
            if photo.has_auraface:
                try:
                    approved_vectors.append(photo.auraface)
                except Exception:
                    pass

        if not approved_vectors:
            print(f"[{pname}] Skipped (no valid AuraFace .npy files found)")
            continue

        centroid = np.mean(np.stack(approved_vectors), axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-8)

        updated = 0
        for img in all_images:
            photo = Photo(persona_name=pname, image_path=img["image_path"], dataset_root=dataset_root)
            if photo.has_auraface:
                try:
                    vec = photo.auraface
                    vec = vec / (np.linalg.norm(vec) + 1e-8)
                    dist = 1.0 - float(np.dot(vec, centroid))
                    db.execute(
                        "UPDATE images SET af_distance = ? WHERE persona_id = ? AND image_path = ?",
                        (dist, pid, img["image_path"])
                    )
                    updated += 1
                except Exception:
                    pass

        total_updated += updated
        print(f"[{pname}] Centroid from {len(approved_vectors)} approved; "
              f"computed af_distance for {updated} images")

    db.commit()
    print(f"\nDone. Updated af_distance for {total_updated} total images.")
    return 0


def compute_zg_distances(db_path: Path, stratum_dir: Path, encoder_path: str, persona: str | None = None, metric: str = "both", zg_max_distance: float = 100.0):
    ds = HegreDataset(stratum_dir.parent)
    db = ds.db_writable

    try:
        encoder = load_encoder(encoder_path)
    except FileNotFoundError:
        print(f"Encoder not found: {encoder_path}", file=sys.stderr)
        return 1

    if persona is not None:
        if str(persona).isdigit():
            personas = [ds.db.execute("SELECT id, name FROM personas WHERE id = ?", (int(persona),)).fetchone()]
        else:
            personas = [ds.db.execute("SELECT id, name FROM personas WHERE name = ?", (persona,)).fetchone()]
        personas = [p for p in personas if p is not None]
        if not personas:
            print(f"Persona '{persona}' not found.")
            return 1
    else:
        personas = ds.db.execute("SELECT id, name FROM personas").fetchall()

    for p in personas:
        pname = p["name"]
        pid = p["id"]

        # Get ALL images for this persona
        images = ds.db.execute(
            "SELECT id, image_path, status FROM images WHERE persona_id = ?",
            (pid,)
        ).fetchall()

        if not images:
            print(f"Skipped {pname} (no images)")
            continue

        base_pname = pname.split("_cluster_")[0]
        persona_dir = stratum_dir / base_pname

        vectors = []
        img_ids = []
        img_statuses = []  # Track status for each image ID
        approved_vectors = []
        bad_geo_ids = []
        image_paths = []   # for pixel average anchors
        face_2ds = []       # for pixel average anchors
        total_updated = 0  # Initialize per persona

        for img in images:
            # We know the specific subdirectory structure Stratum uses!
            # The image_path is "faces/{persona}/{set}/{filename}"
            # where {set} is the shoot directory.
            rel = Path(img["image_path"])
            shoot_name = rel.parent.name
            img_name = rel.stem

            # All extracted face crops live in the stratum directory under:
            #   <stratum>/<persona>/<shoot>/<image_name>/
            img_dir = persona_dir / shoot_name / img_name

            body_pose_path = img_dir / "pose.npy"
            face_pose_path = img_dir / "face_pose.npy"

            pose_path = face_pose_path if face_pose_path.exists() else body_pose_path
            if not pose_path.exists():
                continue

            try:
                pose_data = np.load(pose_path)
                face_2d = pose_data["face_2d"] if isinstance(pose_data, np.lib.npyio.NpzFile) else pose_data

                # Extract face_2d (first 68 points, x/y only, ensure float32 for linalg compatibility)
                if face_2d.shape[0] >= 68:
                    face_2d = face_2d[:68, :2].astype(np.float32)
                else:
                    continue  # Not enough keypoints

                zg = encode_zg(face_2d, encoder)
                vectors.append(zg)
                img_ids.append(img["id"])
                img_statuses.append(img["status"])

                if img["status"] == "approved":
                    approved_vectors.append(zg)
                    image_paths.append(ds.root / img["image_path"])
                    face_2ds.append(face_2d)

                total_images_processed = len(img_ids)
            except Exception as e:
                print(f"  Warning: failed to encode {img['image_path']}: {e}")

        if not vectors:
            print(f"Skipped {pname} (No valid pose.npy files found)")
            continue

        if not approved_vectors:
            print(f"Skipped {pname} (poses found but no approved images for centroid)")
            continue

        centroid = np.mean(np.stack(approved_vectors), axis=0)

        # Compute distances for ALL images from the approved centroid
        vectors = np.array(vectors)
        distances = np.linalg.norm(vectors - centroid, axis=1)

        # Pixel Average (Procrustes Warping) — only from tight approved images
        if generate_pixel_average is not None and len(approved_vectors) > 0:
            approved_dists = np.linalg.norm(np.array(approved_vectors) - centroid, axis=1)
            anchor_mask = approved_dists < 20.0
            anchor_paths = [image_paths[i] for i in range(len(image_paths)) if anchor_mask[i]]
            anchor_marks = [face_2ds[i] for i in range(len(face_2ds)) if anchor_mask[i]]

            n_filtered = len(image_paths) - len(anchor_paths)
            if n_filtered > 0:
                print(f"  -> Excluded {n_filtered} approved images (zg >= 20) from pixel average")

            if len(anchor_paths) > 0:
                pixel_path = stratum_dir / base_pname / f"pixel_{pname}.jpg"
                try:
                    from geometry_pca.zg_inference import decode_zg

                    face_2d_centroid = decode_zg(centroid, encoder)

                    # Rotate so eyes are horizontal
                    left_eye = np.mean(face_2d_centroid[36:42], axis=0)
                    right_eye = np.mean(face_2d_centroid[42:48], axis=0)
                    angle = np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])
                    cos_a, sin_a = np.cos(-angle), np.sin(-angle)
                    rot_mat = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
                    nose_tip = face_2d_centroid[30]
                    shifted = face_2d_centroid - nose_tip
                    rotated_face = shifted @ rot_mat.T
                    rotated_face += nose_tip

                    pixel_img = generate_pixel_average(anchor_paths, anchor_marks, rotated_face)
                    if pixel_img is not None:
                        import cv2
                        cv2.imwrite(str(pixel_path), pixel_img)
                except Exception as e:
                    print(f"  -> Error generating pixel average for {pname}: {e}")

        dist_updates = []
        nonface_ids = []

        for i, dist in enumerate(distances):
            dist_updates.append((float(dist), img_ids[i]))
            # Only auto-label NON-APPROVED images as non-face
            # (approved images with high zg_distance are kept for identity training)
            if dist > zg_max_distance and img_statuses[i] != "approved":
                nonface_ids.append((img_ids[i],))

        if dist_updates:
            db.executemany("UPDATE images SET zg_distance = ? WHERE id = ?", dist_updates)

            if nonface_ids:
                db.executemany("UPDATE images SET status = 'tainted:extraction_nonface', reviewed_at = NOW() WHERE id = ?", nonface_ids)
                print(f"  -> Auto-labeled {len(nonface_ids)} extreme outliers (dist > {zg_max_distance}) as 'Non-face'")

            db.commit()

        total_updated = len(dist_updates)
        print(f"[{total_images_processed} poses loaded] Centroid from {len(approved_vectors)} approved; "
              f"computed distances for {total_updated} images of {pname}")

        # Auto-label bad geometry so they don't skew the centroid
        if bad_geo_ids:
            db.executemany("UPDATE images SET status = 'tainted:approved_bad_geometry', reviewed_at = NOW() WHERE id = ?", bad_geo_ids)
            db.commit()
            print(f"  -> Auto-labeled {len(bad_geo_ids)} DWPose failures as 'Bad Geometry'")

    if len(personas) > 1:
        print(f"\nDone. Updated zg_distance for {total_updated} total images.")
    return 0


def compute_lda_vectors(db_path: Path, dataset_root: Path, persona: str | None = None, overwrite: bool = False) -> int:
    """Compute per-persona LDA identity averages.
    
    Loads AuraFace vectors, batch-projects to LDA, computes per-persona
    L2-normalized mean vectors, and saves to averages/. Does NOT save
    per-image LDA files (use `enrich` for that).
    
    Args:
        db_path: Path to review.db (unused; kept for CLI compat)
        dataset_root: Path to hegre-faces/v1 dataset
        persona: Optional persona name to limit computation
        overwrite: Force recompute even if average files exist
    
    Returns:
        0 on success
    """
    import sys
    from tools.hegre_dataset.dataset import HegreDataset
    
    _geom_pca = Path(__file__).resolve().parent.parent.parent.parent / "experiments" / "geometry_pca"
    if str(_geom_pca) not in sys.path:
        sys.path.insert(0, str(_geom_pca))
    
    os.environ.setdefault('EIDOLON_SKIP_REVIEWDB_GUARD', '1')
    ds = HegreDataset(dataset_root)
    
    try:
        from geometry_pca.auraface_preprocessing import clean_auraface, project_to_lda
    except ImportError as e:
        print(f"Error: Cannot import auraface_preprocessing: {e}")
        return 1
    
    # Query approved images grouped by persona
    if persona:
        persona_obj = ds.persona(persona)
        if persona_obj is None:
            print(f"Persona '{persona}' not found.")
            return 1
        rows = ds.db.execute(
            "SELECT i.persona_id, i.image_path FROM images i WHERE i.status = 'approved' AND i.persona_id = ?",
            (persona_obj.id,)
        ).fetchall()
    else:
        rows = ds.db.execute(
            "SELECT i.persona_id, i.image_path FROM images i WHERE i.status = 'approved' ORDER BY i.persona_id"
        ).fetchall()
    
    from collections import defaultdict
    persona_images = defaultdict(list)
    for pix, img_path in rows:
        af_path = dataset_root / "auraface" / img_path.replace(".jpg", ".npy")
        persona_images[pix].append((img_path, af_path))
    
    total_personas = len(persona_images)
    total_images = sum(len(v) for v in persona_images.values())
    print(f"Found {total_images} images across {total_personas} personas")
    
    averages_dir = dataset_root / "averages"
    averages_dir.mkdir(parents=True, exist_ok=True)
    
    n_avg_computed = 0
    n_avg_skipped = 0
    
    for pix, images in persona_images.items():
        persona_obj = ds.persona(pix)
        persona_name = persona_obj.name if persona_obj else f"persona_{pix}"
        avg_path = averages_dir / f"{persona_name}.lda.npy"
        
        if not overwrite and avg_path.exists():
            n_avg_skipped += 1
            continue
        
        # Batch-load all AuraFace vectors for this persona
        raw_vecs = []
        for img_path, af_path in images:
            try:
                v = np.load(af_path).astype(np.float64)
                if v.shape == (512,):
                    raw_vecs.append(v)
            except Exception:
                continue
        
        if not raw_vecs:
            continue
        
        # Batch clean + project
        raw_stack = np.stack(raw_vecs)
        cleaned = clean_auraface(raw_stack)  # (N, 512)
        lda_coords_all = project_to_lda(cleaned)  # (N, 64)
        
        # Average + L2 normalize
        avg = np.mean(lda_coords_all, axis=0)
        avg = avg / (np.linalg.norm(avg) + 1e-12)
        np.save(avg_path, avg.astype(np.float32))
        n_avg_computed += 1
        
        if n_avg_computed % 50 == 0:
            print(f"  [{n_avg_computed}/{total_personas - n_avg_skipped}] {persona_name}: {len(raw_vecs)} images, avg norm={np.linalg.norm(avg):.4f}")
    
    print(f"\nDone. Computed {n_avg_computed} averages, skipped {n_avg_skipped}.")
    return 0
