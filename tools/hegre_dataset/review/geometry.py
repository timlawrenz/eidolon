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
        approved_vectors = []
        bad_geo_ids = []
        image_paths = []   # for pixel average anchors
        face_2ds = []       # for pixel average anchors

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

                # Extract shoulder pose if available
                shoulder_pose = None
                if isinstance(pose_data, np.lib.npyio.NpzFile) and "shoulder_pose" in pose_data:
                    shoulder_pose = pose_data["shoulder_pose"]

                zg = encode_zg(face_2d, shoulder_pose, encoder)
                vectors.append(zg)
                img_ids.append(img["id"])

                if img["status"] == "approved":
                    approved_vectors.append(zg)
                    image_paths.append(ds.root / img["image_path"])
                    face_2ds.append(face_2d)

                total_images_processed = len(img_ids)
            except Exception:
                pass

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
            if dist > zg_max_distance:
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
