import subprocess
import shutil
import time
from pathlib import Path
import cv2
import numpy as np
from .review.schema import get_db

import sqlite3

def generate_image_list(db_path: Path, faces_dir: Path, status_filter: str = "both",
                         zg_max_distance: float | None = None,
                         sort_by: str | None = None) -> list:
    """Query DB for images matching the status filter and return their absolute paths.

    Args:
        db_path: Path to review.db.
        faces_dir: Root directory for face images.
        status_filter: 'approved', 'unreviewed', or 'both' (default: 'both').
        zg_max_distance: If set, exclude approved images with zg_distance above this
            threshold. Images with NULL zg_distance (not yet computed) are always
            included. Only applies to approved images, not unreviewed.
        sort_by: 'af' to sort by af_distance ASC then zg_distance ASC;
            'zg' to sort by zg_distance ASC then af_distance ASC;
            None for no ordering (default).
    """
    db = sqlite3.connect(f"file:{db_path}?mode=ro&nolock=1", uri=True)
    db.row_factory = sqlite3.Row

    # Build WHERE clause
    if status_filter == "approved":
        where = "WHERE status = 'approved'"
    elif status_filter == "unreviewed":
        where = "WHERE status = 'unreviewed'"
    else:
        where = "WHERE status IN ('approved', 'unreviewed')"

    params: list = []

    # Add zg_distance filter for approved images
    if zg_max_distance is not None:
        # Check if zg_distance column exists
        col_exists = db.execute(
            "SELECT 1 FROM pragma_table_info('images') WHERE name = 'zg_distance'"
        ).fetchone() is not None
        if col_exists:
            if status_filter == "approved":
                where += " AND (zg_distance IS NULL OR zg_distance <= ?)"
                params.append(zg_max_distance)
            elif status_filter == "both":
                where += (" AND (status = 'unreviewed' OR zg_distance IS NULL "
                          "OR (status = 'approved' AND zg_distance <= ?))")
                params.append(zg_max_distance)
            # unreviewed: no filter

    # Build ORDER BY clause
    order = ""
    if sort_by == "af":
        order = "ORDER BY af_distance ASC NULLS LAST, zg_distance ASC NULLS LAST"
    elif sort_by == "zg":
        order = "ORDER BY zg_distance ASC NULLS LAST, af_distance ASC NULLS LAST"

    rows = db.execute(f"SELECT image_path FROM images {where} {order}", params).fetchall()
    db.close()

    paths = []
    for row in rows:
        img_path = (faces_dir / row["image_path"]).absolute()
        paths.append(img_path)
    return paths

def run_stratum_enrichment(dataset_dir: Path, db_path: Path, faces_dir: Path,
                            passes: str = "pose,seg,depth,normal,caption,t5",
                            skip_stratum: bool = False, status_filter: str = "both",
                            zg_max_distance: float | None = None,
                            sort_by: str | None = None):
    """Invoke stratum process only for images that miss Stratum data, and extract AuraFace for images that miss it.

    Args:
        status_filter: 'approved', 'unreviewed', or 'both' (default: 'both').
        zg_max_distance: Exclude approved images with zg_distance above this threshold.
        sort_by: 'af' or 'zg' to sort by distance ascending; None for no ordering.
    """
    stratum_out = dataset_dir / "stratum"
    auraface_out = dataset_dir / "auraface"
    list_file = dataset_dir / "stratum_approved_list.txt"

    paths = generate_image_list(db_path, faces_dir, status_filter=status_filter,
                                zg_max_distance=zg_max_distance, sort_by=sort_by)
    if not paths:
        filter_desc = f"zg<={zg_max_distance} " if zg_max_distance else ""
        print(f"No {filter_desc}{status_filter} images found. Skipping enrichment.")
        return

    # 1. Check for missing Stratum data (respecting the requested passes)
    if not skip_stratum:
        missing_stratum = []
        pass_list = passes.split(',')
        
        for p in paths:
            rel_p = p.relative_to(faces_dir.absolute())
            base_dir = stratum_out.absolute() / rel_p.parent / rel_p.stem
            
            is_missing = False
            if "pose" in pass_list and not (base_dir / "pose.npy").exists():
                is_missing = True
            elif "caption" in pass_list and not (base_dir / "caption.txt").exists():
                is_missing = True
            elif "t5" in pass_list and not (base_dir / "t5_hidden.npy").exists():
                is_missing = True
            elif "pixel" in pass_list and not (base_dir / "pixel.npy").exists():
                is_missing = True
                
            if is_missing:
                missing_stratum.append(str(p))

        if missing_stratum:
            print(f"Found {len(missing_stratum)} images missing Stratum data. Invoking stratum-hq with passes: {passes}...")
            if shutil.which("stratum") is None:
                print("Error: 'stratum' command not found in PATH.")
            else:
                with open(list_file, "w") as f:
                    for p in missing_stratum:
                        f.write(p + "\n")
                cmd = [
                    "stratum", "process", str(faces_dir.absolute()),
                    "--output", str(stratum_out.absolute()), "--passes", passes, "--device", "cpu", "--image-list", str(list_file.absolute())
                ]
                subprocess.run(cmd, check=True)
        else:
            print("All images already have Stratum data.")
    else:
        print("Skipping Stratum enrichment (--skip-stratum).")

    # 1b. Check for missing z_g data (after pose pass may have generated pose.npy)
    zg_dir = dataset_dir / "zg"
    missing_zg = []
    for p in paths:
        rel_p = p.relative_to(faces_dir.absolute())
        zg_file = zg_dir.absolute() / rel_p.with_suffix(".npy")
        if not zg_file.exists():
            # Only extract if pose.npy exists (stratum pose pass ran)
            pose_file = stratum_out.absolute() / rel_p.parent / rel_p.stem / "pose.npy"
            if pose_file.exists():
                missing_zg.append((p, zg_file))

    if missing_zg:
        print(f"Found {len(missing_zg)} {status_filter} images with pose but missing z_g. Extracting...")
        try:
            import sys as _sys, os as _os
            _geom_pca = Path(__file__).resolve().parent.parent.parent / "experiments" / "geometry_pca"
            _sys.path.insert(0, str(_geom_pca))
            from geometry_pca.zg_inference import encode_zg
            from geometry_pca.fit import load_encoder
            from geometry_pca.constants import FACE_SLICE
        except ImportError as e:
            print(f"Error: Cannot import geometry_pca for z_g extraction: {e}")
            print("Skipping z_g extraction.")
        else:
            # Resolve encoder path relative to project root (geometry.py's ancestor)
            _proj_root = Path(__file__).resolve().parent.parent.parent  # eidolon/
            encoder_path = _proj_root / "experiments" / "geometry_pca" / "output" / "encoder_production.npz"
            if not encoder_path.exists():
                print(f"Error: encoder_production.npz not found. Copy it to {encoder_path}")
                print("Skipping z_g extraction.")
            else:
                encoder = load_encoder(str(encoder_path))
                t0_zg = time.time()
                n_zg = 0
                n_zg_skip = 0
                for i, (in_p, out_p) in enumerate(missing_zg):
                    if i > 0 and i % 500 == 0:
                        elapsed = time.time() - t0_zg
                        rate = i / elapsed
                        eta = (len(missing_zg) - i) / rate if rate > 0 else 0
                        print(f"  [{i}/{len(missing_zg)}] {rate:.1f} img/s, ETA: {eta/60:.0f}m", flush=True)

                    rel_p = in_p.relative_to(faces_dir.absolute())
                    pose_path = stratum_out.absolute() / rel_p.parent / rel_p.stem / "pose.npy"
                    try:
                        pose = np.load(pose_path)
                        if pose.shape == (133, 3):
                            face_2d = pose[FACE_SLICE, :2]
                        elif pose.shape == (68, 2):
                            face_2d = pose
                        else:
                            n_zg_skip += 1
                            continue
                        if (face_2d == 0).all():
                            n_zg_skip += 1
                            continue
                        z_g = encode_zg(face_2d.astype(np.float32), encoder)
                        out_p.parent.mkdir(parents=True, exist_ok=True)
                        np.save(out_p, z_g)
                        n_zg += 1
                    except Exception:
                        n_zg_skip += 1

                elapsed = time.time() - t0_zg
                print(f"z_g extraction complete in {elapsed:.0f}s. Extracted {n_zg}, skipped {n_zg_skip}.")
    else:
        print(f"All {status_filter} images with pose already have z_g data.")

    # 2. Check for missing AuraFace data
    missing_auraface = []
    for p in paths:
        rel_p = p.relative_to(faces_dir.absolute())
        # Store deterministically, keeping folder names and filenames intact
        auraface_file = auraface_out.absolute() / rel_p.with_suffix(".npy")
        if not auraface_file.exists():
            missing_auraface.append((p, auraface_file))

    if missing_auraface:
        print(f"Found {len(missing_auraface)} {status_filter} images missing AuraFace data. Extracting...")
        try:
            import sys
            import os
            # Restart if LD_LIBRARY_PATH misses cuDNN to enable GPU AuraFace
            venv_base = '/home/tim/source/activity/eidolon/experiments/geometry_pca/.venv/lib/python3.14/site-packages'
            cudnn_path = f"{venv_base}/nvidia/cudnn/lib"
            cublas_path = f"{venv_base}/nvidia/cublas/lib"
            
            current_ld = os.environ.get('LD_LIBRARY_PATH', '')
            if cudnn_path not in current_ld and os.path.exists(cudnn_path):
                os.environ['LD_LIBRARY_PATH'] = f"{cudnn_path}:{cublas_path}:{current_ld}".strip(':')
                print("Restarting subprocess to load CUDA libraries...")
                # When called as 'python -m tools.hegre_dataset', sys.argv[0]
                # is __main__.py and relative imports break on direct re-exec.
                # Reconstruct the -m invocation so package imports resolve.
                argv0 = sys.argv[0]
                if argv0.endswith('__main__.py') and 'tools/hegre_dataset' in argv0:
                    new_argv = [sys.executable, '-m', 'tools.hegre_dataset'] + sys.argv[1:]
                else:
                    new_argv = [sys.executable] + sys.argv
                os.execv(sys.executable, new_argv)
                
            from insightface.app import FaceAnalysis
        except ImportError:

            print("Error: insightface not installed. Skipping AuraFace extraction.")
            return

        app = FaceAnalysis(name='auraface', root='/mnt/nas-ai-models', providers=['CPUExecutionProvider'])
        # det_size in InsightFace forces the input image to be resized to that resolution before passing 
        # to the SCRFD detector. The default det_size=(640,640) works, but setting it forces padding/scaling logic.
        # But wait, why are we using SCRFD at all? 
        # We ALREADY have a perfectly cropped face. SCRFD is failing because the face fills the frame, 
        # lacking shoulder/background context.
        # 
        # If we remove det_size entirely, InsightFace defaults to (640, 640). 
        # The true fix is padding the 512px MTCNN crop so SCRFD can 'see' the edges.
        app.prepare(ctx_id=0)

        t0 = time.time()
        n_skip = 0
        P = len(missing_auraface)
        for i, (in_p, out_p) in enumerate(missing_auraface):
            if i > 0 and i % 100 == 0:
                elapsed = time.time() - t0
                rate = i / elapsed
                eta = (P - i) / rate
                print(f"  [{i}/{P}] {rate:.1f} img/s, ETA: {eta:.0f}s", flush=True)

            out_p.parent.mkdir(parents=True, exist_ok=True)
            img = cv2.imread(str(in_p))
            if img is None:
                n_skip += 1
                continue
                
            faces = app.get(img)
            if len(faces) == 0:
                # SCRFD often fails to detect faces when the image is already a tightly cropped 512px MTCNN box.
                # We can trick it by padding the image with a black border, running detection, and taking the result.
                # Pad by 20% on all sides
                h, w = img.shape[:2]
                pad_y = int(h * 0.2)
                pad_x = int(w * 0.2)
                padded_img = cv2.copyMakeBorder(img, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=[0,0,0])
                faces = app.get(padded_img)
                
            if len(faces) == 0:
                # If it STILL fails after padding, we skip it
                n_skip += 1
                continue
                
            emb = faces[0].normed_embedding
            np.save(out_p, emb)

        elapsed = time.time() - t0
        print(f"AuraFace extraction complete in {elapsed:.0f}s. Skipped {n_skip} images.")
    else:
        print(f"All {status_filter} images already have AuraFace data.")

    # 2b. Check for missing AuraFace-LDA data (per-image: clean → project to LDA)
    lda_dir = dataset_dir / "lda"
    missing_lda = []
    for p in paths:
        rel_p = p.relative_to(faces_dir.absolute())
        lda_file = lda_dir.absolute() / rel_p.with_suffix(".npy")
        if not lda_file.exists():
            af_file = auraface_out.absolute() / rel_p.with_suffix(".npy")
            if af_file.exists():
                missing_lda.append((p, af_file, lda_file))

    if missing_lda:
        print(f"Found {len(missing_lda)} {status_filter} images missing AuraFace-LDA. Projecting...")
        try:
            import sys as _sys
            _geom_pca = Path(__file__).resolve().parent.parent.parent / "experiments" / "geometry_pca"
            _sys.path.insert(0, str(_geom_pca))
            from geometry_pca.auraface_preprocessing import clean_auraface, project_to_lda
        except ImportError as e:
            print(f"Error: Cannot import auraface_preprocessing: {e}")
            print("Skipping AuraFace-LDA projection.")
        else:
            t0_lda = time.time()
            n_lda = 0
            n_lda_skip = 0
            for i, (in_p, af_file, lda_file) in enumerate(missing_lda):
                if i > 0 and i % 500 == 0:
                    elapsed = time.time() - t0_lda
                    rate = i / elapsed
                    eta = (len(missing_lda) - i) / rate if rate > 0 else 0
                    print(f"  [{i}/{len(missing_lda)}] {rate:.1f} img/s, ETA: {eta/60:.0f}m", flush=True)

                try:
                    v_raw = np.load(af_file)
                    v_clean = clean_auraface(v_raw)
                    lda_coords = project_to_lda(v_clean)
                    lda_file.parent.mkdir(parents=True, exist_ok=True)
                    np.save(lda_file, lda_coords)
                    n_lda += 1
                except Exception:
                    n_lda_skip += 1

            elapsed = time.time() - t0_lda
            print(f"AuraFace-LDA projection complete in {elapsed:.0f}s. Extracted {n_lda}, skipped {n_lda_skip}.")
    else:
        print(f"All {status_filter} images already have AuraFace-LDA data.")
