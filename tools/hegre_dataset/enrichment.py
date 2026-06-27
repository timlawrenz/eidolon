import subprocess
import shutil
import time
from pathlib import Path
import cv2
import numpy as np
from .review.schema import get_db

import sqlite3

def generate_image_list(db_path: Path, faces_dir: Path) -> list:
    """Query DB for approved and unreviewed images and return their absolute paths."""
    db = sqlite3.connect(f"file:{db_path}?mode=ro&nolock=1", uri=True)
    db.row_factory = sqlite3.Row
    rows = db.execute("SELECT image_path FROM images WHERE status IN ('approved', 'unreviewed')").fetchall()
    db.close()
    
    paths = []
    for row in rows:
        img_path = (faces_dir / row["image_path"]).absolute()
        paths.append(img_path)
    return paths

def run_stratum_enrichment(dataset_dir: Path, db_path: Path, faces_dir: Path, passes: str = "pose,seg,depth,normal,caption,t5", skip_stratum: bool = False):
    """Invoke stratum process only for images that miss Stratum data, and extract AuraFace for images that miss it."""
    stratum_out = dataset_dir / "stratum"
    auraface_out = dataset_dir / "auraface"
    list_file = dataset_dir / "stratum_approved_list.txt"
    
    paths = generate_image_list(db_path, faces_dir)
    if not paths:
        print("No approved or unreviewed images found. Skipping enrichment.")
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

    # 2. Check for missing AuraFace data
    missing_auraface = []
    for p in paths:
        rel_p = p.relative_to(faces_dir.absolute())
        # Store deterministically, keeping folder names and filenames intact
        auraface_file = auraface_out.absolute() / rel_p.with_suffix(".npy")
        if not auraface_file.exists():
            missing_auraface.append((p, auraface_file))

    if missing_auraface:
        print(f"Found {len(missing_auraface)} approved images missing AuraFace data. Extracting...")
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
        print("All approved images already have AuraFace data.")
