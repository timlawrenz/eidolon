#!/usr/bin/env python3
import os
import sys

# Ensure LD_LIBRARY_PATH contains cuDNN and cuBLAS for ONNX Runtime GPU
def ensure_cuda_libs():
    venv_base = '/home/tim/source/activity/eidolon/experiments/geometry_pca/.venv/lib/python3.14/site-packages'
    cudnn_path = f"{venv_base}/nvidia/cudnn/lib"
    cublas_path = f"{venv_base}/nvidia/cublas/lib"
    
    current_ld = os.environ.get('LD_LIBRARY_PATH', '')
    if cudnn_path not in current_ld and os.path.exists(cudnn_path):
        os.environ['LD_LIBRARY_PATH'] = f"{cudnn_path}:{cublas_path}:{current_ld}".strip(':')
        os.execv(sys.executable, [sys.executable] + sys.argv)

ensure_cuda_libs()
import time
from pathlib import Path
import cv2
import numpy as np

try:
    from insightface.app import FaceAnalysis
except ImportError:
    print("Error: insightface not installed. Please install insightface.")
    sys.exit(1)

FFHQ_RAW = Path("/mnt/nas-ai-models/training-data/ffhq/raw")
FFHQ_AURA = Path("/mnt/nas-ai-models/training-data/ffhq/auraface")

def main():
    if not FFHQ_RAW.exists():
        print(f"Error: FFHQ raw directory not found at {FFHQ_RAW}")
        return

    FFHQ_AURA.mkdir(parents=True, exist_ok=True)
    
    print("Scanning FFHQ images...")
    images = list(FFHQ_RAW.glob("*.png"))
    N = len(images)
    print(f"Found {N} FFHQ images.")
    
    pending = []
    for p in images:
        out_p = FFHQ_AURA / p.with_suffix('.npy').name
        if not out_p.exists():
            pending.append((p, out_p))
            
    P = len(pending)
    print(f"{P} images need AuraFace extraction.")
    if P == 0:
        return
        
    print("Loading AuraFace model...")
    app = FaceAnalysis(name='auraface', root='/mnt/nas-ai-models', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(512, 512))
    
    t0 = time.time()
    n_skip = 0
    
    for i, (in_p, out_p) in enumerate(pending):
        if i > 0 and i % 1000 == 0:
            elapsed = time.time() - t0
            rate = i / elapsed
            eta = (P - i) / rate
            print(f"  [{i}/{P}] {rate:.1f} img/s, ETA: {eta:.0f}s")
            
        img = cv2.imread(str(in_p))
        if img is None:
            n_skip += 1
            continue
            
        faces = app.get(img)
        if len(faces) == 0:
            n_skip += 1
            continue
            
        emb = faces[0].normed_embedding
        np.save(out_p, emb)
        
    elapsed = time.time() - t0
    print(f"Done. Processed {P} images in {elapsed:.0f}s. Skipped {n_skip} images.")

if __name__ == "__main__":
    main()
