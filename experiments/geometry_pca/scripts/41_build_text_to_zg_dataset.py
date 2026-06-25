#!/usr/bin/env python3
import os
import sys
import sqlite3
import numpy as np
import cv2
from tqdm import tqdm
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from geometry_pca.constants import FACE_SLICE
from geometry_pca.zg_inference import encode_zg
from insightface.app import FaceAnalysis

def main():
    # Target dataset
    dataset_root = "/mnt/nas-ai-models/training-data/eidolon/hegre-faces/v1"
    db_uri = f"file:{os.path.join(dataset_root, 'review.db')}?mode=ro&nolock=1"
    output_dir = "data/text_to_zg"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load production encoder for Z_g
    encoder_path = "output/encoder_production.npz"
    if not os.path.exists(encoder_path):
        print(f"FATAL: Missing Z_g encoder at {encoder_path}")
        return
    prod_encoder = dict(np.load(encoder_path))
    
    # Initialize AuraFace
    print("Loading AuraFace...")
    app_aura = FaceAnalysis(name='auraface', root='.', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app_aura.prepare(ctx_id=0, det_size=(512, 512))
    
    print("Querying valid images from review.db...")
    db = sqlite3.connect(db_uri, uri=True)
    c = db.cursor()
    c.execute('''
        SELECT p.name, i.image_path
        FROM images i JOIN personas p ON i.persona_id = p.id
        WHERE i.status = 'approved'
          AND i.persona_id NOT IN (
              SELECT persona_id FROM images WHERE status = 'tainted:contamination'
          )
    ''')
    rows = c.fetchall()
    db.close()
    
    print("Pass 1: Computing average AuraFace per persona...")
    persona_aura = defaultdict(list)
    valid_images = []
    
    for p_name, img_path in tqdm(rows):
        # Resolve stratum dir (e.g., 'faces/anna-l/anna-l-hegre-model-01.jpg' -> 'stratum/anna-l/anna-l-hegre-model-01')
        stem = os.path.splitext(img_path[len("faces/"):] if img_path.startswith("faces/") else img_path)[0]
        stratum_dir = os.path.join(dataset_root, "stratum", stem)
        
        req_t5 = os.path.join(stratum_dir, "t5_hidden.npy")
        req_pose = os.path.join(stratum_dir, "pose.npy")
        
        if not (os.path.exists(req_t5) and os.path.exists(req_pose)):
            continue
            
        full_img = os.path.join(dataset_root, img_path)
        if not os.path.exists(full_img):
            continue
            
        img = cv2.imread(full_img)
        if img is None:
            continue
            
        faces_aura = app_aura.get(img)
        if len(faces_aura) == 0:
            continue
            
        aura_emb = faces_aura[0].normed_embedding # 512-d
        persona_aura[p_name].append(aura_emb)
        valid_images.append((p_name, full_img, req_t5, req_pose))
        
    print("Averaging vectors...")
    persona_aura_avg = {p: np.mean(embs, axis=0) for p, embs in persona_aura.items()}
    
    print("Pass 2: Building T5 -> [AuraAvg || z_g] dataset...")
    X_t5 = []
    Y_target = []
    
    for p_name, full_img, req_t5, req_pose in tqdm(valid_images):
        # 1. T5
        t5 = np.load(req_t5)
        # Using simple mean pool, or you can grab cls token if preferred
        t5_emb = t5.mean(axis=0).astype(np.float32) # 1024-d
        
        # 2. Z_g
        pose = np.load(req_pose)
        face_2d = pose[FACE_SLICE, :2] # (68, 2)
        z_g = encode_zg(face_2d, prod_encoder) # 50-d
        
        # 3. Aura Avg
        aura_avg = persona_aura_avg[p_name] # 512-d
        
        # Concat Target
        target = np.concatenate([aura_avg, z_g], axis=0) # 562-d
        
        X_t5.append(t5_emb)
        Y_target.append(target)
        
    X_t5 = np.array(X_t5)
    Y_target = np.array(Y_target)
    
    print(f"Dataset shape: X_t5={X_t5.shape}, Y_target={Y_target.shape}")
    np.savez_compressed(os.path.join(output_dir, "dataset_hegre.npz"), X=X_t5, Y=Y_target)
    print(f"Saved to {output_dir}/dataset_hegre.npz")

if __name__ == '__main__':
    main()
