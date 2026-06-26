#!/usr/bin/env python3
import os
import sys
import sqlite3
import numpy as np
import cv2
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from geometry_pca.constants import FACE_SLICE
from geometry_pca.zg_inference import encode_zg

from insightface.app import FaceAnalysis

def main():
    db_uri = "file:data/review.db?mode=ro"
    stratum_root = "/mnt/nas-ai-models/training-data/eidolon/geometry_pca_data/hegre_faces_stratum"
    crops_dir = "/mnt/nas-ai-models/training-data/eidolon/geometry_pca_data/hegre_face_crops"
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
    app_aura = FaceAnalysis(name='auraface', root='/mnt/nas-ai-models', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app_aura.prepare(ctx_id=0, det_size=(512, 512))
    
    print("Querying valid images from review.db...")
    db = sqlite3.connect(db_uri, uri=True)
    c = db.cursor()
    c.execute('''
        SELECT i.enriched_dir, p.name, i.persona_id, i.image_path
        FROM images i JOIN personas p ON i.persona_id = p.id
        WHERE i.status = 'approved'
          AND i.persona_id NOT IN (
              SELECT persona_id FROM images WHERE status = 'tainted:contamination'
          )
    ''')
    rows = c.fetchall()
    db.close()
    
    X_t5 = []
    Y_target = []
    
    print(f"Processing {len(rows)} images...")
    
    for row in tqdm(rows):
        ed_orig, p_name, pid, img_path = row
        # Resolve stratum dir
        rel_path = ed_orig.split("hegre_enriched/", 1)[1]
        stratum_dir = os.path.join(stratum_root, rel_path)
        
        req_t5 = os.path.join(stratum_dir, "t5_hidden.npy")
        req_mask = os.path.join(stratum_dir, "t5_mask.npy")
        req_pose = os.path.join(stratum_dir, "pose.npy")
        
        if not (os.path.exists(req_t5) and os.path.exists(req_mask) and os.path.exists(req_pose)):
            continue
            
        # Resolve face crop
        # crops are like: hegre_face_crops/anna-l/anna-l-hegre-model-01.jpg
        # Let's derive it from image_path
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        # remove -14000px if present
        base_name = base_name.replace('-14000px', '')
        crop_path = os.path.join(crops_dir, p_name, f"{base_name}.jpg")
        
        if not os.path.exists(crop_path):
            continue
            
        # 1. AuraFace
        img = cv2.imread(crop_path)
        if img is None:
            continue
            
        faces_aura = app_aura.get(img)
        if len(faces_aura) == 0:
            continue
        # get highest confidence face or largest
        aura_emb = faces_aura[0].normed_embedding # 512-d
        
        # 2. T5
        t5 = np.load(req_t5)
        mask = np.load(req_mask)
        seq_len = mask.sum()
        if seq_len == 0:
            continue
        t5_emb = t5[:seq_len].mean(axis=0).astype(np.float32) # 1024-d
        
        # 3. Z_g
        pose = np.load(req_pose)
        face_2d = pose[FACE_SLICE, :2] # (68, 2)
        # Z_g encode
        z_g = encode_zg(face_2d, prod_encoder) # 50-d
        
        # Concat Target
        target = np.concatenate([aura_emb, z_g], axis=0) # 562-d
        
        X_t5.append(t5_emb)
        Y_target.append(target)
        
    X_t5 = np.array(X_t5)
    Y_target = np.array(Y_target)
    
    print(f"Dataset shape: X={X_t5.shape}, Y={Y_target.shape}")
    np.savez_compressed(os.path.join(output_dir, "dataset.npz"), X=X_t5, Y=Y_target)
    print(f"Saved to {output_dir}/dataset.npz")

if __name__ == '__main__':
    main()
