#!/usr/bin/env python3
import os
import sys
import sqlite3
import numpy as np
import cv2
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'geometry_pca'))
from geometry_pca.verification import verification_auc
from geometry_pca.constants import FACE_SLICE
from geometry_pca.depth_encoder import face_bbox_px

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "smirk"))
from src.smirk_encoder import SmirkEncoder

def get_flame_targets(db_uri, stratum_root, img_root):
    db = sqlite3.connect(db_uri, uri=True)
    c = db.cursor()
    c.execute('''
        SELECT i.image_path, p.name, i.persona_id
        FROM images i JOIN personas p ON i.persona_id = p.id
        WHERE i.status = 'approved'
          AND i.persona_id NOT IN (
              SELECT persona_id FROM images WHERE status = 'tainted:contamination'
          )
    ''')
    rows = c.fetchall()
    db.close()
    
    targets = []
    for img_path, name, pid in rows:
        stem = os.path.splitext(img_path[len("faces/"):] if img_path.startswith("faces/") else img_path)[0]
        stratum_dir = os.path.join(stratum_root, stem)
        pose_path = os.path.join(stratum_dir, "pose.npy")
        full_img = os.path.join(img_root, img_path)
        
        if os.path.exists(pose_path) and os.path.exists(full_img):
            targets.append((full_img, pose_path, pid))
            
    return targets

def crop_for_smirk(img, pose, target_size=224):
    """SMIRK expects a tight face crop resized to 224x224."""
    h, w = img.shape[:2]
    face_2d = pose[FACE_SLICE, :2]
    
    # Use existing depth_encoder logic to get a robust bounding box
    x0, y0, x1, y1 = face_bbox_px(face_2d, h, w, pad=0.2)
    crop = img[y0:y1, x0:x1]
    
    if crop.size == 0:
        return None
        
    crop = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_AREA)
    
    # SMIRK transforms: RGB, [0, 1], normalize
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    crop = crop.astype(np.float32) / 255.0
    
    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    crop = (crop - mean) / std
    
    # (C, H, W)
    crop = np.transpose(crop, (2, 0, 1))
    return torch.from_numpy(crop).unsqueeze(0)

def main():
    print("=" * 60)
    print("  Tier 1.1: FLAME Beta Spike (SMIRK Encoder)")
    print("=" * 60)
    
    db_uri = "file:/mnt/nas-ai-models/training-data/eidolon/hegre-faces/v1/review.db?mode=ro&nolock=1"
    stratum_root = "/mnt/nas-ai-models/training-data/eidolon/hegre-faces/v1/stratum"
    img_root = "/mnt/nas-ai-models/training-data/eidolon/hegre-faces/v1"
    
    targets = get_flame_targets(db_uri, stratum_root, img_root)
    print(f"Found {len(targets)} approved images with pose data.")
    
    print("Loading SMIRK Shape Encoder...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmirkEncoder(n_shape=300).to(device)
    checkpoint = torch.load("experiments/flame_spike/smirk/pretrained_models/SMIRK_em1.pt", map_location=device)
    
    encoder_state_dict = {}
    for k, v in checkpoint.items():
        if k.startswith("smirk_encoder."):
            encoder_state_dict[k.replace("smirk_encoder.", "")] = v
            
    model.load_state_dict(encoder_state_dict, strict=False)
    model.eval()
    
    X_beta = []
    y_labels = []
    
    n_skip = 0
    import time
    t0 = time.time()
    
    with torch.no_grad():
        batch_tensors = []
        batch_pids = []
        
        def flush_batch():
            if not batch_tensors: return
            b_crop = torch.cat(batch_tensors, dim=0).to(device)
            outputs = model(b_crop)
            betas = outputs['shape_params'].cpu().numpy()
            X_beta.extend(betas)
            y_labels.extend(batch_pids)
            batch_tensors.clear()
            batch_pids.clear()

        for idx, (img_path, pose_path, pid) in enumerate(targets):
            if idx % 500 == 0 and idx > 0:
                print(f"  Processed {idx}/{len(targets)}... ({(time.time()-t0)/idx:.4f}s/img)")
                
            img = cv2.imread(img_path)
            if img is None:
                n_skip += 1; continue
                
            pose = np.load(pose_path).astype(np.float32)
            
            # Note: in Phase 2, we filtered faces with low pose confidence
            if pose[23:91, 2].mean() < 0.5:
                n_skip += 1; continue
                
            tensor_crop = crop_for_smirk(img, pose)
            if tensor_crop is None:
                n_skip += 1; continue
                
            batch_tensors.append(tensor_crop)
            batch_pids.append(pid)
            
            if len(batch_tensors) >= 64:
                flush_batch()
                
        flush_batch()
            
    print(f"Extraction complete. Skipped {n_skip}.")
    
    # Save the beta vectors for fast re-eval
    os.makedirs("experiments/flame_spike/output", exist_ok=True)
    np.savez("experiments/flame_spike/output/flame_beta_gate.npz", X=np.array(X_beta), y=np.array(y_labels))
    
    X = np.array(X_beta)
    y = np.array(y_labels)
    
    SEEDS = [0, 1, 2]
    auc_beta = [verification_auc(X, y, seed=s)[0] for s in SEEDS]
    
    print("-" * 60)
    print("Verification AUC (N=40k balanced pairs per seed)")
    print("-" * 60)
    print(f"FLAME Beta (300-d): {np.mean(auc_beta):.4f}  {[f'{a:.4f}' for a in auc_beta]}")
    
    # Note: To fully satisfy the Tier 1.1 condition, we need AUC([z_g | beta]).
    # We would need to load the production z_g encoder and extract X_g for these exact images.
    # However, AUC(beta) > 0.60 is the first required gating threshold.
    if np.mean(auc_beta) > 0.60:
        print("\n=> PASS (Condition 1): FLAME Beta exceeds the noise floor (>0.60).")
        print("   Next step is to test AUC([z_g | beta]) to verify complementarity.")
    else:
        print("\n=> FAIL: FLAME Beta alone does not strongly encode identity on this dataset.")

if __name__ == "__main__":
    main()
