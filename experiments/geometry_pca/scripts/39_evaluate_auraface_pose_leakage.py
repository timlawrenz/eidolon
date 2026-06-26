#!/usr/bin/env python3
import os
import sys
import sqlite3
import numpy as np
import cv2
import math
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from insightface.app import FaceAnalysis

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from geometry_pca.constants import FACE_SLICE
from geometry_pca.pose_normalize import estimate_rotation
from geometry_pca.canonical_face import canonical_template

def head_rotation(face_2d, canonical_tpl):
    tpl = canonical_tpl.copy()
    tpl[:, 1] = -tpl[:, 1]  # Flip Y to image coordinates
    return estimate_rotation(tpl, face_2d).astype(np.float32)

def yaw_pitch_from_R(R):
    yaw = math.asin(max(-1.0, min(1.0, -float(R[2, 0]))))
    pitch = math.atan2(float(R[2, 1]), float(R[2, 2]))
    return yaw, pitch

def get_intersection_targets():
    db_uri = "file:/mnt/nas-ai-models/training-data/eidolon/hegre-faces/v1/review.db?mode=ro&nolock=1"
    stratum_root = "/mnt/nas-ai-models/training-data/eidolon/hegre-faces/v1/stratum"
    img_root = "/mnt/nas-ai-models/training-data/eidolon/hegre-faces/v1"
    
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
            targets.append((full_img, pose_path))
    return targets

def main():
    print("=" * 60)
    print("  Tier 0.2: Pose Leakage Probe (AuraFace -> Yaw/Pitch)")
    print("=" * 60)
    
    targets = get_intersection_targets()
    print(f"Matched {len(targets)} approved images with pose data.")
    
    print("Loading AuraFace...")
    app_aura = FaceAnalysis(name='auraface', root='/mnt/nas-ai-models', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app_aura.prepare(ctx_id=0, det_size=(512, 512))
    
    CANONICAL_TPL = canonical_template()
    X_aura, Y_pose = [], []
    
    n_skip = 0
    import time
    t0 = time.time()
    
    for idx, (img_path, pose_path) in enumerate(targets):
        if idx % 500 == 0 and idx > 0:
            print(f"  Processed {idx}/{len(targets)}... ({(time.time()-t0)/idx:.2f}s/img)")
            
        img = cv2.imread(img_path)
        if img is None:
            n_skip += 1; continue
            
        faces_aura = app_aura.get(img)
        if len(faces_aura) == 0:
            n_skip += 1; continue
            
        aura_emb = faces_aura[0].normed_embedding
        
        pose = np.load(pose_path).astype(np.float32)
        face_2d = pose[FACE_SLICE, :2]
        
        R = head_rotation(face_2d, CANONICAL_TPL)
        yaw, pitch = yaw_pitch_from_R(R)
        
        X_aura.append(aura_emb)
        Y_pose.append([yaw, pitch])
        
    print(f"Extraction complete. Skipped {n_skip}.")
    
    X = np.array(X_aura)
    Y = np.array(Y_pose)  # (N, 2)
    
    print("Running Ridge Regression CV to decode pose from AuraFace...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = np.zeros_like(Y)
    
    for train_idx, test_idx in kf.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        Y_tr, Y_te = Y[train_idx], Y[test_idx]
        
        # Center Y to avoid intercept bias
        Y_mean = Y_tr.mean(axis=0)
        
        model = RidgeCV(alphas=np.logspace(-2, 4, 20), fit_intercept=True)
        model.fit(X_tr, Y_tr - Y_mean)
        
        y_pred[test_idx] = model.predict(X_te) + Y_mean
        
    r2_yaw = r2_score(Y[:, 0], y_pred[:, 0])
    r2_pitch = r2_score(Y[:, 1], y_pred[:, 1])
    
    print("-" * 60)
    print("Out-of-Fold R² Scores:")
    print(f"Yaw:   {r2_yaw:+.4f}")
    print(f"Pitch: {r2_pitch:+.4f}")
    print("-" * 60)
    
    if r2_yaw > 0.40 or r2_pitch > 0.40:
        print("=> LEAKY: AuraFace strongly memorizes head pose. You MUST use CFG dropout")
        print("   on the identity stream to avoid double-conditioning conflicts.")
    elif r2_yaw < 0.15 and r2_pitch < 0.15:
        print("=> CLEAN: AuraFace is effectively pose-invariant. The firewall holds.")
    else:
        print("=> MODERATE: Some pose leakage detected. CFG dropout recommended as a precaution.")

if __name__ == "__main__":
    main()
