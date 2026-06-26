#!/usr/bin/env python3
import os
import sys
import numpy as np
import cv2
from pathlib import Path

# Must import from insightface before setting up paths sometimes, but let's setup sys.path first
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from geometry_pca.verification import verification_auc
from insightface.app import FaceAnalysis

def main():
    print("=" * 60)
    print("  Tier 0.1: AuraFace vs ArcFace Baseline (Legacy Dataset)")
    print("=" * 60)
    
    crops_dir = "/mnt/nas-ai-models/training-data/eidolon/geometry_pca_data/hegre_face_crops"
    if not os.path.exists(crops_dir):
        print(f"Error: Could not find {crops_dir}")
        return
    
    print("Loading ArcFace (buffalo_l)...")
    app_arc = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app_arc.prepare(ctx_id=0, det_size=(512, 512))
    
    print("Loading AuraFace...")
    app_aura = FaceAnalysis(name='auraface', root='/mnt/nas-ai-models', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app_aura.prepare(ctx_id=0, det_size=(512, 512))
    
    # 3. MAP: For each persona in gate names, collect all face crop images
    X_arc, X_aura, y_labels = [], [], []
    personas = sorted([p for p in os.listdir(crops_dir) if os.path.isdir(os.path.join(crops_dir, p))])
    
    n_skip = 0
    img_count = 0
    import time
    t0 = time.time()
    
    for pid, persona in enumerate(personas):
        persona_dir = os.path.join(crops_dir, persona)
        for img_name in sorted(os.listdir(persona_dir)):
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
                
            img_path = os.path.join(persona_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                n_skip += 1
                continue
                
            # 4. EXTRACT: cv2.imread -> face_app.get() -> .normed_embedding
            faces_arc = app_arc.get(img)
            faces_aura = app_aura.get(img)
            
            if len(faces_arc) == 0 or len(faces_aura) == 0:
                n_skip += 1
                continue
                
            # Take the largest face if multiple
            arc_emb = faces_arc[0].normed_embedding
            aura_emb = faces_aura[0].normed_embedding
            
            X_arc.append(arc_emb)
            X_aura.append(aura_emb)
            y_labels.append(pid)
            
            img_count += 1
            if img_count % 200 == 0:
                print(f"  Extracted {img_count} images... ({(time.time()-t0)/img_count:.2f}s/img)")
                
    print(f"\nExtraction complete. Processed {img_count} images across {len(personas)} personas. Skipped {n_skip}.")
    
    X_arc = np.array(X_arc)
    X_aura = np.array(X_aura)
    y = np.array(y_labels)
    
    # 5. GATE: verification_auc
    SEEDS = [0, 1, 2]
    auc_arc = [verification_auc(X_arc, y, n_pairs=40000, seed=s)[0] for s in SEEDS]
    auc_aura = [verification_auc(X_aura, y, n_pairs=40000, seed=s)[0] for s in SEEDS]
    
    arc_mean = np.mean(auc_arc)
    aura_mean = np.mean(auc_aura)
    dino_baseline = 0.797
    
    # 6. REPORT
    print("-" * 60)
    print("Verification AUC (N=40k balanced pairs per seed)")
    print("-" * 60)
    print(f"DINOv3 Patches (Phase 4): {dino_baseline:.4f}  [Baseline]")
    print(f"ArcFace (buffalo_l):      {arc_mean:.4f}  {[f'{a:.4f}' for a in auc_arc]}")
    print(f"AuraFace (fal-v1):        {aura_mean:.4f}  {[f'{a:.4f}' for a in auc_aura]}")
    
    print("-" * 60)
    delta_license = arc_mean - aura_mean
    print(f"Licensing Cost (ArcFace - AuraFace): {delta_license:+.4f} AUC")
    
    delta_dino = aura_mean - dino_baseline
    print(f"AuraFace vs DINO Baseline:           {delta_dino:+.4f} AUC")
    
    print("\nImplication:")
    if aura_mean > 0.85:
        print("=> Strongly supports AuraFace as the identity stream; DINO patches become secondary.")
    elif aura_mean >= 0.80:
        print("=> AuraFace is modestly better; keep both streams or choose based on architecture fit.")
    elif aura_mean >= 0.70:
        print("=> Comparable to DINO patches; decision is architectural, not performance-driven.")
    else:
        print("=> AuraFace is weaker; DINO patches remain the identity stream.")

if __name__ == "__main__":
    main()
