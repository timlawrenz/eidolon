#!/usr/bin/env python3
import os
import sys
import sqlite3
import numpy as np
import cv2
from pathlib import Path

def get_legacy_intersection_targets():
    db_uri = "file:experiments/geometry_pca/data/review.db?mode=ro"
    stratum_root = "/mnt/nas-ai-models/training-data/eidolon/geometry_pca_data/hegre_faces_stratum"
    
    db = sqlite3.connect(db_uri, uri=True)
    c = db.cursor()
    c.execute('''
        SELECT i.enriched_dir, p.name, i.persona_id
        FROM images i JOIN personas p ON i.persona_id = p.id
        WHERE i.status = 'approved'
          AND i.persona_id NOT IN (
              SELECT persona_id FROM images WHERE status = 'tainted:contamination'
          )
    ''')
    rows = c.fetchall()
    db.close()
    
    targets = []
    for ed_orig, name, pid in rows:
        # ed_orig looks like: hegre_enriched/6850_anna-l-hegre-model/...
        rel_path = ed_orig.split("hegre_enriched/", 1)[1]
        stratum_dir = os.path.join(stratum_root, rel_path)
        
        req1 = os.path.join(stratum_dir, "dinov3_patches.npy")
        req2 = os.path.join(stratum_dir, "pose.npy")
        req3 = os.path.join(stratum_dir, "seg.npy")
        
        # The face crop itself is saved by stratum process inside the stratum directory!
        # Let's find the jpg inside stratum_dir or hegre_face_crops.
        # Actually, Phase 4 script says "FACE = /mnt/nas-ai-models/training-data/eidolon/geometry_pca_data/hegre_faces_stratum"
        # The crop is named 'face.jpg' inside the stratum directory? No, let's check what's inside.
        
        if os.path.exists(req1) and os.path.exists(req2) and os.path.exists(req3):
            targets.append((stratum_dir, pid))
            
    return targets

if __name__ == '__main__':
    targets = get_legacy_intersection_targets()
    print("Found", len(targets))
