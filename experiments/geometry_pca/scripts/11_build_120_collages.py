#!/usr/bin/env python3
"""
Generate collages for the 120 overnight identities.
Extracts face crops from the original images based on the enriched pose.npy.
"""
import os, sys, json
import numpy as np
from PIL import Image

ROOT = "/mnt/nas-ai-models/training-data/loras/hegre-14000px"
ENRICHED = "data/hegre_enriched"
MAP = "data/overnight_identity_map.json"
OUT = "output/collages_120"
MAX_DIM = 1024
FACE_SLICE = slice(23, 91)

def load_rgb(path, max_dim=MAX_DIM):
    Image.MAX_IMAGE_PIXELS = None
    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = min(1.0, max_dim / max(w, h))
    if s < 1.0:
        im = im.resize((int(w*s), int(h*s)), Image.LANCZOS)
    return np.asarray(im), s

def main():
    os.makedirs(OUT, exist_ok=True)
    mapping = json.load(open(MAP))
    
    by_id = {}
    for p, model in mapping.items():
        if model not in by_id: by_id[model] = []
        by_id[model].append(p)
        
    print(f"Generating collages for {len(by_id)} identities...")
    
    for model, paths in by_id.items():
        out_path = os.path.join(OUT, f"{model}.png")
        if os.path.exists(out_path):
            continue # skip if already done
            
        crops = []
        for rel in paths:
            # The enriched directory structure matches the root
            base_dir = rel.replace(".jpg", "")
            pose_path = os.path.join(ENRICHED, base_dir, "pose.npy")
            orig_path = os.path.join(ROOT, rel)
            
            if not os.path.exists(pose_path):
                continue
                
            try:
                pose = np.load(pose_path)
                face = pose[FACE_SLICE]
                # Normalize [-1, 1] back to pixel coords for the downscaled image
                img, scale = load_rgb(orig_path)
                h, w = img.shape[:2]
                
                # pose is [-1, 1], map to [0, w] and [0, h]
                px_x = (face[:, 0] + 1.0) / 2.0 * w
                px_y = (face[:, 1] + 1.0) / 2.0 * h
                
                mn_x, mx_x = px_x.min(), px_x.max()
                mn_y, mx_y = px_y.min(), px_y.max()
                
                cx, cy = (mn_x + mx_x) / 2, (mn_y + mx_y) / 2
                span = max(mx_x - mn_x, mx_y - mn_y) * 0.8
                
                x0, y0 = int(max(0, cx - span)), int(max(0, cy - span))
                x1, y1 = int(min(w, cx + span)), int(min(h, cy + span))
                
                crop = Image.fromarray(img[y0:y1, x0:x1]).resize((160, 160))
                crops.append(crop)
            except Exception as e:
                print(f"  {model} skip {os.path.basename(rel)}: {e}")
                
        if not crops:
            continue
            
        # tile in a grid, 5 per row
        cols = 5
        rows = (len(crops) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * 160, rows * 160), (20, 20, 20))
        for i, c in enumerate(crops):
            sheet.paste(c, ((i % cols) * 160, (i // cols) * 160))
        sheet.save(out_path)
        print(f"[{model}] saved with {len(crops)} crops")

if __name__ == "__main__":
    main()
