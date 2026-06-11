from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import sqlite3
import os
import json
from pathlib import Path

from facenet_pytorch import MTCNN
import torch
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

DB_URI = "file:experiments/geometry_pca/data/review.db?mode=ro"
SOURCE_ROOT = "/mnt/nas-ai-models/training-data/loras/hegre-14000px/"
# This data/ dir is a symlink to /mnt/nas-ai-models/training-data/eidolon/geometry_pca_data/
DEST_ROOT = "experiments/geometry_pca/data/hegre_faces/"

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=False, device=device)

def get_square_box(box, img_width, img_height, expand_ratio=1.5):
    # box is [x1, y1, x2, y2]
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    cx, cy = x1 + w/2, y1 + h/2
    
    # Expand box based on larger dimension to ensure face fits with margin
    side = max(w, h) * expand_ratio
    
    # Calculate new box coordinates
    nx1, ny1 = cx - side/2, cy - side/2
    nx2, ny2 = cx + side/2, cy + side/2
    
    # Shift box if it goes out of image bounds
    if nx1 < 0:
        nx2 -= nx1; nx1 = 0
    if ny1 < 0:
        ny2 -= ny1; ny1 = 0
    if nx2 > img_width:
        nx1 -= (nx2 - img_width); nx2 = img_width
    if ny2 > img_height:
        ny1 -= (ny2 - img_height); ny2 = img_height
        
    # If shifting pushed the other side out of bounds, clamp and enforce square
    nx1 = max(0, nx1)
    ny1 = max(0, ny1)
    
    # Final square size bounded by image edges
    final_side = min(nx2 - nx1, ny2 - ny1)
    
    # Recenter final box
    fx1 = cx - final_side/2
    fy1 = cy - final_side/2
    fx2 = cx + final_side/2
    fy2 = cy + final_side/2
    
    return [max(0, int(fx1)), max(0, int(fy1)), int(fx2), int(fy2)]

def process_image(img_path):
    try:
        # Lineage: Preserve folder structure and image name
        # e.g. /mnt/.../hegre-14000px/1000_yanna/img.jpg -> 1000_yanna/img.jpg
        rel_path = os.path.relpath(img_path, SOURCE_ROOT)
        # e.g. experiments/geometry_pca/data/hegre_faces/1000_yanna/img.jpg
        out_path = os.path.join(DEST_ROOT, rel_path)
        
        # Idempotency check
        if os.path.exists(out_path):
            return True, f"Already processed: {rel_path}"
            
        img = Image.open(img_path).convert('RGB')
        res = mtcnn.detect(img)
        # mtcnn.detect returns (boxes, probs) or (boxes, probs, points)
        if hasattr(res, '__len__'):
            boxes = res[0]
        else:
            boxes = res
            
        if boxes is None or len(boxes) == 0:
            return False, "No face detected"
            
        # Take highest confidence box (usually first)
        box = boxes[0]
        
        sq_box = get_square_box(box, img.width, img.height)
        face_crop = img.crop(tuple(sq_box))
        face_resized = face_crop.resize((1024, 1024), Image.Resampling.LANCZOS)
        
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        face_resized.save(out_path, quality=95)
        return True, out_path
    except Exception as e:
        return False, str(e)

def get_approved_images():
    with sqlite3.connect(DB_URI, uri=True) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT image_path FROM images WHERE status = 'approved'")
        return [row[0] for row in cursor.fetchall()]

if __name__ == "__main__":
    images = get_approved_images()
    print(f"Processing {len(images)} images...")
    
    # MTCNN is thread-safe on CPU but can bottleneck on GPU. 
    # Use max_workers=4 as a safe default for a mix of IO and inference.
    results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        for res in tqdm(executor.map(process_image, images), total=len(images)):
            results.append(res)
            
    successes = sum(1 for r in results if r[0])
    print(f"Completed: {successes}/{len(images)} successful.")
