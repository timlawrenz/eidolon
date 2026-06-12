import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

try:
    from facenet_pytorch import MTCNN
except ImportError:
    MTCNN = None

# Reusing singleton pattern to avoid multiple model loads
_mtcnn = None

def get_mtcnn(device="cuda:0"):
    global _mtcnn
    if _mtcnn is None:
        if not MTCNN:
            raise RuntimeError("facenet_pytorch is required.")
        _mtcnn = MTCNN(keep_all=True, device=device)
    return _mtcnn

def get_square_box(box, img_width, img_height, expand_ratio=1.5):
    """Square crop box from MTCNN shifted/clamped to image bounds (no padding)."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    cx, cy = x1 + w / 2, y1 + h / 2
    side = max(w, h) * expand_ratio

    nx1, ny1 = cx - side / 2, cy - side / 2
    nx2, ny2 = cx + side / 2, cy + side / 2

    if nx1 < 0:
        nx2 -= nx1
        nx1 = 0
    if ny1 < 0:
        ny2 -= ny1
        ny1 = 0
    if nx2 > img_width:
        nx1 -= nx2 - img_width
        nx2 = img_width
    if ny2 > img_height:
        ny1 -= ny2 - img_height
        ny2 = img_height

    nx1 = max(0, nx1)
    ny1 = max(0, ny1)

    final_side = min(nx2 - nx1, ny2 - ny1)
    fx1, fy1 = cx - final_side / 2, cy - final_side / 2
    fx2, fy2 = cx + final_side / 2, cy + final_side / 2

    return [max(0, int(fx1)), max(0, int(fy1)), int(fx2), int(fy2)]

def extract_faces_for_image(image_path: str, output_dir: Path, identity: str, set_slug: str, filename: str, mtcnn, max_dim=1024, expand_ratio=1.5):
    """Extracts all faces from a single image and returns relative paths."""
    name_stem = os.path.splitext(filename)[0]
    ext = os.path.splitext(filename)[1]
    
    out_dir = output_dir / "faces" / identity / set_slug
    out_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"ERROR opening {image_path}: {e}")
        return []
        
    try:
        boxes, probs = mtcnn.detect(img)
    except Exception as e:
        print(f"ERROR detecting {image_path}: {e}")
        return []
        
    if boxes is None or len(boxes) == 0:
        return []
        
    saved = []
    resample_filter = getattr(Image.Resampling, "LANCZOS", getattr(Image, "LANCZOS", 1))
    
    for i, box in enumerate(boxes):
        face_index = i + 1
        out_name = f"{name_stem}_face{face_index}{ext}"
        out_path = out_dir / out_name
        
        if out_path.exists():
            saved.append(str(out_path.relative_to(output_dir)))
            continue
            
        try:
            sq_box = get_square_box(box, img.width, img.height, expand_ratio)
            face_crop = img.crop(tuple(sq_box))
            if face_crop.width > max_dim or face_crop.height > max_dim:
                face_crop = face_crop.resize((max_dim, max_dim), resample_filter)
            face_crop.save(out_path, quality=95)
            saved.append(str(out_path.relative_to(output_dir)))
        except Exception as e:
            print(f"ERROR cropping {out_name}: {e}")
            
    return saved

def extract_all(manifest: dict, output_dir: Path, device="cuda:0", max_workers=4):
    mtcnn = get_mtcnn(device)
    tasks = []
    for identity, entries in manifest.items():
        for entry in entries:
            tasks.append((entry["image_path"], identity, entry["set_slug"], entry["filename"]))
            
    def _process(task):
        img_path, ident, slug, fname = task
        return extract_faces_for_image(img_path, output_dir, ident, slug, fname, mtcnn)
        
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = list(executor.map(_process, tasks))
        
    print(f"Processed {len(futures)} images.")
    return futures
