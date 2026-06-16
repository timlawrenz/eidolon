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
        _mtcnn = MTCNN(keep_all=True, device=device, image_size=512, margin=20, min_face_size=40)
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
        original_width = img.width
        original_height = img.height
        # Downscale 14000px monstrosities so MTCNN's image pyramid doesn't OOM the 4090
        if img.width > 4000 or img.height > 4000:
            scale = min(4000 / img.width, 4000 / img.height)
            new_w = int(img.width * scale)
            new_h = int(img.height * scale)
            img = img.resize((new_w, new_h), getattr(Image.Resampling, "LANCZOS", getattr(Image, "LANCZOS", 1)))
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
            
    # Scale boxes back up to original image coordinates
    if img.width != original_width or img.height != original_height:
        scale_x = original_width / img.width
        scale_y = original_height / img.height
        scaled_boxes = []
        for box in boxes:
            scaled_boxes.append([
                box[0] * scale_x,
                box[1] * scale_y,
                box[2] * scale_x,
                box[3] * scale_y
            ])
        boxes = scaled_boxes
            
        # Reopen full resolution image to perform the actual high-quality crop
        img = Image.open(image_path).convert("RGB")
            
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
            elif face_crop.width < max_dim or face_crop.height < max_dim:
                # Upscale small faces to exactly 512x512
                face_crop = face_crop.resize((max_dim, max_dim), resample_filter)
                
            face_crop.save(out_path, quality=95)
            saved.append(str(out_path.relative_to(output_dir)))
        except Exception as e:
            print(f"ERROR cropping {out_name}: {e}")
            
    return saved

def extract_all(manifest: dict, output_dir: Path, device="cuda:0", max_workers=4):
    import itertools
    from tqdm import tqdm
    
    mtcnn = get_mtcnn(device)
    
    # Breadth-first task generation (round-robin across identities)
    tasks_per_identity = []
    
    print("Scanning manifest for un-extracted faces...")
    for identity, entries in manifest.items():
        ident_tasks = []
        for entry in entries:
            # Idempotency check: see if the crop already exists BEFORE adding to the queue.
            # This makes resuming nearly instantaneous instead of spinning up thread pool workers
            # just to check disk presence.
            img_basename = Path(entry["filename"]).stem
            out_dir = output_dir / "faces" / identity / entry["set_slug"]
            if not (out_dir / f"{img_basename}_face1.jpg").exists():
                ident_tasks.append((entry["image_path"], identity, entry["set_slug"], entry["filename"]))
        if ident_tasks:
            tasks_per_identity.append(ident_tasks)
            
    tasks = []
    for task_batch in itertools.zip_longest(*tasks_per_identity):
        for task in task_batch:
            if task is not None:
                tasks.append(task)
                
    if not tasks:
        print("All faces in manifest have already been extracted.")
        return []
        
    print(f"Found {len(tasks)} images remaining to process. Starting extraction...")
    
    def _process(task):
        img_path, ident, slug, fname = task
        return extract_faces_for_image(img_path, output_dir, ident, slug, fname, mtcnn)
        
    futures = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Wrap the mapping in tqdm for a progress bar
        for result in tqdm(executor.map(_process, tasks), total=len(tasks), desc="Extracting faces"):
            futures.append(result)
            
    return futures
