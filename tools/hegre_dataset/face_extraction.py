import json
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

# Optional import, fallback to None if not installed for some tests
try:
    from facenet_pytorch import MTCNN
except ImportError:
    MTCNN = None

def get_square_box(box, img_width, img_height):
    """
    Convert a bounding box [x1, y1, x2, y2] to a square box that fits within img bounds.
    No padding is added. The box is shifted and clamped to stay within the image.
    """
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    
    # Target size is the maximum dimension to make a square
    size = max(w, h)
    
    # Calculate center
    cx = x1 + w / 2
    cy = y1 + h / 2
    
    # Initial square box
    nx1 = cx - size / 2
    ny1 = cy - size / 2
    nx2 = cx + size / 2
    ny2 = cy + size / 2
    
    # Shift if out of bounds (left/top)
    if nx1 < 0:
        nx2 -= nx1
        nx1 = 0
    if ny1 < 0:
        ny2 -= ny1
        ny1 = 0
        
    # Shift if out of bounds (right/bottom)
    if nx2 > img_width:
        shift = nx2 - img_width
        nx1 -= shift
        nx2 = img_width
        if nx1 < 0: # Image too small, clamp
            nx1 = 0
    if ny2 > img_height:
        shift = ny2 - img_height
        ny1 -= shift
        ny2 = img_height
        if ny1 < 0: # Image too small, clamp
            ny1 = 0
            
    return (int(nx1), int(ny1), int(nx2), int(ny2))


def extract_faces(dataset_path: Path):
    """
    Extract faces from images in the dataset and save them.
    Multi-face outputs are saved as <basename>_face<idx>.jpg (1-indexed).
    """
    manifest_file = dataset_path / "manifest.json"
    if not manifest_file.exists():
        logging.warning("Manifest not found.")
        return

    with open(manifest_file, "r") as f:
        manifest = json.load(f)

    if not MTCNN:
        raise RuntimeError("facenet_pytorch is required for face extraction.")

    mtcnn = MTCNN(keep_all=True, device='cpu') # multi-face detection
    
    def process_image(identity, rel_path):
        img_path = (dataset_path / rel_path).resolve()
        if not img_path.is_relative_to(dataset_path.resolve()):
            return
        if not img_path.exists():
            return
            
        parent_rel = Path(rel_path).parent
        img_basename = img_path.stem
        out_dir = (dataset_path / identity / parent_rel).resolve()
        if not out_dir.is_relative_to(dataset_path.resolve()):
            return
            
        out_dir.mkdir(parents=True, exist_ok=True)
        
        if (out_dir / f"{img_basename}_face1.jpg").exists():
            return

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            logging.error(f"Failed to open {img_path}: {e}")
            return

        boxes, probs = mtcnn.detect(img)
        
        if boxes is None:
            return
            
        for idx, box in enumerate(boxes):
            sq_box = get_square_box(box, img.width, img.height)
            crop = img.crop(sq_box)
            crop = crop.resize((512, 512), Image.Resampling.LANCZOS)
            
            out_path = out_dir / f"{img_basename}_face{idx+1}.jpg"
            crop.save(out_path)
            
    with ThreadPoolExecutor() as executor:
        futures = []
        for identity, image_paths in manifest.items():
            for rel_path in image_paths:
                futures.append(executor.submit(process_image, identity, rel_path))
        for future in futures:
            future.result()
