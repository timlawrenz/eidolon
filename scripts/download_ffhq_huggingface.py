
from datasets import load_dataset
import os
from tqdm import tqdm
from PIL import Image # PIL is an alias for Pillow, which should be installed via torchvision

# --- Configuration ---
# This will download the 1024x1024 version. We will resize it.
# Hugging Face doesn't have the 128x128 thumbnail version pre-packaged,
# so we'll create our own thumbnails.
DATASET_NAME = "ffhq" 
SAVE_DIR = "data/ffhq_thumbnails_128"
IMAGE_SIZE = 128

os.makedirs(SAVE_DIR, exist_ok=True)

print(f"Loading dataset '{DATASET_NAME}' from Hugging Face...")

# Load the dataset. It will download and cache it.
# The 'train' split is the only split for FFHQ.
dataset = load_dataset(DATASET_NAME, split='train')

print(f"Dataset loaded. Found {len(dataset)} images.")
print(f"Resizing and saving thumbnails to {SAVE_DIR} (target size: {IMAGE_SIZE}x{IMAGE_SIZE})...")

# --- Iterate, Resize, and Save ---
for i, example in enumerate(tqdm(dataset, desc="Processing images")):
    try:
        image = example['image']
        
        # Ensure image is in RGB format before resizing, if it's not already.
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Resize the image to our desired thumbnail size
        # Using ANTIALIAS for better quality downscaling
        thumbnail = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS) # Or Image.ANTIALIAS for older Pillow
        
        # Create a filename. We'll pad with zeros for nice sorting.
        filename = f"thumb_{i:05d}.png"
        save_path = os.path.join(SAVE_DIR, filename)
        
        # Save as PNG to preserve quality
        thumbnail.save(save_path)
    except Exception as e:
        print(f"Could not process image {i} (original filename might be {example.get('id', 'N/A')}): {e}")

print(f"Thumbnail creation complete. Images saved in {SAVE_DIR}")
