
from datasets import load_dataset
import os
from tqdm import tqdm
from PIL import Image # PIL is an alias for Pillow, which should be installed via torchvision

# --- Configuration ---
# Using a dataset that provides 128x128 thumbnails directly.
DATASET_NAME = "nuwandaa/ffhq128" 
SAVE_DIR = "data/ffhq_thumbnails_128"
# Images from this dataset are expected to be 128x128.

os.makedirs(SAVE_DIR, exist_ok=True)

print(f"Loading dataset '{DATASET_NAME}' (128x128 thumbnails) from Hugging Face...")

# Load the dataset. It will download and cache it.
# The 'train' split is the only split for FFHQ.
dataset = load_dataset(DATASET_NAME, split='train')

print(f"Dataset loaded. Found {len(dataset)} images.")
print(f"Saving thumbnails to {SAVE_DIR}...")

# --- Iterate and Save ---
# Images are already 128x128, so no resizing is needed.
for i, example in enumerate(tqdm(dataset, desc="Processing images")):
    try:
        image = example['image'] # This should be a PIL Image object
        
        # Ensure image is in RGB format before saving, if it's not already.
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Create a filename. We'll pad with zeros for nice sorting.
        filename = f"thumb_{i:05d}.png"
        save_path = os.path.join(SAVE_DIR, filename)
        
        # Save as PNG to preserve quality
        image.save(save_path)
    except Exception as e:
        print(f"Could not process image {i} (original filename might be {example.get('id', 'N/A')}): {e}")

print(f"Thumbnail saving complete. Images saved in {SAVE_DIR}")
