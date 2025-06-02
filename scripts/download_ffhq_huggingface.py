
from datasets import load_dataset
import os
from tqdm import tqdm
from PIL import Image # PIL is an alias for Pillow, which should be installed via torchvision
import numpy as np
import face_alignment
import torch # For device selection

# --- Configuration ---
# Using a dataset that provides 128x128 thumbnails directly.
DATASET_NAME = "nuwandaa/ffhq128" 
IMAGE_SAVE_DIR = "data/ffhq_thumbnails_128"
LANDMARK_SAVE_DIR = "data/ffhq_landmarks_128" # Directory for pre-computed landmarks
# Images from this dataset are expected to be 128x128.

os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)
os.makedirs(LANDMARK_SAVE_DIR, exist_ok=True)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE} for landmark detection.")

# Initialize landmark detector
print("Initializing landmark detector...")
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device=str(DEVICE))
print("Landmark detector initialized.")


print(f"Loading dataset '{DATASET_NAME}' (128x128 thumbnails) from Hugging Face...")
# Load the dataset. It will download and cache it.
# The 'train' split is the only split for FFHQ.
dataset = load_dataset(DATASET_NAME, split='train', trust_remote_code=True)

print(f"Dataset loaded. Found {len(dataset)} images.")
print(f"Saving thumbnails to {IMAGE_SAVE_DIR} and landmarks to {LANDMARK_SAVE_DIR}...")

# Standard landmark count for the 2D model used by face_alignment
EXPECTED_NUM_LANDMARKS = 68
EXPECTED_LANDMARK_DIM = 2
processed_count = 0
skipped_count = 0

# --- Iterate, Save Image, Detect and Save Landmarks ---
for i, example in enumerate(tqdm(dataset, desc="Processing images and landmarks")):
    try:
        pil_image = example['image'] # This should be a PIL Image object
        
        # Ensure image is in RGB format before saving
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Create image filename and save path
        image_filename = f"thumb_{i:05d}.png"
        image_save_path = os.path.join(IMAGE_SAVE_DIR, image_filename)
        
        # Save image as PNG
        pil_image.save(image_save_path)

        # Prepare image for landmark detection (numpy array)
        # face_alignment expects (H, W, C) numpy array
        np_image_for_fa = np.array(pil_image) 

        # Detect landmarks. fa.get_landmarks returns a list (even for single image).
        # We pass a single image, so expect a list with one element or None.
        landmarks_list = fa.get_landmarks(np_image_for_fa)

        current_face_landmarks_np = None
        if landmarks_list: # If list is not empty (face detected)
            lms_data = landmarks_list[0] # Get the landmarks for the (first) detected face
            
            # Handle potential multiple faces detected in one image (lms_data could be (num_faces, 68, 2))
            # or single face (68,2)
            if lms_data.ndim == 3: # Multiple faces detected
                if lms_data.shape[0] > 0 and lms_data.shape[1:] == (EXPECTED_NUM_LANDMARKS, EXPECTED_LANDMARK_DIM):
                    current_face_landmarks_np = lms_data[0] # Take the first face
                else:
                    print(f"Warning: Image {i} had multi-face landmarks of unexpected shape {lms_data.shape}. Skipping landmark save.")
            elif lms_data.ndim == 2: # Single face detected
                if lms_data.shape == (EXPECTED_NUM_LANDMARKS, EXPECTED_LANDMARK_DIM):
                    current_face_landmarks_np = lms_data
                else:
                    print(f"Warning: Image {i} had single-face landmarks of unexpected shape {lms_data.shape}. Skipping landmark save.")
            else:
                print(f"Warning: Image {i} had landmarks of unexpected ndim {lms_data.ndim} and shape {lms_data.shape}. Skipping landmark save.")
        else:
            print(f"Warning: No landmarks detected for image {i}. Skipping landmark save.")

        if current_face_landmarks_np is not None:
            # Create landmark filename and save path
            landmark_filename = f"thumb_{i:05d}.npy"
            landmark_save_path = os.path.join(LANDMARK_SAVE_DIR, landmark_filename)
            np.save(landmark_save_path, current_face_landmarks_np)
            processed_count += 1
        else:
            skipped_count +=1
            
    except Exception as e:
        print(f"Error processing image {i} (original ID might be {example.get('id', 'N/A')}): {e}")
        skipped_count +=1

print(f"Image and landmark processing complete.")
print(f"Images saved in {IMAGE_SAVE_DIR}")
print(f"Landmarks saved in {LANDMARK_SAVE_DIR}")
print(f"Successfully processed and saved landmarks for {processed_count} images.")
print(f"Skipped saving landmarks for {skipped_count} images (no face detected or shape mismatch).")
