"""
Main training script for the Eidolon Encoder model.

This script orchestrates the training process, including:
- Setting up the device (CPU/GPU).
- Defining hyperparameters and configurations.
- Initializing the EidolonEncoder model, FLAME model (placeholder), loss function, and optimizer.
- Creating a FaceDataset and DataLoader for image data.
- Running the training loop over a specified number of epochs.
- Performing forward and backward passes (currently with placeholders for some components).
- Printing loss information.
- (Placeholder for saving the trained model).

Note: This script is a skeleton and requires further implementation of FLAME model
integration, landmark projection, rendering, and ground-truth landmark loading
for full functionality. The IMAGE_DIR constant must be set to a valid dataset path.
"""

# 1. Imports and Setup
import torch
from torch.utils.data import DataLoader
import numpy as np # For image unnormalization
import face_alignment # For landmark detection
# Assuming src.dataset, src.model, src.loss are in the Python path
# If train.py is in the root, and src is a subdirectory:
from src.dataset import FaceDataset
from src.model import EidolonEncoder # Assuming FLAME class is not yet defined or needed here
# from src.model import FLAME # Placeholder if you have a FLAME nn.Module
from src.loss import TotalLoss

# --- Hyperparameters and Config ---
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
NUM_EPOCHS = 50
# IMAGE_DIR is no longer used by FaceDataset, replaced by HF_DATASET_NAME
# IMAGE_DIR = "path/to/your/face/dataset" 
HF_DATASET_NAME = "nuwandaa/ffhq128" # Dataset name on Hugging Face Hub
HF_DATASET_SPLIT = "train"
NUM_COEFFS = 227 # The number you chose for your encoder
LOSS_WEIGHTS = {
    'pixel': 1.0,
    'landmark': 1e-4, # Landmarks are sensitive, start with a small weight
    'reg_shape': 1e-6,
    'reg_expression': 1e-6
}

# 2. Initialize everything
encoder = EidolonEncoder(num_coeffs=NUM_COEFFS).to(DEVICE)
# flame = FLAME().to(DEVICE) # Assuming your FLAME class is also an nn.Module
# renderer = ... # Your PyTorch3D renderer, needed for projecting landmarks
# cameras = ... # Your PyTorch3D camera, needed for projecting landmarks
loss_fn = TotalLoss(loss_weights=LOSS_WEIGHTS).to(DEVICE)
optimizer = torch.optim.Adam(encoder.parameters(), lr=LEARNING_RATE)

# Initialize the landmark detector. This will load its own model.
# Using '2d' for 2D landmarks. It will use DEVICE.
print("Initializing landmark detector...")
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device=str(DEVICE))
print("Landmark detector initialized.")

print(f"Initializing FaceDataset with Hugging Face dataset: {HF_DATASET_NAME}")
# FaceDataset now loads directly from Hugging Face
dataset = FaceDataset(hf_dataset_name=HF_DATASET_NAME, hf_dataset_split=HF_DATASET_SPLIT)
# For Hugging Face datasets, num_workers=0 is often safer to start with,
# as the Dataset object itself might not be easily picklable for multiprocessing,
# or it might handle its own parallelism.
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

print(f"Using device: {DEVICE}")
print(f"Starting training with LEARNING_RATE={LEARNING_RATE}, BATCH_SIZE={BATCH_SIZE}, NUM_EPOCHS={NUM_EPOCHS}")

# 3. The Training Loop
for epoch in range(NUM_EPOCHS):
    for i, batch in enumerate(data_loader):
        gt_images = batch['image'].to(DEVICE)
        
        # --- A. Get Ground Truth 2D Landmarks from the input image ---
        # Unnormalize images for face_alignment: (B,C,H,W) tensor to (B,H,W,C) numpy [0,255]
        unnormalized_images_np = gt_images.cpu().numpy().transpose(0, 2, 3, 1)
        # These are standard ImageNet mean/std used in FaceDataset transforms
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        unnormalized_images_np = (unnormalized_images_np * std + mean) * 255
        unnormalized_images_np = unnormalized_images_np.astype(np.uint8)

        # Get landmarks for the entire batch.
        # Convert the numpy array (B,H,W,C) to a tensor (B,C,H,W) as expected by the face_detector.
        images_for_fa = torch.from_numpy(unnormalized_images_np).permute(0, 3, 1, 2).float().to(DEVICE)
        
        # gt_landmarks_list contains a list of numpy arrays (or None if no face detected)
        gt_landmarks_list = fa.get_landmarks_from_batch(images_for_fa)
        
        # --- Process the batch to handle failures in landmark detection ---
        valid_indices = [idx for idx, lms in enumerate(gt_landmarks_list) if lms is not None]

        if not valid_indices:
            print(f"Warning: No faces detected in batch step {i}. Skipping.")
            continue
            
        # Filter the batch to only include images and landmarks that are valid
        gt_images_filtered = gt_images[valid_indices]
        valid_landmarks_list = [gt_landmarks_list[idx] for idx in valid_indices]
        
        # Convert the clean list of landmarks to a tensor
        gt_landmarks_2d = torch.from_numpy(np.array(valid_landmarks_list)).float().to(DEVICE)

        # --- Forward Pass (Encoder) ---
        optimizer.zero_grad()
        # Pass only the valid images to the encoder
        pred_coeffs_vec = encoder(gt_images_filtered) 
        
        # Deconstruct pred_coeffs_vec for regularization loss (based on filtered batch)
        num_shape_coeffs = 100  # Assuming these are defined or known
        num_expression_coeffs = 50
        
        shape_params = pred_coeffs_vec[:, :num_shape_coeffs]
        expression_params = pred_coeffs_vec[:, num_shape_coeffs : num_shape_coeffs + num_expression_coeffs]
        
        pred_coeffs_for_loss = {
            'shape': shape_params,
            'expression': expression_params
        }
        
        # --- Dummy tensors for unimplemented parts of the pipeline ---
        # These should be adjusted to match the filtered batch size
        current_batch_size = gt_images_filtered.size(0)
        
        # TODO: Instantiate FLAME model and pass the full deconstructed pred_coeffs_dict
        # pred_verts, pred_landmarks_3d = flame(**pred_coeffs_dict_full) # pred_coeffs_dict_full from filtered pred_coeffs_vec
        pred_verts_dummy = torch.zeros(current_batch_size, 5023, 3).to(DEVICE)

        # TODO: Project 3D landmarks to 2D screen space using the PyTorch3D camera
        # pred_landmarks_2d_model = cameras.transform_points_screen(pred_landmarks_3d_from_flame)[:, :, :2]
        pred_landmarks_2d_model_dummy = torch.zeros(current_batch_size, 68, 2).to(DEVICE)

        # TODO: Render the image using the predicted vertices and a renderer
        # rendered_images = renderer(pred_verts_for_renderer, ...)
        rendered_images_dummy = torch.zeros_like(gt_images_filtered) # Match filtered batch
        
        # --- C. Loss Calculation ---
        # Now using real gt_landmarks_2d and gt_images_filtered.
        # Other inputs are still dummies but sized for the filtered batch.
        total_loss, loss_dict = loss_fn(
            pred_coeffs_for_loss,       # Actual shape/expression params from filtered batch
            pred_verts_dummy,           # Dummy, but sized for filtered batch
            pred_landmarks_2d_model_dummy, # Dummy, but sized for filtered batch
            rendered_images_dummy,      # Dummy, but sized for filtered batch
            gt_images_filtered,         # Filtered ground-truth images
            gt_landmarks_2d             # Real, filtered ground-truth landmarks
        )

        # --- Backward Pass ---
        total_loss.backward() 
        optimizer.step()
        
        if i % 10 == 0:
            current_loss = total_loss.item()
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{i+1}/{len(data_loader)}], "
                  f"Batch Size (valid): {current_batch_size}, Loss: {current_loss:.4f}")
            # TODO: Log individual losses from loss_dict
            # print(f"    Losses: {loss_dict}") # Example logging

print("Training finished (skeleton).")

# --- Save the final model ---
# torch.save(encoder.state_dict(), 'eidolon_encoder_final.pth')
# print("Encoder model saved to eidolon_encoder_final.pth")
