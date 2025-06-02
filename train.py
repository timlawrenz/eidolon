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
# renderer = ... # Your PyTorch3D renderer
loss_fn = TotalLoss(loss_weights=LOSS_WEIGHTS).to(DEVICE)
optimizer = torch.optim.Adam(encoder.parameters(), lr=LEARNING_RATE)

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
        
        # --- Forward Pass ---
        optimizer.zero_grad()
        pred_coeffs_vec = encoder(gt_images)
        # TODO: Deconstruct pred_coeffs_vec into a dictionary for your FLAME model
        # pred_coeffs_dict = deconstruct_vector(pred_coeffs_vec) 
        
        # TODO: Instantiate FLAME model and pass coefficients
        # pred_verts, pred_landmarks_3d = flame(**pred_coeffs_dict)

        # TODO: Project 3D landmarks to 2D
        # pred_landmarks_2d = project_landmarks(pred_landmarks_3d, camera) # camera needs to be defined

        # TODO: Render the image using the predicted vertices
        # rendered_images = renderer(pred_verts, ...) # renderer needs to be defined
        
        # --- Loss Calculation ---
        # For now, let's assume dummy values for placeholders to avoid errors
        # Replace these with actual values as you implement the TODOs
        pred_coeffs_dict_dummy = {'shape': torch.zeros(gt_images.size(0), 100).to(DEVICE), 'expression': torch.zeros(gt_images.size(0), 50).to(DEVICE)} # Example
        pred_verts_dummy = torch.zeros(gt_images.size(0), 5023, 3).to(DEVICE) # Example: num_vertices for FLAME
        pred_landmarks_2d_dummy = torch.zeros(gt_images.size(0), 68, 2).to(DEVICE) # Example: 68 2D landmarks
        rendered_images_dummy = torch.zeros_like(gt_images) # Dummy rendered image
        gt_landmarks_2d_dummy = torch.zeros(gt_images.size(0), 68, 2).to(DEVICE) # Dummy ground truth landmarks


        # TODO: Get ground-truth landmarks for the batch
        # gt_landmarks_2d = ...
        
        # Use dummy values for now in the loss function call
        total_loss, loss_dict = loss_fn(
            pred_coeffs_dict_dummy,  # pred_coeffs_dict,
            pred_verts_dummy,        # pred_verts,
            pred_landmarks_2d_dummy, # pred_landmarks_2d,
            rendered_images_dummy,   # rendered_images,
            gt_images,
            gt_landmarks_2d_dummy    # gt_landmarks_2d
        )

        # --- Backward Pass ---
        # total_loss.backward() # This will error until all inputs to loss_fn are real tensors requiring grad
        # optimizer.step()
        
        if i % 10 == 0:
            # Ensure total_loss is a scalar tensor for .item()
            current_loss = total_loss.item() if torch.is_tensor(total_loss) else total_loss 
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{i+1}/{len(data_loader)}], Loss: {current_loss:.4f}")
            # TODO: Log individual losses from loss_dict
            # print(f"    Losses: {loss_dict}") # Example logging

print("Training finished (skeleton).")

# --- Save the final model ---
# torch.save(encoder.state_dict(), 'eidolon_encoder_final.pth')
# print("Encoder model saved to eidolon_encoder_final.pth")
