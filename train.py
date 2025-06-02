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
# import face_alignment # No longer needed for on-the-fly detection
import os # For os.makedirs and os.path.join
# Assuming src.dataset, src.model, src.loss are in the Python path
# If train.py is in the root, and src is a subdirectory:
from src.dataset import FaceDataset
from src.model import EidolonEncoder, FLAME # Import FLAME model
from src.loss import TotalLoss
from src.utils import save_validation_images # Import the new utility function
import pickle # For loading FLAME model faces
from torch.utils.tensorboard import SummaryWriter # For TensorBoard logging
import torchvision # For making image grids for TensorBoard

# PyTorch3D imports for renderer and camera
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform, FoVPerspectiveCameras, PointLights, RasterizationSettings,
    MeshRenderer, MeshRasterizer, SoftPhongShader, TexturesVertex
)

# Initialize the SummaryWriter
# This will create a directory 'runs/project_eidolon_experiment1' for your logs
# You can change 'project_eidolon_experiment1' for different experiments
writer = SummaryWriter('runs/project_eidolon_experiment1')

# --- Hyperparameters and Config ---
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 1e-4
BATCH_SIZE = 128 # Start small (e.g., 8-16) and increase if memory allows
NUM_EPOCHS = 50
IMAGE_DIR = "data/ffhq_thumbnails_128" # Directory for pre-processed images
LANDMARK_DIR = "data/ffhq_landmarks_128" # Directory for pre-computed landmarks
NUM_COEFFS = 227 # Total number of FLAME parameters the encoder will predict
# Example breakdown (adjust based on your actual FLAME parameterization):
NUM_SHAPE_COEFFS = 100
NUM_EXPRESSION_COEFFS = 50
NUM_GLOBAL_POSE_COEFFS = 6 # e.g., axis-angle for global rotation
NUM_JAW_POSE_COEFFS = 3    # Jaw pose
NUM_EYE_POSE_COEFFS = 6    # Left and right eye pose (3 each)
NUM_NECK_POSE_COEFFS = 3   # Neck pose
NUM_TRANSLATION_COEFFS = 3 # Global translation

# Remaining coefficients, e.g., for texture, lighting, or other details
# Calculated as: NUM_COEFFS - (sum of above)
# Current sum: 100+50+6+3+6+3+3 = 171
# NUM_COEFFS = 227, so 227 - 171 = 56
NUM_DETAIL_COEFFS = 56 
# Ensure NUM_COEFFS == SUM_OF_ALL_DECONSTRUCTED_PARTS
FLAME_MODEL_PKL_PATH = './data/flame_model/flame2023.pkl'
LANDMARK_EMBEDDING_PATH = './data/flame_model/flame_static_embedding.pkl' # Updated path

VISUALIZATION_INTERVAL = 500 # Steps between generating validation images
LOGGING_INTERVAL = 10 # Steps between printing loss

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

# flame = FLAME().to(DEVICE) # Assuming your FLAME class is also an nn.Module
# renderer = ... # Your PyTorch3D renderer, needed for projecting landmarks
# cameras = ... # Your PyTorch3D camera, needed for projecting landmarks
loss_fn = TotalLoss(loss_weights=LOSS_WEIGHTS).to(DEVICE)
optimizer = torch.optim.Adam(encoder.parameters(), lr=LEARNING_RATE)

# Initialize FLAME model
# Pass paths and parameter dimensions to the FLAME model constructor
flame_model = FLAME(
    flame_model_path=FLAME_MODEL_PKL_PATH,
    landmark_embedding_path=LANDMARK_EMBEDDING_PATH,
    n_shape=NUM_SHAPE_COEFFS,
    n_exp=NUM_EXPRESSION_COEFFS
).to(DEVICE)

# FLAME faces are now loaded within the FLAME class, access via flame_model.faces_idx
# flame_faces_tensor = flame_model.faces_idx # This is already on DEVICE if registered as buffer

# Setup PyTorch3D renderer and cameras (similar to main.py)
R, T = look_at_view_transform(dist=2.0, elev=0, azim=0) # Adjusted dist for potential variance
cameras = FoVPerspectiveCameras(device=DEVICE, R=R, T=T)
raster_settings = RasterizationSettings(image_size=224, blur_radius=0.0, faces_per_pixel=1) # Match image size
lights = PointLights(device=DEVICE, location=[[0.0, 0.0, 3.0]])
# Using a simple shader. For albedo/texture, a different shader might be needed later.
shader = SoftPhongShader(device=DEVICE, cameras=cameras, lights=lights)
renderer = MeshRenderer(
    rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
    shader=shader
)

print(f"Initializing FaceDataset with images from: {IMAGE_DIR} and landmarks from: {LANDMARK_DIR}")
dataset = FaceDataset(image_dir=IMAGE_DIR, landmark_dir=LANDMARK_DIR)
data_loader = DataLoader(
    dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=8,
    pin_memory=True
)

print(f"Using device: {DEVICE}")
print(f"Starting training with LEARNING_RATE={LEARNING_RATE}, BATCH_SIZE={BATCH_SIZE}, NUM_EPOCHS={NUM_EPOCHS}")

# Helper function to deconstruct the coefficient vector
# This needs to match how your FLAME model expects its parameters
# and the order/size defined by NUM_SHAPE_COEFFS, etc.
def deconstruct_flame_coeffs(pred_coeffs_vec_batch):
    batch_size = pred_coeffs_vec_batch.shape[0]
    current_idx = 0
    
    shape_params = pred_coeffs_vec_batch[:, current_idx : current_idx + NUM_SHAPE_COEFFS]
    current_idx += NUM_SHAPE_COEFFS
    
    expression_params = pred_coeffs_vec_batch[:, current_idx : current_idx + NUM_EXPRESSION_COEFFS]
    current_idx += NUM_EXPRESSION_COEFFS
    
    # Global pose (e.g., axis-angle)
    pose_params = pred_coeffs_vec_batch[:, current_idx : current_idx + NUM_GLOBAL_POSE_COEFFS]
    current_idx += NUM_GLOBAL_POSE_COEFFS

    # Jaw pose
    jaw_pose_params = pred_coeffs_vec_batch[:, current_idx : current_idx + NUM_JAW_POSE_COEFFS]
    current_idx += NUM_JAW_POSE_COEFFS
    
    # Eye pose (left and right eye)
    eye_pose_params = pred_coeffs_vec_batch[:, current_idx : current_idx + NUM_EYE_POSE_COEFFS]
    current_idx += NUM_EYE_POSE_COEFFS

    # Neck pose
    neck_pose_params = pred_coeffs_vec_batch[:, current_idx : current_idx + NUM_NECK_POSE_COEFFS]
    current_idx += NUM_NECK_POSE_COEFFS

    # Translation
    transl_params = pred_coeffs_vec_batch[:, current_idx : current_idx + NUM_TRANSLATION_COEFFS]
    current_idx += NUM_TRANSLATION_COEFFS

    # Detail/Other parameters (e.g., texture, lighting)
    detail_params = pred_coeffs_vec_batch[:, current_idx : current_idx + NUM_DETAIL_COEFFS]
    current_idx += NUM_DETAIL_COEFFS
    
    # Assert that all coefficients have been deconstructed
    assert current_idx == NUM_COEFFS, f"Mismatch in deconstructed coeffs: expected {NUM_COEFFS}, got {current_idx}"

    return {
        'shape_params': shape_params,
        'expression_params': expression_params,
        'pose_params': pose_params,
        'jaw_pose_params': jaw_pose_params,
        'eye_pose_params': eye_pose_params,
        'neck_pose_params': neck_pose_params,
        'transl': transl_params, # FLAME model might expect 'transl'
        'detail_params': detail_params
    }

# 3. The Training Loop
for epoch in range(NUM_EPOCHS):
    for i, batch in enumerate(data_loader):
        gt_images = batch['image'].to(DEVICE)
        gt_landmarks_2d = batch['gt_landmarks'].to(DEVICE) # Pre-loaded landmarks
        
        current_batch_size = gt_images.size(0)

        # --- Forward Pass ---
        optimizer.zero_grad()
        
        pred_coeffs_vec = encoder(gt_images)
        
        # Deconstruct the predicted coefficient vector into a dictionary for FLAME
        pred_coeffs_dict = deconstruct_flame_coeffs(pred_coeffs_vec)
        
        # Run the FLAME model to get 3D vertices and 3D landmarks
        # Note: The FLAME model used here is currently a placeholder (src/model.py)
        # and needs to be implemented with actual FLAME deformation logic.
        pred_verts, pred_landmarks_3d = flame_model(
            shape_params=pred_coeffs_dict['shape_params'],
            expression_params=pred_coeffs_dict['expression_params'],
            pose_params=pred_coeffs_dict['pose_params'],
            jaw_pose_params=pred_coeffs_dict['jaw_pose_params'],
            eye_pose_params=pred_coeffs_dict['eye_pose_params'],
            neck_pose_params=pred_coeffs_dict['neck_pose_params'],
            transl=pred_coeffs_dict['transl']
            # detail_params are deconstructed but not used by the FLAME placeholder model yet
        )
        
        # Project 3D landmarks to 2D screen space
        # cameras.transform_points_screen outputs (x, y, z_ndc), we only need x, y.
        image_size_for_projection = (raster_settings.image_size, raster_settings.image_size)
        pred_landmarks_2d_model = cameras.transform_points_screen(pred_landmarks_3d, image_size=image_size_for_projection)[:, :, :2]

        # Render the image using the predicted vertices
        # Create a batch of Meshes.
        # For textures, a generic gray color is used.
        # Ensure pred_verts is (B, N, 3) and flame_faces_tensor is (F, 3).
        # We need to repeat faces for each item in the batch if rendering multiple meshes
        # Or, if renderer supports batched Meshes with shared topology, that's simpler.
        # PyTorch3D Meshes can take a list of verts and faces.
        
        # Create a generic texture for the batch
        num_vertices_flame = pred_verts.shape[1]
        generic_vertex_colors = torch.ones_like(pred_verts) * 0.7 # Gray
        textures_batch = TexturesVertex(verts_features=generic_vertex_colors.to(DEVICE))

        meshes_batch = Meshes(
            verts=list(pred_verts), # List of (N,3) tensors
            faces=[flame_model.faces_idx] * current_batch_size, # Repeat faces for each mesh in batch
            textures=textures_batch
        )
        rendered_images = renderer(meshes_batch) # renderer outputs (B, H, W, C)
        # Loss function might expect (B, C, H, W), so permute if necessary
        rendered_images = rendered_images.permute(0, 3, 1, 2)[:, :3, :, :] # Keep only RGB, drop Alpha if present

        # --- C. Loss Calculation (Now with all REAL tensors) ---
        # For regularization, TotalLoss expects 'shape' and 'expression' keys
        # We pass the relevant parts of pred_coeffs_dict
        coeffs_for_loss_fn = {
            'shape': pred_coeffs_dict['shape_params'],
            'expression': pred_coeffs_dict['expression_params']
        }
        total_loss, loss_dict = loss_fn(
            coeffs_for_loss_fn,      
            pred_verts,          
            pred_landmarks_2d_model, 
            rendered_images,     
            gt_images,         
            gt_landmarks_2d      
        )

        # --- Backward Pass ---
        total_loss.backward() 
        optimizer.step()
        
        # Visual validation step
        if i % VISUALIZATION_INTERVAL == 0: # Every VISUALIZATION_INTERVAL steps
            encoder.eval() # Set model to evaluation mode
            with torch.no_grad(): # No gradients needed for validation
                # Take a few images from the current batch for visualization
                # Ensure there are enough images in the batch, otherwise take all
                num_val_samples = min(4, gt_images.shape[0]) 
                val_gt_images = gt_images[:num_val_samples]
                val_gt_landmarks = gt_landmarks_2d[:num_val_samples]

                val_pred_coeffs_vec = encoder(val_gt_images)
                val_pred_coeffs_dict = deconstruct_flame_coeffs(val_pred_coeffs_vec)
                
                val_pred_verts, val_pred_landmarks_3d = flame_model(
                    shape_params=val_pred_coeffs_dict['shape_params'],
                    expression_params=val_pred_coeffs_dict['expression_params'],
                    pose_params=val_pred_coeffs_dict['pose_params'],
                    jaw_pose_params=val_pred_coeffs_dict['jaw_pose_params'],
                    eye_pose_params=val_pred_coeffs_dict['eye_pose_params'],
                    neck_pose_params=val_pred_coeffs_dict['neck_pose_params'],
                    transl=val_pred_coeffs_dict['transl']
                )
                # image_size_for_projection is defined in the main loop scope
                val_pred_landmarks_2d_model = cameras.transform_points_screen(
                    val_pred_landmarks_3d, image_size=image_size_for_projection
                )[:, :, :2]
                
                # Create Meshes for visualization
                val_generic_vertex_colors = torch.ones_like(val_pred_verts) * 0.7 # Gray
                val_textures_batch = TexturesVertex(verts_features=val_generic_vertex_colors.to(DEVICE))
                
                val_meshes_batch = Meshes(
                    verts=list(val_pred_verts), # List of (N,3) tensors
                    faces=[flame_model.faces_idx] * val_pred_verts.shape[0], # Repeat faces for each mesh
                    textures=val_textures_batch
                )
                val_rendered_images = renderer(val_meshes_batch) # (B, H, W, C)
                val_rendered_images = val_rendered_images.permute(0, 3, 1, 2)[:, :3, :, :] # (B, C, H, W), RGB

                output_dir = "outputs/validation_images"
                os.makedirs(output_dir, exist_ok=True)
                save_path_prefix = os.path.join(output_dir, f"epoch_{epoch+1}_step_{i+1}")
                
                save_validation_images(
                    val_gt_images, val_rendered_images, 
                    val_gt_landmarks, val_pred_landmarks_2d_model,
                    save_path_prefix, # Pass the prefix, function will append _sample_idx.png
                    num_images=num_val_samples 
                )

                # --- TensorBoard Logging for Images ---
                # Unnormalize gt_images (assuming they are normalized like training data)
                mean_tb = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
                std_tb = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)
                val_gt_images_unnorm_tb = val_gt_images * std_tb + mean_tb
                
                # Create a grid of images
                # global_step is defined in the LOGGING_INTERVAL block, ensure it's available or define it here too
                # For simplicity, let's use the same global_step as scalar logging,
                # assuming VISUALIZATION_INTERVAL is a multiple of LOGGING_INTERVAL
                current_global_step = epoch * len(data_loader) + i 
                
                img_grid_gt = torchvision.utils.make_grid(val_gt_images_unnorm_tb.clamp(0,1)) 
                writer.add_image('Validation/ground_truth', img_grid_gt, current_global_step)
                
                # Rendered images are likely already in a good range [0,1] but clamp to be safe
                img_grid_rendered = torchvision.utils.make_grid(val_rendered_images.clamp(0,1))
                writer.add_image('Validation/prediction', img_grid_rendered, current_global_step)

            encoder.train() # Set model back to training mode
        
        if i % LOGGING_INTERVAL == 0:
            current_loss = total_loss.item()
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{i+1}/{len(data_loader)}], "
                  f"Batch Size: {current_batch_size}, Loss: {current_loss:.4f}")
            
            # --- TensorBoard Logging for Scalars ---
            global_step = epoch * len(data_loader) + i
            writer.add_scalar('Loss/train_total', total_loss.item(), global_step)
            
            # Log individual losses from loss_dict
            for loss_name, loss_value in loss_dict.items():
                if loss_name != 'total' and hasattr(loss_value, 'item'): # Ensure it's a tensor
                    writer.add_scalar(f'Loss/train_{loss_name}', loss_value.item(), global_step)
                elif loss_name != 'total': # if it's already a float/int
                     writer.add_scalar(f'Loss/train_{loss_name}', loss_value, global_step)
            
            # Log learning rate
            writer.add_scalar('Hyperparameters/learning_rate', LEARNING_RATE, global_step)
            # TODO: Log individual losses from loss_dict to console if desired
            # print(f"    Losses: {loss_dict}") # Example logging

print("Training finished (skeleton).")

# --- Save the final model ---
torch.save(encoder.state_dict(), 'eidolon_encoder_final.pth')
print("Encoder model saved to eidolon_encoder_final.pth")

writer.close() # Close the TensorBoard SummaryWriter
