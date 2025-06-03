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
from src.utils import save_validation_images, draw_landmarks_on_images_tensor, plot_landmarks_ascii # Import the new utility functions
import pickle # For loading FLAME model faces
from torch.utils.tensorboard import SummaryWriter # For TensorBoard logging
import torchvision # For making image grids for TensorBoard
import datetime # For timestamping log directories

# PyTorch3D imports for renderer and camera
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform, FoVPerspectiveCameras, PointLights, RasterizationSettings,
    MeshRenderer, MeshRasterizer, SoftPhongShader, TexturesVertex
)

# Initialize the SummaryWriter
timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M')
log_dir_name = f'runs/project_eidolon_{timestamp}'
writer = SummaryWriter(log_dir_name)
print(f"TensorBoard logs will be saved to: {log_dir_name}")

# --- Hyperparameters and Config ---
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 1e-4
BATCH_SIZE = 190 # Start small (e.g., 8-16) and increase if memory allows
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
DECA_LANDMARK_EMBEDDING_PATH = './data/flame_model/deca_landmark_embedding.npy' # Updated path for DECA landmarks

# VISUALIZATION_INTERVAL = 500 # Removed, snapshots are now per epoch.

LOSS_WEIGHTS = {
    'pixel': 1.0,
    'landmark': 1e-4, # Keep this for now, or even consider slightly increasing later
    'reg_shape': 1e-3,  # Increased from 1e-4
    'reg_expression': 1e-5 # << Maybe increase slightly too if expressions also look extreme
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
    deca_landmark_embedding_path=DECA_LANDMARK_EMBEDDING_PATH, # Updated argument name
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
        gt_images = batch['image'].to(DEVICE) # These are already transformed to 224x224 for the encoder
        gt_landmarks_2d_original_scale = batch['gt_landmarks'].to(DEVICE) # Shape (B, 68, 2) - in 128x128 space
        
        current_batch_size = gt_images.size(0)

        # Define original and target sizes for landmark scaling
        original_landmark_img_width = 128.0 # Width of images landmarks were detected on
        original_landmark_img_height = 128.0 # Height of images landmarks were detected on
        target_projection_img_width = float(raster_settings.image_size) # Should be 224.0
        target_projection_img_height = float(raster_settings.image_size) # Should be 224.0

        # Calculate scaling factors
        scale_x = target_projection_img_width / original_landmark_img_width
        scale_y = target_projection_img_height / original_landmark_img_height

        # Scale the ground truth landmarks
        gt_landmarks_2d_scaled = gt_landmarks_2d_original_scale.clone()
        gt_landmarks_2d_scaled[..., 0] *= scale_x # Scale x coordinates
        gt_landmarks_2d_scaled[..., 1] *= scale_y # Scale y coordinates
        # Now gt_landmarks_2d_scaled is in 224x224 space

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
            pred_landmarks_2d_model,  # This is already in 224x224 screen space
            rendered_images,     
            gt_images,                # This is the 224x224 input image
            gt_landmarks_2d_scaled    # USE THE SCALED VERSION HERE
        )

        # --- Backward Pass ---
        total_loss.backward() 
        optimizer.step()
        
    # --- EPOCH-END SNAPSHOT: Visual validation, TensorBoard logging, and detailed console output ---
    # This block runs once at the end of each epoch.
    # It uses variables from the last batch of the epoch (gt_images, gt_landmarks_2d_original_scale, loss_dict, total_loss).
    
    current_global_step = (epoch + 1) * len(data_loader) # Global step for TensorBoard
    
    # --- Console Logging (Epoch End) ---
    # Fetch specific loss values from the last batch for console logging
    loss_pixel_val = loss_dict.get('pixel', torch.tensor(0.0)).item()
    loss_landmark_val = loss_dict.get('landmark', torch.tensor(0.0)).item()
    loss_reg_shape_val = loss_dict.get('reg_shape', torch.tensor(0.0)).item()
    loss_reg_expression_val = loss_dict.get('reg_expression', torch.tensor(0.0)).item() # Get expression loss
    loss_total_val = total_loss.item() # From the last batch

    print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} Completed ---")
    # Updated print statement to include all individual losses
    loss_summary_str = f"  Last Batch Losses: Total: {loss_total_val:.4f}"
    for loss_name, loss_component in loss_dict.items():
        if loss_name != 'total': # Total is already included
            loss_summary_str += f", {loss_name.capitalize()}: {loss_component.item():.4f}"
    print(loss_summary_str)
    print(f"  Config: Batch Size: {BATCH_SIZE}, LR: {LEARNING_RATE}")

    # --- TensorBoard Scalar Logging (Epoch End) ---
    writer.add_scalar('Loss/train_total_epoch_last_batch', loss_total_val, current_global_step)
    for loss_name, loss_value in loss_dict.items():
        if loss_name != 'total': # total is already logged
            val_to_log = loss_value.item() if hasattr(loss_value, 'item') else loss_value
            writer.add_scalar(f'Loss/train_{loss_name}_epoch_last_batch', val_to_log, current_global_step)
    writer.add_scalar('Hyperparameters/learning_rate_epoch', LEARNING_RATE, current_global_step) # Log LR per epoch too

    # --- Validation Image Generation and TensorBoard Image Logging (Epoch End) ---
    encoder.eval() # Set model to evaluation mode
    with torch.no_grad(): # No gradients needed for validation
        # Use the last batch's gt_images for validation visualization
        num_val_samples = min(4, gt_images.shape[0]) 
        val_gt_images = gt_images[:num_val_samples] # From last batch of epoch
        val_gt_landmarks_original_scale = gt_landmarks_2d_original_scale[:num_val_samples] # From last batch
        
        # Recalculate scaling factors for these specific validation landmarks
        # (though they are constant if raster_settings.image_size doesn't change)
        _vis_original_landmark_img_width = 128.0
        _vis_original_landmark_img_height = 128.0
        _vis_target_projection_img_width = float(raster_settings.image_size)
        _vis_target_projection_img_height = float(raster_settings.image_size)
        _vis_scale_x = _vis_target_projection_img_width / _vis_original_landmark_img_width
        _vis_scale_y = _vis_target_projection_img_height / _vis_original_landmark_img_height

        val_gt_landmarks_scaled = val_gt_landmarks_original_scale.clone()
        val_gt_landmarks_scaled[..., 0] *= _vis_scale_x
        val_gt_landmarks_scaled[..., 1] *= _vis_scale_y
        val_gt_landmarks_for_vis = val_gt_landmarks_scaled # Use this for save_validation_images

        val_pred_coeffs_vec = encoder(val_gt_images)
        val_pred_coeffs_dict = deconstruct_flame_coeffs(val_pred_coeffs_vec)

        # --- Debug: Print Predicted FLAME Parameter Magnitudes (Epoch End) ---
        print(f"--- Validation Predicted FLAME Parameters (Epoch {epoch+1} End) ---")
        for pname in ['shape_params', 'expression_params', 'pose_params', 'jaw_pose_params', 'neck_pose_params', 'eye_pose_params', 'transl']:
            if pname in val_pred_coeffs_dict:
                p_tensor = val_pred_coeffs_dict[pname]
                print(f"  {pname}: mean={p_tensor.mean().item():.4f}, std={p_tensor.std().item():.4f}, "
                      f"min={p_tensor.min().item():.4f}, max={p_tensor.max().item():.4f}")
        print("--------------------------------------------------\n")
        # --- End Debug ---
        
        val_pred_verts, val_pred_landmarks_3d = flame_model(
            shape_params=val_pred_coeffs_dict['shape_params'],
            expression_params=val_pred_coeffs_dict['expression_params'],
            pose_params=val_pred_coeffs_dict['pose_params'],
            jaw_pose_params=val_pred_coeffs_dict['jaw_pose_params'],
            eye_pose_params=val_pred_coeffs_dict['eye_pose_params'],
            neck_pose_params=val_pred_coeffs_dict['neck_pose_params'],
            transl=val_pred_coeffs_dict['transl']
        )
        
        val_generic_vertex_colors = torch.ones_like(val_pred_verts) * 0.7
        val_textures_batch = TexturesVertex(verts_features=val_generic_vertex_colors.to(DEVICE))
        
        val_meshes_batch = Meshes(
            verts=list(val_pred_verts),
            faces=[flame_model.faces_idx] * val_pred_verts.shape[0],
            textures=val_textures_batch
        )
        val_rendered_images = renderer(val_meshes_batch).permute(0, 3, 1, 2)[:, :3, :, :]
        
        # image_size_for_projection is defined in the main training loop, ensure it's available
        # or redefine. It's (raster_settings.image_size, raster_settings.image_size)
        _image_size_for_projection_val = (raster_settings.image_size, raster_settings.image_size)
        val_pred_landmarks_2d_model = cameras.transform_points_screen(
            val_pred_landmarks_3d, image_size=_image_size_for_projection_val
        )[:, :, :2]

        # Images are now only logged to TensorBoard, not saved as separate files.
        # output_dir = "outputs/validation_images" # No longer saving separate files
        # os.makedirs(output_dir, exist_ok=True)
        # save_path_prefix = os.path.join(output_dir, f"epoch_{epoch+1}") 
        
        mean_tb = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
        std_tb = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)
        val_gt_images_unnorm_tb = val_gt_images * std_tb + mean_tb

        # The save_validation_images call is removed.
        # Visual output is handled by TensorBoard logging below.

        # --- ASCII Landmark Plotting for Console Debug ---
        # Plot GT landmarks (already scaled to target_projection_img_width/height)
        ascii_plot_gt = plot_landmarks_ascii(
            val_gt_landmarks_for_vis, 
            original_img_width=_vis_target_projection_img_width, # These are already in 224 space
            original_img_height=_vis_target_projection_img_height,
            title="GT Landmarks (Scaled to 224x224)"
        )
        print(ascii_plot_gt)

        # Plot predicted landmarks (also in target_projection_img_width/height space)
        ascii_plot_pred = plot_landmarks_ascii(
            val_pred_landmarks_2d_model,
            original_img_width=_vis_target_projection_img_width,
            original_img_height=_vis_target_projection_img_height,
            title="Predicted Landmarks (224x224)"
        )
        print(ascii_plot_pred)
        # --- End ASCII Landmark Plotting ---

        # Draw landmarks on images for TensorBoard
        # val_gt_images_unnorm_tb is (B,C,H,W) float [0,1]
        # val_gt_landmarks_for_vis is (B, N, 2) float, scaled
        # val_rendered_images is (B,C,H,W) float [0,1]
        # val_pred_landmarks_2d_model is (B, N, 2) float, scaled
        
        gt_images_tb_with_landmarks = draw_landmarks_on_images_tensor(
            val_gt_images_unnorm_tb, 
            val_gt_landmarks_for_vis, 
            color='red'
        )
        pred_images_tb_with_landmarks = draw_landmarks_on_images_tensor(
            val_rendered_images,
            val_pred_landmarks_2d_model,
            color='blue'
        )
        
        img_grid_gt = torchvision.utils.make_grid(gt_images_tb_with_landmarks.clamp(0,1)) 
        writer.add_image('Validation/ground_truth_with_landmarks_epoch_end', img_grid_gt, current_global_step)
        
        img_grid_rendered = torchvision.utils.make_grid(pred_images_tb_with_landmarks.clamp(0,1))
        writer.add_image('Validation/prediction_with_landmarks_epoch_end', img_grid_rendered, current_global_step)

    encoder.train() # Set model back to training mode

print("Training finished (skeleton).")

# --- Save the final model ---
torch.save(encoder.state_dict(), 'eidolon_encoder_final.pth')
print("Encoder model saved to eidolon_encoder_final.pth")

writer.close() # Close the TensorBoard SummaryWriter
