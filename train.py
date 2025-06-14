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
NUM_EPOCHS = 20 # Total epochs for the final two-stage training schedule.
IMAGE_DIR = "data/ffhq_thumbnails_128" # Directory for pre-processed images
LANDMARK_DIR = "data/ffhq_landmarks_128" # Directory for pre-computed landmarks
NUM_COEFFS = 227 # Total number of FLAME parameters the encoder will predict
# Example breakdown (adjust based on your actual FLAME parameterization):
NUM_SHAPE_COEFFS = 100
NUM_EXPRESSION_COEFFS = 0 # Disabled as 'expressedirs' are missing from the FLAME model pkl
NUM_GLOBAL_POSE_COEFFS = 6 # e.g., axis-angle for global rotation
NUM_JAW_POSE_COEFFS = 3    # Jaw pose
NUM_EYE_POSE_COEFFS = 6    # Left and right eye pose (3 each)
NUM_NECK_POSE_COEFFS = 3   # Neck pose
NUM_TRANSLATION_COEFFS = 3 # Global translation

# Remaining coefficients, e.g., for texture, lighting, or other details
# Calculated as: NUM_COEFFS - (sum of above)
# Current sum (with expressions disabled): 100+0+6+3+6+3+3 = 121
# NUM_COEFFS = 227, so 227 - 121 = 106
NUM_DETAIL_COEFFS = NUM_COEFFS - (NUM_SHAPE_COEFFS + NUM_EXPRESSION_COEFFS + \
                                 NUM_GLOBAL_POSE_COEFFS + NUM_JAW_POSE_COEFFS + \
                                 NUM_EYE_POSE_COEFFS + NUM_NECK_POSE_COEFFS + \
                                 NUM_TRANSLATION_COEFFS)
# Ensure NUM_COEFFS == SUM_OF_ALL_DECONSTRUCTED_PARTS
FLAME_MODEL_PKL_PATH = './data/flame_model/flame2023.pkl'
DECA_LANDMARK_EMBEDDING_PATH = './data/flame_model/deca_landmark_embedding.npy' # Updated path for DECA landmarks

# VISUALIZATION_INTERVAL = 500 # Removed, snapshots are now per epoch.
# Define epochs for verbose LBS debugging (e.g., first, middle, last)
# This set will be checked against the current *overall* epoch index (0-based)
# NUM_EPOCHS will now be the total epochs across all stages.
# VERBOSE_LBS_DEBUG_EPOCHS will be calculated after total_epochs_all_stages is known.


# --- Multi-Stage Training Configuration ---
# Each stage is a dictionary with 'epochs' and 'loss_weights'.
TRAINING_STAGES = [
    {
        'name': 'Stage1_CoarseAlignment',
        'epochs': 5, # Number of epochs for this stage
        'learning_rate': 1e-5, # Lower LR for stabilization
        'loss_weights': {
            'pixel': 0.0,           # Disable pixel loss; focus on geometric fit
            'landmark': 0.2,        # Reduced landmark guidance to prevent distortion
            'reg_shape': 0.5,       # Keep shape regularization strong
            'reg_transl': 0.5,      # Keep translation regularization strong
            'reg_global_pose': 1.0, # Strong global pose regularization
            'reg_jaw_pose': 1.0,    # Moderated jaw pose regularization
            'reg_neck_pose': 1.0,   # Moderated neck pose regularization
            'reg_eye_pose': 1.0,    # Moderated eye pose regularization
            'reg_detail': 1e-3,
        }
    },
    {
        'name': 'Stage2_FinetuneShape',
        'epochs': 15, # Use a good number of epochs to find a stable shape
        'learning_rate': 1e-5, # Keep the small LR for stability
        'loss_weights': {
            'pixel': 0.0,
            'landmark': 0.2,        # Keep landmark weight moderate
            'reg_shape': 0.2,       # Relax shape regularization a bit
            'reg_transl': 0.1,
            'reg_global_pose': 0.1,
            'reg_jaw_pose': 0.5,
            'reg_neck_pose': 0.5,
            'reg_eye_pose': 0.5,
            'reg_detail': 1e-4,
        }
    }
    # Stage 3 was removed. After experimentation, it was found that the aggressive
    # landmark fitting in Stage 3 led to distorted, unrealistic face shapes (overfitting),
    # while only providing a minor improvement in 2D landmark alignment. The model from
    # the end of Stage 2 represents the best balance between accurate pose and plausible shape.
]

total_epochs_all_stages = sum(stage['epochs'] for stage in TRAINING_STAGES)
NUM_EPOCHS = total_epochs_all_stages # Update NUM_EPOCHS to be the total

VERBOSE_LBS_DEBUG_EPOCHS = {0, total_epochs_all_stages // 2, total_epochs_all_stages - 1} if total_epochs_all_stages > 0 else {0}


# Initial LOSS_WEIGHTS will be set by the first stage.
# We still need loss_fn initialized, it will be updated per stage.
INITIAL_LOSS_WEIGHTS_FOR_SETUP = TRAINING_STAGES[0]['loss_weights']
INITIAL_LEARNING_RATE_FOR_SETUP = TRAINING_STAGES[0].get('learning_rate', LEARNING_RATE)


# 2. Initialize everything
encoder = EidolonEncoder(num_coeffs=NUM_COEFFS).to(DEVICE)
# flame = FLAME().to(DEVICE) # Assuming your FLAME class is also an nn.Module
# renderer = ... # Your PyTorch3D renderer, needed for projecting landmarks
# cameras = ... # Your PyTorch3D camera, needed for projecting landmarks
loss_fn = TotalLoss(loss_weights=INITIAL_LOSS_WEIGHTS_FOR_SETUP).to(DEVICE) # Use initial weights for setup
optimizer = torch.optim.Adam(encoder.parameters(), lr=INITIAL_LEARNING_RATE_FOR_SETUP) # Use initial LR for setup

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
# Using a slightly larger distance and wider FoV for more stable initial projections.
# dist=2.7 is a common value in similar 3DMM fitting frameworks.
R, T = look_at_view_transform(dist=2.7, elev=0, azim=0) 
cameras = FoVPerspectiveCameras(device=DEVICE, R=R, T=T, fov=30.0) 
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
global_epoch_idx = 0 # Tracks the true overall epoch number (0-indexed)
for stage_idx, stage_config in enumerate(TRAINING_STAGES):
    print(f"DEBUG: MAIN LOOP - Starting Stage {stage_idx + 1} ({stage_config['name']})") # DEBUG PRINT
    stage_name = stage_config['name']
    num_epochs_this_stage = stage_config['epochs']
    stage_loss_weights = stage_config['loss_weights']
    stage_lr = stage_config.get('learning_rate', LEARNING_RATE) # Get LR for stage, default to global LR

    # Update optimizer learning rate for the current stage
    for param_group in optimizer.param_groups:
        param_group['lr'] = stage_lr
    
    print(f"\n--- Starting Training Stage: {stage_name} for {num_epochs_this_stage} epochs ---")
    print(f"Using Loss Weights: {stage_loss_weights}")
    print(f"Using Learning Rate: {stage_lr}")
    loss_fn.weights = stage_loss_weights # Update loss function weights for the current stage

    for current_stage_epoch_idx in range(num_epochs_this_stage): # Loops 0 to num_epochs_this_stage-1
        # 'epoch' variable must be updated with the current global_epoch_idx for THIS iteration
        epoch = global_epoch_idx 
        # Conditional print for epoch progress: 1st, every 5th, last epoch of stage
        if current_stage_epoch_idx == 0 or \
           (current_stage_epoch_idx + 1) % 5 == 0 or \
           current_stage_epoch_idx == num_epochs_this_stage - 1:
            print(f"DEBUG: MAIN LOOP - Global Epoch: {epoch + 1}, Stage Epoch: {current_stage_epoch_idx + 1}/{num_epochs_this_stage}") # DEBUG PRINT
        
        # <<< START of moved block: Batch processing and validation >>>
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
            pred_verts, pred_landmarks_3d = flame_model(
                shape_params=pred_coeffs_dict['shape_params'],
                expression_params=pred_coeffs_dict['expression_params'],
                pose_params=pred_coeffs_dict['pose_params'],
                jaw_pose_params=pred_coeffs_dict['jaw_pose_params'],
                eye_pose_params=pred_coeffs_dict['eye_pose_params'],
                neck_pose_params=pred_coeffs_dict['neck_pose_params'],
                transl=pred_coeffs_dict['transl']
            )
            
            image_size_for_projection = (raster_settings.image_size, raster_settings.image_size)
            pred_landmarks_2d_model = cameras.transform_points_screen(pred_landmarks_3d, image_size=image_size_for_projection)[:, :, :2]

            num_vertices_flame = pred_verts.shape[1]
            generic_vertex_colors = torch.ones_like(pred_verts) * 0.7 
            textures_batch = TexturesVertex(verts_features=generic_vertex_colors.to(DEVICE))

            meshes_batch = Meshes(
                verts=list(pred_verts), 
                faces=[flame_model.faces_idx] * current_batch_size, 
                textures=textures_batch
            )
            rendered_images = renderer(meshes_batch) 
            rendered_images = rendered_images.permute(0, 3, 1, 2)[:, :3, :, :]

            coeffs_for_loss_fn = {
                'shape': pred_coeffs_dict['shape_params'],
                'expression': pred_coeffs_dict['expression_params'],
                'transl': pred_coeffs_dict['transl'],
                'global_pose': pred_coeffs_dict['pose_params'],
                'jaw_pose': pred_coeffs_dict['jaw_pose_params'],
                'neck_pose': pred_coeffs_dict['neck_pose_params'],
                'eye_pose': pred_coeffs_dict['eye_pose_params'],
                'detail': pred_coeffs_dict['detail_params']
            }
            total_loss, loss_dict = loss_fn(
                coeffs_for_loss_fn,
                pred_verts,
                pred_landmarks_2d_model,
                rendered_images,     
                gt_images,
                gt_landmarks_2d_scaled
            )
            
            is_stage1_training_batch0 = (stage_idx == 0 and i == 0)
            is_verbose_epoch_training_batch0 = (i == 0 and epoch in VERBOSE_LBS_DEBUG_EPOCHS)
            
            if is_stage1_training_batch0 or is_verbose_epoch_training_batch0:
                print(f"--- DEBUG Training Landmark Coords (Epoch {epoch+1}, Batch {i}) ---")
                print(f"  gt_landmarks_2d_scaled[0, :5, :]:\n{gt_landmarks_2d_scaled[0, :5, :]}")
                print(f"  pred_landmarks_2d_model[0, :5, :]:\n{pred_landmarks_2d_model[0, :5, :]}")
                print(f"----------------------------------------------------")

            total_loss.backward() 
            optimizer.step()
            
        # --- EPOCH-END SNAPSHOT: Visual validation, TensorBoard logging, and detailed console output ---
        current_tensorboard_step = epoch + 1
        
        loss_pixel_val = loss_dict.get('pixel', torch.tensor(0.0)).item()
        loss_landmark_val = loss_dict.get('landmark', torch.tensor(0.0)).item()
        loss_reg_shape_val = loss_dict.get('reg_shape', torch.tensor(0.0)).item()
        loss_reg_expression_val = loss_dict.get('reg_expression', torch.tensor(0.0)).item()
        loss_total_val = total_loss.item()

        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} (Stage: {stage_name} - {current_stage_epoch_idx+1}/{num_epochs_this_stage}) Completed ---")
        loss_summary_str = f"  Last Batch Losses: Total: {loss_total_val:.4f}"
        for loss_name, loss_component in loss_dict.items():
            if loss_name != 'total':
                loss_summary_str += f", {loss_name.capitalize()}: {loss_component.item():.4f}"
        print(loss_summary_str)
        current_lr_for_log = optimizer.param_groups[0]['lr']
        print(f"  Config: Batch Size: {BATCH_SIZE}, LR: {current_lr_for_log}")

        writer.add_scalar('Loss/train_total_epoch_last_batch', loss_total_val, current_tensorboard_step)
        for loss_name, loss_value in loss_dict.items():
            if loss_name != 'total':
                val_to_log = loss_value.item() if hasattr(loss_value, 'item') else loss_value
                writer.add_scalar(f'Loss/train_{loss_name}_epoch_last_batch', val_to_log, current_tensorboard_step)
        writer.add_scalar('Hyperparameters/learning_rate_epoch', current_lr_for_log, current_tensorboard_step)

        encoder.eval() 
        with torch.no_grad(): 
            num_val_samples = min(4, gt_images.shape[0]) 
            val_gt_images = gt_images[:num_val_samples] 
            val_gt_landmarks_original_scale = gt_landmarks_2d_original_scale[:num_val_samples]
            
            _vis_original_landmark_img_width = 128.0
            _vis_original_landmark_img_height = 128.0
            _vis_target_projection_img_width = float(raster_settings.image_size)
            _vis_target_projection_img_height = float(raster_settings.image_size)
            _vis_scale_x = _vis_target_projection_img_width / _vis_original_landmark_img_width
            _vis_scale_y = _vis_target_projection_img_height / _vis_original_landmark_img_height

            val_gt_landmarks_scaled = val_gt_landmarks_original_scale.clone()
            val_gt_landmarks_scaled[..., 0] *= _vis_scale_x
            val_gt_landmarks_scaled[..., 1] *= _vis_scale_y
            val_gt_landmarks_for_vis = val_gt_landmarks_scaled

            val_pred_coeffs_vec = encoder(val_gt_images)
            val_pred_coeffs_dict = deconstruct_flame_coeffs(val_pred_coeffs_vec)
            
            debug_lbs_this_epoch = epoch in VERBOSE_LBS_DEBUG_EPOCHS
            if debug_lbs_this_epoch:
                print(f"--- INFO: Enabling verbose LBS debug prints for epoch {epoch+1} (using actual predicted params) ---")
            
            val_pred_verts, val_pred_landmarks_3d = flame_model(
                shape_params=val_pred_coeffs_dict['shape_params'],
                expression_params=val_pred_coeffs_dict['expression_params'],
                pose_params=val_pred_coeffs_dict['pose_params'], 
                jaw_pose_params=val_pred_coeffs_dict['jaw_pose_params'], 
                eye_pose_params=val_pred_coeffs_dict['eye_pose_params'], 
                neck_pose_params=val_pred_coeffs_dict['neck_pose_params'], 
                transl=val_pred_coeffs_dict['transl'],
                debug_print=debug_lbs_this_epoch
            )
            
            print(f"--- Predicted 3D Vertices Stats (Epoch {epoch+1} End, First Sample of Batch) ---")
            if val_pred_verts.numel() > 0 and val_pred_verts.shape[0] > 0:
                print(f"  Shape: {val_pred_verts.shape}")
                print(f"  X: mean={val_pred_verts[0, :, 0].mean().item():.4f}, std={val_pred_verts[0, :, 0].std().item():.4f}, "
                      f"min={val_pred_verts[0, :, 0].min().item():.4f}, max={val_pred_verts[0, :, 0].max().item():.4f}")
                print(f"  Y: mean={val_pred_verts[0, :, 1].mean().item():.4f}, std={val_pred_verts[0, :, 1].std().item():.4f}, "
                      f"min={val_pred_verts[0, :, 1].min().item():.4f}, max={val_pred_verts[0, :, 1].max().item():.4f}")
                print(f"  Z: mean={val_pred_verts[0, :, 2].mean().item():.4f}, std={val_pred_verts[0, :, 2].std().item():.4f}, "
                      f"min={val_pred_verts[0, :, 2].min().item():.4f}, max={val_pred_verts[0, :, 2].max().item():.4f}")
            else:
                print("  val_pred_verts is empty or has zero batch size.")
            
            print(f"--- Predicted 3D Landmarks Stats (Epoch {epoch+1} End, First Sample of Batch) ---")
            if val_pred_landmarks_3d.numel() > 0 and val_pred_landmarks_3d.shape[0] > 0:
                print(f"  Shape: {val_pred_landmarks_3d.shape}")
                print(f"  X: mean={val_pred_landmarks_3d[0, :, 0].mean().item():.4f}, std={val_pred_landmarks_3d[0, :, 0].std().item():.4f}, "
                      f"min={val_pred_landmarks_3d[0, :, 0].min().item():.4f}, max={val_pred_landmarks_3d[0, :, 0].max().item():.4f}")
                print(f"  Y: mean={val_pred_landmarks_3d[0, :, 1].mean().item():.4f}, std={val_pred_landmarks_3d[0, :, 1].std().item():.4f}, "
                      f"min={val_pred_landmarks_3d[0, :, 1].min().item():.4f}, max={val_pred_landmarks_3d[0, :, 1].max().item():.4f}")
                print(f"  Z: mean={val_pred_landmarks_3d[0, :, 2].mean().item():.4f}, std={val_pred_landmarks_3d[0, :, 2].std().item():.4f}, "
                      f"min={val_pred_landmarks_3d[0, :, 2].min().item():.4f}, max={val_pred_landmarks_3d[0, :, 2].max().item():.4f}")
            else:
                print("  val_pred_landmarks_3d is empty or has zero batch size.")

            print(f"--- Validation Predicted FLAME Parameters (Epoch {epoch+1} End) ---")
            for pname in ['shape_params', 'expression_params', 'pose_params', 'jaw_pose_params', 'neck_pose_params', 'eye_pose_params', 'transl']:
                if pname in val_pred_coeffs_dict:
                    p_tensor = val_pred_coeffs_dict[pname]
                    if p_tensor is not None and p_tensor.numel() > 0:
                        print(f"  {pname}: mean={p_tensor.mean().item():.4f}, std={p_tensor.std().item():.4f}, "
                              f"min={p_tensor.min().item():.4f}, max={p_tensor.max().item():.4f}")
                    else:
                        print(f"  {pname}: Not used or empty tensor.")
                else:
                    print(f"  {pname}: Not found in predicted coefficients.")
            print("--------------------------------------------------\n")

            _image_size_for_projection = (raster_settings.image_size, raster_settings.image_size)
            val_pred_landmarks_2d_model = cameras.transform_points_screen(
                val_pred_landmarks_3d, image_size=_image_size_for_projection
            )[:, :, :2]

            val_generic_vertex_colors = torch.ones_like(val_pred_verts) * 0.7
            val_textures_batch = TexturesVertex(verts_features=val_generic_vertex_colors.to(DEVICE))
            
            val_meshes_batch = Meshes(
                verts=list(val_pred_verts),
                faces=[flame_model.faces_idx] * val_pred_verts.shape[0],
                textures=val_textures_batch
            )
            val_rendered_images = renderer(val_meshes_batch).permute(0, 3, 1, 2)[:, :3, :, :]

            if epoch == 0: 
                print(f"\n--- DEBUG: TEMPLATE LANDMARK PROJECTION (Epoch {epoch+1}) ---")
                _bs_one = 1 
                template_shape_params = torch.zeros(_bs_one, NUM_SHAPE_COEFFS, device=DEVICE)
                template_expr_params = torch.zeros(_bs_one, NUM_EXPRESSION_COEFFS, device=DEVICE)
                template_pose_params = torch.zeros(_bs_one, NUM_GLOBAL_POSE_COEFFS, device=DEVICE)
                template_pose_params[:, 0] = 1.0 
                template_pose_params[:, 4] = 1.0
                template_jaw_pose_params = torch.zeros(_bs_one, NUM_JAW_POSE_COEFFS, device=DEVICE)
                template_eye_pose_params = torch.zeros(_bs_one, NUM_EYE_POSE_COEFFS, device=DEVICE)
                template_neck_pose_params = torch.zeros(_bs_one, NUM_NECK_POSE_COEFFS, device=DEVICE)
                template_transl_params = torch.zeros(_bs_one, NUM_TRANSLATION_COEFFS, device=DEVICE)
                _template_verts, _template_landmarks_3d = flame_model(
                    shape_params=template_shape_params, expression_params=template_expr_params,
                    pose_params=template_pose_params, jaw_pose_params=template_jaw_pose_params,
                    eye_pose_params=template_eye_pose_params, neck_pose_params=template_neck_pose_params,
                    transl=template_transl_params, debug_print=True
                )
                _template_landmarks_2d = cameras.transform_points_screen(
                    _template_landmarks_3d, image_size=_image_size_for_projection
                )[:, :, :2]
                print(f"  Template 2D Landmarks (first 5 points):\n{_template_landmarks_2d[0, :5, :]}")
                ascii_plot_template_lmks = plot_landmarks_ascii(
                    _template_landmarks_2d, 
                    original_img_width=_vis_target_projection_img_width,
                    original_img_height=_vis_target_projection_img_height,
                    title=f"Template Projected Landmarks (Epoch {epoch+1}, 224x224)"
                )
                print(ascii_plot_template_lmks)
                dummy_bg_for_template_lmks = torch.ones(1, 3, int(_vis_target_projection_img_height), int(_vis_target_projection_img_width), device=DEVICE) * 0.3
                template_lmks_img_tb = draw_landmarks_on_images_tensor(
                    dummy_bg_for_template_lmks, _template_landmarks_2d, color='green'
                )
                writer.add_image('Debug/template_projected_landmarks', torchvision.utils.make_grid(template_lmks_img_tb.clamp(0,1)), epoch + 1)
                print(f"--- END DEBUG: TEMPLATE LANDMARK PROJECTION ---\n")

            if epoch in VERBOSE_LBS_DEBUG_EPOCHS:
                print(f"--- DEBUG Actual Predicted Landmark Coords (Validation, Epoch {epoch+1}) ---")
                print(f"  val_gt_landmarks_scaled[0, :5, :]:\n{val_gt_landmarks_for_vis[0, :5, :]}") 
                print(f"  val_pred_landmarks_2d_model[0, :5, :]:\n{val_pred_landmarks_2d_model[0, :5, :]}")
                print(f"----------------------------------------------------")

            mean_tb = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
            std_tb = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)
            val_gt_images_unnorm_tb = val_gt_images * std_tb + mean_tb

            ascii_plot_gt = plot_landmarks_ascii(
                val_gt_landmarks_for_vis,
                original_img_width=_vis_target_projection_img_width, 
                original_img_height=_vis_target_projection_img_height,
                title=f"GT Landmarks (Epoch {epoch+1}, Scaled to 224x224)"
            )
            print(ascii_plot_gt)
            ascii_plot_pred = plot_landmarks_ascii(
                val_pred_landmarks_2d_model,
                original_img_width=_vis_target_projection_img_width,
                original_img_height=_vis_target_projection_img_height,
                title=f"Predicted Landmarks (Epoch {epoch+1}, 224x224)"
            )
            print(ascii_plot_pred)
            
            gt_images_tb_with_landmarks = draw_landmarks_on_images_tensor(
                val_gt_images_unnorm_tb, val_gt_landmarks_for_vis, color='red'
            )
            pred_images_tb_with_landmarks = draw_landmarks_on_images_tensor(
                val_rendered_images, val_pred_landmarks_2d_model, color='blue'
            )
            
            img_grid_gt = torchvision.utils.make_grid(gt_images_tb_with_landmarks.clamp(0,1))
            writer.add_image(f'Validation_Stage_{stage_idx+1}/ground_truth_with_landmarks', img_grid_gt, current_tensorboard_step)
            
            img_grid_rendered = torchvision.utils.make_grid(pred_images_tb_with_landmarks.clamp(0,1))
            writer.add_image(f'Validation_Stage_{stage_idx+1}/prediction_with_landmarks', img_grid_rendered, current_tensorboard_step)

        encoder.train() # Set model back to training mode
        print(f"DEBUG: MAIN LOOP - Completed Global Epoch {global_epoch_idx + 1}. Incrementing global_epoch_idx.") # DEBUG PRINT
        global_epoch_idx += 1 # Increment global_epoch_idx after each true epoch is completed
    print(f"DEBUG: MAIN LOOP - Finished Stage {stage_idx + 1}") # DEBUG PRINT

print("Training finished.")

# --- Save the final model ---
torch.save(encoder.state_dict(), 'eidolon_encoder_final.pth')
print("Encoder model saved to eidolon_encoder_final.pth")

writer.close() # Close the TensorBoard SummaryWriter
