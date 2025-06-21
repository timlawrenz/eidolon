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
from src.utils import save_validation_images, draw_landmarks_on_images_tensor, plot_landmarks_ascii, deconstruct_flame_coeffs, apply_coordinate_system_correction # Import the new utility functions
import pickle # For loading FLAME model faces
from torch.utils.tensorboard import SummaryWriter # For TensorBoard logging
import torchvision # For making image grids for TensorBoard
import datetime # For timestamping log directories

# PyTorch3D imports for renderer and camera
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform, FoVPerspectiveCameras, OrthographicCameras, PointLights, RasterizationSettings,
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
# NUM_EPOCHS is now determined by the sum of epochs in TRAINING_STAGES
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
LANDMARK_EMBEDDING_PATH = './data/flame_model/deca_landmark_embedding.npz'

# VISUALIZATION_INTERVAL = 500 # Removed, snapshots are now per epoch.
# Define epochs for verbose LBS debugging (e.g., first, middle, last)
# This set will be checked against the current *overall* epoch index (0-based)
# NUM_EPOCHS will now be the total epochs across all stages.
# VERBOSE_LBS_DEBUG_EPOCHS will be calculated after total_epochs_all_stages is known.

# --- Coordinate System Configuration ---
# Set this to enable/disable coordinate system correction for landmarks
USE_COORDINATE_CORRECTION = True
COORDINATE_CORRECTION_TYPE = 'flame_to_pytorch3d'  # Options: 'flame_to_pytorch3d', 'flame_to_pytorch3d_alt', 'none'

# --- Multi-Stage Training Configuration ---
# Each stage is a dictionary with 'epochs' and 'loss_weights'.
TRAINING_STAGES = [
    {
        'name': 'Stage1_OrthoCoarseFit',
        'epochs': 10,
        'camera_type': 'orthographic',
        'learning_rate': 1e-4,
        'use_posedirs': False, # Learn coarse shape/pose without fine details
        'loss_weights': {
            'pixel': 0.0,
            'landmark': 1.0,
            'landmark_shape': 1.0,     # Enforce relative landmark configuration
            'reg_shape': 1.0,
            'reg_transl': 1.0,
            # --- Increased Pose Regularization ---
            # Penalize deviation from a frontal pose more strongly in the first stage
            # to ensure a stable starting point for the fit.
            'reg_global_pose': 10.0,
            'reg_jaw_pose': 10.0,
            'reg_neck_pose': 10.0,
            'reg_eye_pose': 10.0,
            'reg_detail': 1e-4,
        }
    },
    {
        'name': 'Stage2_PerspectiveCoarse',
        'epochs': 10,
        'camera_type': 'perspective',
        'learning_rate': 1e-5,
        'use_posedirs': False, # Continue learning coarse shape/pose in perspective
        'loss_weights': {
            'pixel': 0.0,
            'landmark': 1.0,
            'landmark_shape': 1.0,
            'reg_shape': 0.5,
            'reg_transl': 1.0,
            'reg_global_pose': 1.0,
            'reg_jaw_pose': 1.0,
            'reg_neck_pose': 1.0,
            'reg_eye_pose': 1.0,
            'reg_detail': 1e-3,
        }
    },
    {
        'name': 'Stage3_PoseDetailFinetune',
        'epochs': 10,
        'camera_type': 'perspective',
        'learning_rate': 1e-6, # Use a very low learning rate for fine-tuning
        'use_posedirs': True, # Re-enable posedirs to learn fine details
        'loss_weights': {
            'pixel': 0.0,
            'landmark': 1.0,      # Focus strongly on final landmark accuracy
            'landmark_shape': 1.0,
            'reg_shape': 0.1,     # Relax shape regularization slightly for fine adjustments
            'reg_transl': 1.0,
            'reg_global_pose': 0.5,
            'reg_jaw_pose': 0.5,
            'reg_neck_pose': 0.5,
            'reg_eye_pose': 0.5,
            'reg_detail': 1e-4,
        }
    }
]

total_epochs_all_stages = sum(stage['epochs'] for stage in TRAINING_STAGES)
NUM_EPOCHS = total_epochs_all_stages # Update NUM_EPOCHS to be the total

# Define epochs for verbose LBS debugging. Set to empty to disable.
VERBOSE_LBS_DEBUG_EPOCHS = set()


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
print(f"DEBUG: Initial loss weights in loss_fn: {loss_fn.weights}")
optimizer = torch.optim.Adam(encoder.parameters(), lr=INITIAL_LEARNING_RATE_FOR_SETUP) # Use initial LR for setup

# Initialize FLAME model
# Pass paths and parameter dimensions to the FLAME model constructor
flame_model = FLAME(
    flame_model_path=FLAME_MODEL_PKL_PATH,
    deca_landmark_embedding_path=LANDMARK_EMBEDDING_PATH,
    n_shape=NUM_SHAPE_COEFFS,
    n_exp=NUM_EXPRESSION_COEFFS
).to(DEVICE)

# FLAME faces are now loaded within the FLAME class, access via flame_model.faces_idx

# Setup common rendering settings
raster_settings = RasterizationSettings(image_size=224, blur_radius=0.0, faces_per_pixel=1)
lights = PointLights(device=DEVICE, location=[[0.0, 0.0, 3.0]])

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
print(f"Coordinate system correction: {'ENABLED' if USE_COORDINATE_CORRECTION else 'DISABLED'} ({COORDINATE_CORRECTION_TYPE})")

# --- INITIAL COORDINATE SYSTEM VALIDATION ---
print("\n" + "="*80)
print("PERFORMING INITIAL SYSTEM VALIDATION")
print("="*80)

# First, test with orthographic camera (first stage)
R, T = look_at_view_transform(dist=10.0, elev=0, azim=0, device=DEVICE)
test_cameras_ortho = OrthographicCameras(device=DEVICE, R=R, T=T, focal_length=6.25)

from src.utils import validate_coordinate_system_consistency
validation_report = validate_coordinate_system_consistency(flame_model, test_cameras_ortho, DEVICE)
print(validation_report)

print("="*80)
print("SYSTEM VALIDATION COMPLETE - STARTING TRAINING")
print("="*80)

# 3. The Training Loop
global_epoch_idx = 0 # Tracks the true overall epoch number (0-indexed)

# Initialize tracking variables for complete run analysis
loss_history = []
param_history = {}
convergence_info = {}

for stage_idx, stage_config in enumerate(TRAINING_STAGES):
    stage_name = stage_config['name']
    num_epochs_this_stage = stage_config['epochs']
    stage_loss_weights = stage_config['loss_weights']
    stage_lr = stage_config.get('learning_rate', LEARNING_RATE)
    stage_camera_type = stage_config.get('camera_type', 'perspective')
    stage_use_posedirs = stage_config.get('use_posedirs', True) # Default to True for safety

    # --- Stage-specific Setup ---
    # Update optimizer learning rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = stage_lr
    
    # Update loss function weights
    loss_fn.weights = stage_loss_weights
    print(f"DEBUG: Stage {stage_idx+1} updated loss_fn.weights = {loss_fn.weights}")
    
    # Setup camera for the current stage
    if stage_camera_type == 'orthographic':
        print("--- Using Orthographic Camera for this stage ---")
        # Use look_at_view_transform for proper front view
        R, T = look_at_view_transform(dist=10.0, elev=0, azim=0, device=DEVICE)
        cameras = OrthographicCameras(device=DEVICE, R=R, T=T, focal_length=6.25)
    else: # 'perspective'
        print("--- Using Perspective Camera for this stage ---")
        # Use look_at_view_transform for proper front view
        R, T = look_at_view_transform(dist=2.7, elev=0, azim=0, device=DEVICE)
        cameras = FoVPerspectiveCameras(device=DEVICE, R=R, T=T, fov=12.0)

    # The renderer needs to be re-initialized if the camera changes
    shader = SoftPhongShader(device=DEVICE, cameras=cameras, lights=lights)
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=shader
    )

    print(f"\n--- Starting Training Stage: {stage_name} for {num_epochs_this_stage} epochs ---")
    print(f"Using Loss Weights: {stage_loss_weights}")
    print(f"Using Learning Rate: {stage_lr}")

    for current_stage_epoch_idx in range(num_epochs_this_stage):
        # 'epoch' variable must be updated with the current global_epoch_idx for THIS iteration
        epoch = global_epoch_idx
        
        # <<< START of moved block: Batch processing and validation >>>
        for i, batch in enumerate(data_loader):
            gt_images = batch['image'].to(DEVICE) # These are already transformed to 224x224 for the encoder
            # The FaceDataset already scales landmarks to the 224x224 space.
            gt_landmarks_2d_scaled = batch['gt_landmarks'].to(DEVICE) # Shape (B, 68, 2)
        
            current_batch_size = gt_images.size(0)

            # --- Forward Pass ---
            optimizer.zero_grad()
            
            pred_coeffs_vec = encoder(gt_images)
            
            # Deconstruct the predicted coefficient vector into a dictionary for FLAME
            pred_coeffs_dict = deconstruct_flame_coeffs(
                pred_coeffs_vec,
                NUM_SHAPE_COEFFS, NUM_EXPRESSION_COEFFS, NUM_GLOBAL_POSE_COEFFS,
                NUM_JAW_POSE_COEFFS, NUM_EYE_POSE_COEFFS, NUM_NECK_POSE_COEFFS,
                NUM_TRANSLATION_COEFFS, NUM_DETAIL_COEFFS
            )
            
            # --- COMPREHENSIVE DEBUGGING ---
            if epoch == 0 and i == 0:  # First batch of first epoch
                print("\n" + "="*80)
                print("COMPREHENSIVE DEBUG OUTPUT - FIRST BATCH")
                print("="*80)
                
                # 1. Validate input data
                from src.utils import validate_landmark_data, plot_landmarks_ascii
                print(validate_landmark_data(gt_landmarks_2d_scaled, gt_landmarks_2d_scaled, 
                                           image_size=224, title="Input Data Validation"))
                
                print(plot_landmarks_ascii(gt_landmarks_2d_scaled, 224, 224, 
                                         title="GT Landmarks Visualization"))
                
                # Check Ground Truth Landmark Validity
                gt_sample = gt_landmarks_2d_scaled[0].cpu().numpy()
                print(f"\nDEBUG: GT Landmark positions:")
                print(f"  Chin center (landmark 8): {gt_sample[8]}")
                print(f"  Left eye center (landmark 36): {gt_sample[36]}")
                print(f"  Right eye center (landmark 45): {gt_sample[45]}")
                print(f"  Nose tip (landmark 30): {gt_sample[30]}")
                
                # 2. Check pose parameters before FLAME
                pose_debug_dict = {
                    'shape_params': pred_coeffs_dict['shape_params'],
                    'global_pose': pred_coeffs_dict['pose_params'],
                    'jaw_pose': pred_coeffs_dict['jaw_pose_params'], 
                    'neck_pose': pred_coeffs_dict['neck_pose_params'],
                    'eye_pose': pred_coeffs_dict['eye_pose_params'],
                    'translation': pred_coeffs_dict['transl']
                }
                
                from src.utils import plot_pose_parameters_ascii
                print(plot_pose_parameters_ascii(pose_debug_dict, title="Predicted FLAME Parameters"))
                
                # Validate loss function calculations  
                print(f"\nDEBUG: Raw parameter values for regularization:")
                print(f"  global_pose: {pred_coeffs_dict['pose_params'][0].detach().cpu().numpy()}")
                print(f"  Target global_pose: [1, 0, 0, 0, 1, 0]")
                
                # Manual loss calculation
                target = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], device=DEVICE)
                manual_reg_loss = ((pred_coeffs_dict['pose_params'][0] - target) ** 2).mean()
                print(f"  Manual reg_global_pose loss: {manual_reg_loss.item()}")
                print(f"  Weight: {stage_loss_weights['reg_global_pose']}")
                print(f"  Expected weighted loss: {manual_reg_loss.item() * stage_loss_weights['reg_global_pose']}")
                
                print("="*80)
            
            # Every few batches, do lighter debugging
            if i % 50 == 0:  # Every 50 batches
                print(f"\n--- Quick Debug: Epoch {epoch+1}, Batch {i} ---")
                
                # Check for extreme pose values
                global_pose_vals = pred_coeffs_dict['pose_params'][0].detach().cpu().numpy()
                transl_vals = pred_coeffs_dict['transl'][0].detach().cpu().numpy()
                
                if np.abs(global_pose_vals).max() > 2.0:
                    print(f"*** WARNING: Extreme global pose detected: {global_pose_vals} ***")
                if np.abs(transl_vals).max() > 5.0:
                    print(f"*** WARNING: Extreme translation detected: {transl_vals} ***")

            # Run the FLAME model to get 3D vertices and 3D landmarks
            pred_verts, pred_landmarks_3d = flame_model(
                shape_params=pred_coeffs_dict['shape_params'],
                expression_params=pred_coeffs_dict['expression_params'],
                pose_params=pred_coeffs_dict['pose_params'],
                jaw_pose_params=pred_coeffs_dict['jaw_pose_params'],
                eye_pose_params=pred_coeffs_dict['eye_pose_params'],
                neck_pose_params=pred_coeffs_dict['neck_pose_params'],
                transl=pred_coeffs_dict['transl'],
                use_posedirs=stage_use_posedirs
            )
            
            # --- DEBUG FLAME OUTPUT ---
            if epoch == 0 and i == 0:  # First batch debugging
                print("\n--- FLAME Output Analysis ---")
                
                # Check 3D landmarks
                lmks_3d_sample = pred_landmarks_3d[0].detach().cpu().numpy()
                print(f"3D Landmarks shape: {pred_landmarks_3d.shape}")
                for axis_idx, axis_name in enumerate(['X', 'Y', 'Z']):
                    values = lmks_3d_sample[:, axis_idx]
                    print(f"  3D {axis_name}: range=[{values.min():.4f}, {values.max():.4f}], mean={values.mean():.4f}")
                
                # Check for degenerate mesh (all vertices at origin, etc.)
                verts_sample = pred_verts[0].detach().cpu().numpy()
                verts_center = verts_sample.mean(axis=0)
                verts_spread = verts_sample.std(axis=0)
                print(f"Mesh center: {verts_center}")
                print(f"Mesh spread (std): {verts_spread}")
                
                if np.all(verts_spread < 0.01):
                    print("*** CRITICAL: Mesh appears collapsed to a point! ***")
            
            # === COORDINATE SYSTEM CORRECTION FOR LANDMARKS ===
            # Apply coordinate system correction to 3D landmarks before projection
            if USE_COORDINATE_CORRECTION:
                pred_landmarks_3d_corrected = apply_coordinate_system_correction(
                    pred_landmarks_3d, COORDINATE_CORRECTION_TYPE
                )
                if epoch == 0 and i == 0:
                    print(f"\n--- Coordinate System Correction Applied ---")
                    print(f"Correction type: {COORDINATE_CORRECTION_TYPE}")
                    
                    # Show before/after for first few landmarks
                    orig_sample = pred_landmarks_3d[0][:5].detach().cpu().numpy()
                    corr_sample = pred_landmarks_3d_corrected[0][:5].detach().cpu().numpy()
                    print(f"Original 3D landmarks (first 5):")
                    for j, (orig, corr) in enumerate(zip(orig_sample, corr_sample)):
                        print(f"  Landmark {j}: {orig} -> {corr}")
            else:
                pred_landmarks_3d_corrected = pred_landmarks_3d
                if epoch == 0 and i == 0:
                    print(f"\n--- No Coordinate System Correction Applied ---")
            
            # Project corrected landmarks to 2D
            image_size_for_projection = (raster_settings.image_size, raster_settings.image_size)
            pred_landmarks_2d_model = cameras.transform_points_screen(
                pred_landmarks_3d_corrected, image_size=image_size_for_projection
            )[:, :, :2]
            
            # --- DEBUG CAMERA PROJECTION ---
            if epoch == 0 and i == 0:
                from src.utils import validate_camera_projection, plot_landmarks_ascii
                
                # Test both uncorrected and corrected projections
                if USE_COORDINATE_CORRECTION:
                    print("\n--- Projection Comparison (Uncorrected vs Corrected) ---")
                    
                    # Project uncorrected for comparison
                    landmarks_2d_uncorrected = cameras.transform_points_screen(
                        pred_landmarks_3d, image_size=image_size_for_projection
                    )[:, :, :2]
                    
                    print(validate_camera_projection(pred_landmarks_3d, cameras, image_size_for_projection, 
                                                   "UNCORRECTED Camera Projection"))
                    
                    print(plot_landmarks_ascii(landmarks_2d_uncorrected, 224, 224, 
                                             title="UNCORRECTED Projected Landmarks"))
                    
                    print(validate_camera_projection(pred_landmarks_3d_corrected, cameras, image_size_for_projection, 
                                                   "CORRECTED Camera Projection"))
                    
                    print(plot_landmarks_ascii(pred_landmarks_2d_model, 224, 224, 
                                             title="CORRECTED Projected Landmarks"))
                else:
                    print(validate_camera_projection(pred_landmarks_3d, cameras, image_size_for_projection, 
                                                   "Camera Projection Debug"))
                    
                    print(plot_landmarks_ascii(pred_landmarks_2d_model, 224, 224, 
                                             title="Projected Landmarks Visualization"))
                
                # Validate projected landmarks against GT
                print(validate_landmark_data(gt_landmarks_2d_scaled, pred_landmarks_2d_model, 
                                           image_size=224, title="GT vs Predicted Landmarks"))

            # Use vertices directly for rendering (no coordinate correction needed for mesh)
            pred_verts_for_render = pred_verts
            
            num_vertices_flame = pred_verts_for_render.shape[1]
            generic_vertex_colors = torch.ones_like(pred_verts_for_render) * 0.7 
            textures_batch = TexturesVertex(verts_features=generic_vertex_colors.to(DEVICE))

            meshes_batch = Meshes(
                verts=list(pred_verts_for_render), 
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
            
            # --- DEBUG LOSS COMPONENTS ---
            if epoch == 0 and i == 0:
                from src.utils import plot_loss_components_ascii
                print(plot_loss_components_ascii(loss_dict, title="Loss Components Analysis"))
                
                # Check for problematic loss values
                for loss_name, loss_val in loss_dict.items():
                    val = loss_val.item() if hasattr(loss_val, 'item') else float(loss_val)
                    if val > 1000:
                        print(f"*** WARNING: Very high {loss_name} loss: {val} ***")
                    elif val != val:  # Check for NaN
                        print(f"*** CRITICAL: NaN detected in {loss_name} loss! ***")
            
            # Monitor loss components every 50 batches
            if i % 50 == 0:
                landmark_loss_val = loss_dict['landmark'].item()
                total_loss_val = loss_dict['total'].item()
                print(f"  Landmark Loss: {landmark_loss_val:.6f}, Total Loss: {total_loss_val:.6f}")
                
                if landmark_loss_val > 100:
                    print(f"  *** WARNING: High landmark loss suggests misalignment! ***")
                
                # Additional parameter monitoring
                current_weights = loss_fn.weights
                pose_vals = pred_coeffs_dict['pose_params'][0].detach().cpu().numpy()
                print(f"  Current reg_global_pose weight: {current_weights['reg_global_pose']}")
                print(f"  Current pose params: {pose_vals}")
                print(f"  Deviation from identity: {np.abs(pose_vals - np.array([1,0,0,0,1,0])).max():.6f}")
                
                # Check if loss function is broken
                if abs(loss_dict['reg_global_pose'].item()) < 1e-8 and current_weights['reg_global_pose'] > 0:
                    print(f"  *** CRITICAL: reg_global_pose loss is 0 but weight is {current_weights['reg_global_pose']} ***")
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Check if model parameters are actually changing
            if i == 0:  # First batch of epoch
                print(f"DEBUG: First 5 encoder FC weights: {encoder.backbone.fc.weight[0, :5].detach().cpu().numpy()}")
                print(f"DEBUG: First 5 encoder FC biases: {encoder.backbone.fc.bias[:5].detach().cpu().numpy()}")
            
        # --- EPOCH-END SNAPSHOT: Visual validation, TensorBoard logging, and detailed console output ---
        current_tensorboard_step = epoch + 1
        loss_total_val = total_loss.item()
        
        # Add to loss history for convergence tracking
        loss_history.append(loss_total_val)
        
        # Analyze training health every few epochs
        if (current_stage_epoch_idx + 1) % 5 == 0 or (current_stage_epoch_idx + 1) == num_epochs_this_stage:
            from src.utils import track_convergence, analyze_gradient_health, monitor_parameter_changes
            
            # Track convergence
            convergence_info = track_convergence(loss_history)
            
            # Analyze gradient health  
            grad_health = analyze_gradient_health(encoder)
            
            # Monitor parameter changes (just encoder backbone vs head)
            param_changes = monitor_parameter_changes(encoder, param_history, 
                                                    ['backbone.fc.weight', 'backbone.fc.bias'])
            
            print(f"\n--- Training Health Report (Epoch {epoch+1}) ---")
            print(f"Convergence: {convergence_info['trend']} | Improvement: {convergence_info['improvement']:.6f}")
            print(f"Gradients: {grad_health['status']} | Total norm: {grad_health['total_norm']:.6f}")
            
            for param_name, change_info in param_changes.items():
                print(f"Param {param_name}: rel_change={change_info['relative_change']:.6f}")
            
            if convergence_info['trend'] == 'plateaued':
                print("*** INFO: Loss has plateaued - consider adjusting learning rate ***")
            if grad_health['status'] != 'healthy':
                print(f"*** WARNING: {grad_health['status']} gradients detected! ***")
        
        # Log the final batch loss for the epoch to the console
        if (current_stage_epoch_idx + 1) % 1 == 0 or (current_stage_epoch_idx + 1) == num_epochs_this_stage:
             print(f"  Epoch {epoch+1}/{total_epochs_all_stages} (Stage Epoch {current_stage_epoch_idx+1}/{num_epochs_this_stage}) | "
                   f"LR: {optimizer.param_groups[0]['lr']:.1e} | "
                   f"Total Loss: {loss_total_val:.4f}")

        writer.add_scalar('Loss/train_total_epoch_last_batch', loss_total_val, current_tensorboard_step)
        for loss_name, loss_value in loss_dict.items():
            if loss_name != 'total':
                val_to_log = loss_value.item() if hasattr(loss_value, 'item') else loss_value
                writer.add_scalar(f'Loss/train_{loss_name}_epoch_last_batch', val_to_log, current_tensorboard_step)
        writer.add_scalar('Hyperparameters/learning_rate_epoch', optimizer.param_groups[0]['lr'], current_tensorboard_step)

        encoder.eval()
        with torch.no_grad():
            # Get a small number of validation samples from the last batch
            num_val_samples = min(4, gt_images.shape[0])
            val_gt_images = gt_images[:num_val_samples]
            val_gt_landmarks_for_vis = gt_landmarks_2d_scaled[:num_val_samples]

            # Run validation forward pass
            val_pred_coeffs_vec = encoder(val_gt_images)
            val_pred_coeffs_dict = deconstruct_flame_coeffs(
                val_pred_coeffs_vec,
                NUM_SHAPE_COEFFS, NUM_EXPRESSION_COEFFS, NUM_GLOBAL_POSE_COEFFS,
                NUM_JAW_POSE_COEFFS, NUM_EYE_POSE_COEFFS, NUM_NECK_POSE_COEFFS,
                NUM_TRANSLATION_COEFFS, NUM_DETAIL_COEFFS
            )

            val_pred_verts, val_pred_landmarks_3d = flame_model(
                shape_params=val_pred_coeffs_dict['shape_params'],
                expression_params=val_pred_coeffs_dict['expression_params'],
                pose_params=val_pred_coeffs_dict['pose_params'],
                jaw_pose_params=val_pred_coeffs_dict['jaw_pose_params'],
                eye_pose_params=val_pred_coeffs_dict['eye_pose_params'],
                neck_pose_params=val_pred_coeffs_dict['neck_pose_params'],
                transl=val_pred_coeffs_dict['transl'],
                use_posedirs=stage_use_posedirs
            )

            # Apply coordinate correction for validation landmarks too
            if USE_COORDINATE_CORRECTION:
                val_pred_landmarks_3d_corrected = apply_coordinate_system_correction(
                    val_pred_landmarks_3d, COORDINATE_CORRECTION_TYPE
                )
            else:
                val_pred_landmarks_3d_corrected = val_pred_landmarks_3d

            # Project landmarks directly for visualization
            image_size_for_projection = (raster_settings.image_size, raster_settings.image_size)
            val_pred_landmarks_2d_model = cameras.transform_points_screen(
                val_pred_landmarks_3d_corrected, image_size=image_size_for_projection
            )[:, :, :2]

            val_generic_vertex_colors = torch.ones_like(val_pred_verts) * 0.7
            val_textures_batch = TexturesVertex(verts_features=val_generic_vertex_colors.to(DEVICE))

            # Use vertices directly for rendering
            val_pred_verts_for_render = val_pred_verts

            val_meshes_batch = Meshes(
                verts=list(val_pred_verts_for_render),
                faces=[flame_model.faces_idx] * val_pred_verts.shape[0],
                textures=val_textures_batch
            )
            val_rendered_images = renderer(val_meshes_batch).permute(0, 3, 1, 2)[:, :3, :, :]

            # Un-normalize images for visualization
            mean_tb = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
            std_tb = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)
            val_gt_images_unnorm_tb = val_gt_images * std_tb + mean_tb

            # Draw landmarks and create image grids for TensorBoard
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
        global_epoch_idx += 1 # Increment global_epoch_idx after each true epoch is completed
    
    # --- STAGE-END CHECKPOINT ---
    stage_model_save_path = f"eidolon_encoder_stage_{stage_idx+1}.pth"
    torch.save(encoder.state_dict(), stage_model_save_path)
    
    # Save training metadata
    metadata = {
        'stage_name': stage_config['name'],
        'final_loss': loss_history[-1] if loss_history else 0,
        'loss_history': loss_history,
        'convergence_info': convergence_info,
        'total_epochs_completed': epoch + 1
    }
    metadata_path = f"training_metadata_stage_{stage_idx+1}.pkl"
    
    import pickle
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"\n--- Stage {stage_idx + 1} ({stage_config['name']}) Finished ---")
    print(f"Final Loss: {loss_history[-1]:.6f}")
    print(f"Loss Trend: {convergence_info.get('trend', 'unknown')}")
    print(f"Saved model checkpoint to: {stage_model_save_path}")
    print(f"Saved training metadata to: {metadata_path}\n")

print("All training stages finished.")

# --- Save the final model name to a file for easy reference ---
with open('latest_model.txt', 'w') as f:
    f.write(stage_model_save_path)


writer.close() # Close the TensorBoard SummaryWriter
