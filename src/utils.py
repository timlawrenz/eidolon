import torch
import torchvision
import matplotlib
matplotlib.use('Agg') # Set non-interactive backend BEFORE importing pyplot
import matplotlib.pyplot as plt
from tqdm import tqdm # Import tqdm
import numpy as np
import os


def validate_coordinate_system_consistency(flame_model, cameras, device='cuda', title="Coordinate System Validation"):
    """
    Validates coordinate system consistency between FLAME, camera, and expected outputs.
    
    Args:
        flame_model: FLAME model instance
        cameras: PyTorch3D camera object
        device: Device to run validation on
        title: Title for the validation report
        
    Returns:
        str: Multi-line validation report
    """
    report = f"{title}:\n"
    report += "=" * 70 + "\n"
    
    with torch.no_grad():
        # Create a test face with known parameters
        batch_size = 1
        n_shape = 100 if hasattr(flame_model, 'shapedirs') else 0
        n_exp = 0  # Typically disabled
        
        # Neutral parameters
        shape_params = torch.zeros(batch_size, n_shape, device=device)
        expression_params = torch.zeros(batch_size, n_exp, device=device)
        
        # Identity pose (should result in frontal face)
        pose_params = torch.zeros(batch_size, 6, device=device)
        pose_params[0, 0] = 1.0  # First column of rotation matrix
        pose_params[0, 4] = 1.0  # Second column of rotation matrix
        
        jaw_pose_params = torch.zeros(batch_size, 3, device=device)
        eye_pose_params = torch.zeros(batch_size, 6, device=device)
        neck_pose_params = torch.zeros(batch_size, 3, device=device)
        transl = torch.zeros(batch_size, 3, device=device)
        
        # Run FLAME with neutral parameters
        try:
            pred_verts, pred_landmarks_3d = flame_model(
                shape_params=shape_params,
                expression_params=expression_params,
                pose_params=pose_params,
                jaw_pose_params=jaw_pose_params,
                eye_pose_params=eye_pose_params,
                neck_pose_params=neck_pose_params,
                transl=transl
            )
            
            report += "FLAME Forward Pass: SUCCESS\n"
            
            # Analyze 3D output
            verts_np = pred_verts[0].cpu().numpy()
            lmks_3d_np = pred_landmarks_3d[0].cpu().numpy()
            
            report += f"\nNeutral Face Analysis:\n"
            report += f"  Vertices shape: {pred_verts.shape}\n"
            report += f"  Landmarks shape: {pred_landmarks_3d.shape}\n"
            
            # Check vertex distribution (should be roughly centered around origin after centering)
            verts_center = verts_np.mean(axis=0)
            verts_extent = verts_np.max(axis=0) - verts_np.min(axis=0)
            report += f"  Vertex center: [{verts_center[0]:.4f}, {verts_center[1]:.4f}, {verts_center[2]:.4f}]\n"
            report += f"  Vertex extent: [{verts_extent[0]:.4f}, {verts_extent[1]:.4f}, {verts_extent[2]:.4f}]\n"
            
            # Check landmark distribution
            lmks_center = lmks_3d_np.mean(axis=0)
            lmks_extent = lmks_3d_np.max(axis=0) - lmks_3d_np.min(axis=0)
            report += f"  Landmark center: [{lmks_center[0]:.4f}, {lmks_center[1]:.4f}, {lmks_center[2]:.4f}]\n"
            report += f"  Landmark extent: [{lmks_extent[0]:.4f}, {lmks_extent[1]:.4f}, {lmks_extent[2]:.4f}]\n"
            
            # Test camera projection
            image_size = (224, 224)
            landmarks_2d_proj = cameras.transform_points_screen(pred_landmarks_3d, image_size=image_size)[:, :, :2]
            lmks_2d_np = landmarks_2d_proj[0].cpu().numpy()
            
            report += f"\nCamera Projection Analysis:\n"
            report += f"  Camera type: {type(cameras).__name__}\n"
            
            # Check 2D landmark distribution
            lmks_2d_center = lmks_2d_np.mean(axis=0)
            lmks_2d_extent = lmks_2d_np.max(axis=0) - lmks_2d_np.min(axis=0)
            report += f"  2D Landmark center: [{lmks_2d_center[0]:.2f}, {lmks_2d_center[1]:.2f}]\n"
            report += f"  2D Landmark extent: [{lmks_2d_extent[0]:.2f}, {lmks_2d_extent[1]:.2f}]\n"
            
            # Expected: landmarks should be roughly centered in image and span reasonable area
            expected_center = np.array([112, 112])  # Center of 224x224 image
            center_offset = np.abs(lmks_2d_center - expected_center)
            
            if np.any(center_offset > 50):
                report += f"  *** WARNING: Landmarks significantly off-center! Offset: {center_offset} ***\n"
            
            if lmks_2d_extent[0] < 30 or lmks_2d_extent[1] < 30:
                report += f"  *** WARNING: Landmarks very small in image! ***\n"
            elif lmks_2d_extent[0] > 200 or lmks_2d_extent[1] > 200:
                report += f"  *** WARNING: Landmarks extend beyond reasonable image area! ***\n"
            
            # Check for landmarks outside image bounds
            out_of_bounds = np.sum((lmks_2d_np < 0) | (lmks_2d_np > 224))
            if out_of_bounds > 0:
                report += f"  *** WARNING: {out_of_bounds} landmarks outside image bounds! ***\n"
            
            # Test with extreme pose to see if system behaves reasonably
            extreme_pose_params = pose_params.clone()
            extreme_pose_params[0, 1] = 0.5  # Rotate around Y axis
            
            extreme_verts, extreme_landmarks_3d = flame_model(
                shape_params=shape_params,
                expression_params=expression_params,
                pose_params=extreme_pose_params,
                jaw_pose_params=jaw_pose_params,
                eye_pose_params=eye_pose_params,
                neck_pose_params=neck_pose_params,
                transl=transl
            )
            
            extreme_landmarks_2d = cameras.transform_points_screen(extreme_landmarks_3d, image_size=image_size)[:, :, :2]
            extreme_lmks_2d_np = extreme_landmarks_2d[0].cpu().numpy()
            
            # Check if rotation changes landmark positions reasonably
            movement = np.abs(extreme_lmks_2d_np - lmks_2d_np).mean()
            report += f"\nPose Sensitivity Test:\n"
            report += f"  Average landmark movement with rotation: {movement:.2f} pixels\n"
            
            if movement < 5:
                report += f"  *** WARNING: Pose changes have little effect on landmarks! ***\n"
            elif movement > 100:
                report += f"  *** WARNING: Small pose changes cause extreme landmark movement! ***\n"
                
        except Exception as e:
            report += f"FLAME Forward Pass: FAILED - {str(e)}\n"
            import traceback
            report += f"Traceback:\n{traceback.format_exc()}\n"
    
    report += "=" * 70 + "\n"
    return report

def save_obj(filepath, vertices, faces=None):
    """
    Saves a 3D mesh to an .obj file.

    Args:
        filepath (str): Path to save the .obj file.
        vertices (torch.Tensor or np.ndarray): Vertices of shape (N, 3).
        faces (torch.Tensor or np.ndarray, optional): Faces of shape (F, 3). Indices are 0-based. Defaults to None.
    """
    assert vertices.ndim == 2 and vertices.shape[1] == 3, "Vertices must be of shape (N, 3)"
    if hasattr(vertices, 'cpu'): # Check if it's a torch.Tensor
        vertices = vertices.detach().cpu().numpy()

    if faces is not None:
        assert faces.ndim == 2 and faces.shape[1] == 3, "Faces must be of shape (F, 3)"
        if hasattr(faces, 'cpu'): # Check if it's a torch.Tensor
            faces = faces.detach().cpu().numpy()
        assert np.issubdtype(faces.dtype, np.integer), f"Faces dtype must be integer, got {faces.dtype}"

    with open(filepath, 'w') as f:
        for v_idx in range(vertices.shape[0]):
            v = vertices[v_idx]
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        if faces is not None:
            # .obj files are 1-indexed
            for face_idx in range(faces.shape[0]):
                face = faces[face_idx]
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    # Use tqdm.write to avoid interfering with progress bars if used during training
    tqdm.write(f"Saved mesh to {filepath}")


def deconstruct_flame_coeffs(pred_coeffs_vec,
                               num_shape_coeffs, num_expression_coeffs, num_global_pose_coeffs,
                               num_jaw_pose_coeffs, num_eye_pose_coeffs, num_neck_pose_coeffs,
                               num_translation_coeffs, num_detail_coeffs):
    """
    Deconstructs a flat FLAME coefficient vector into a dictionary of named parameters.

    Args:
        pred_coeffs_vec (torch.Tensor): A tensor of shape (B, N_coeffs) or (N_coeffs,).
        num_..._coeffs (int): The number of coefficients for each parameter type.

    Returns:
        dict: A dictionary mapping parameter names to their corresponding tensor slices.
    """
    if pred_coeffs_vec.ndim == 1:
        pred_coeffs_vec = pred_coeffs_vec.unsqueeze(0)  # Add batch dimension if missing

    coeffs_dict = {}
    current_idx = 0

    # Shape
    coeffs_dict['shape_params'] = pred_coeffs_vec[:, current_idx:current_idx + num_shape_coeffs]
    current_idx += num_shape_coeffs

    # Expression
    coeffs_dict['expression_params'] = pred_coeffs_vec[:, current_idx:current_idx + num_expression_coeffs]
    current_idx += num_expression_coeffs

    # Global Pose
    coeffs_dict['pose_params'] = pred_coeffs_vec[:, current_idx:current_idx + num_global_pose_coeffs]
    current_idx += num_global_pose_coeffs

    # Jaw Pose
    coeffs_dict['jaw_pose_params'] = pred_coeffs_vec[:, current_idx:current_idx + num_jaw_pose_coeffs]
    current_idx += num_jaw_pose_coeffs

    # Eye Pose
    coeffs_dict['eye_pose_params'] = pred_coeffs_vec[:, current_idx:current_idx + num_eye_pose_coeffs]
    current_idx += num_eye_pose_coeffs
    
    # Neck Pose
    coeffs_dict['neck_pose_params'] = pred_coeffs_vec[:, current_idx:current_idx + num_neck_pose_coeffs]
    current_idx += num_neck_pose_coeffs

    # Translation
    coeffs_dict['transl'] = pred_coeffs_vec[:, current_idx:current_idx + num_translation_coeffs]
    current_idx += num_translation_coeffs
    
    # Detail (e.g., for texture, lighting - not used by FLAME geometry but might be predicted by encoder)
    coeffs_dict['detail_params'] = pred_coeffs_vec[:, current_idx:current_idx + num_detail_coeffs]

    return coeffs_dict


def save_validation_images(gt_images_display_unnormalized, rendered_images, 
                           gt_landmarks_for_display_scaled, pred_landmarks_for_display, 
                           save_path_prefix, num_images=4):
    """
    Saves a grid of ground truth vs. prediction images with landmarks.

    Args:
        gt_images_display_unnormalized (torch.Tensor): Batch of ground truth images (B, C, H, W), already UNNORMALIZED.
        rendered_images (torch.Tensor): Batch of rendered images (B, C, H, W).
        gt_landmarks_for_display_scaled (torch.Tensor): Batch of ground truth 2D landmarks (B, N_landmarks, 2), ALREADY SCALED to display size.
        pred_landmarks_for_display (torch.Tensor): Batch of predicted 2D landmarks (B, N_landmarks, 2), already in display size.
        save_path_prefix (str): Base path and filename prefix for saving images (e.g., "outputs/epoch_1_step_500").
                                Each sample will be saved as prefix_sample_idx.png.
        num_images (int): Number of images from the batch to save.
    """
    # Ensure we don't try to save more images than we have
    num_images = min(num_images, gt_images_display_unnormalized.shape[0])

    # Select subset and move to CPU
    gt_images_cpu = gt_images_display_unnormalized[:num_images].cpu()
    rendered_images_cpu = rendered_images[:num_images].cpu()
    # gt_landmarks are already scaled, directly use them after moving to CPU
    gt_landmarks_cpu_scaled = gt_landmarks_for_display_scaled[:num_images].cpu().numpy() 
    pred_landmarks_cpu = pred_landmarks_for_display[:num_images].cpu().numpy()

    # gt_images_cpu are already unnormalized.
    # For plotting, permute to (num_images, H, W, C)
    gt_images_display = gt_images_cpu.permute(0, 2, 3, 1).numpy().clip(0, 1)

    # Ground truth landmarks are already scaled (gt_landmarks_cpu_scaled).
    # No further scaling needed here.

    # Permute rendered images for display
    rendered_images_display = rendered_images_cpu.permute(0, 2, 3, 1).numpy().clip(0, 1) # (num_images, H, W, C)


    for i in range(num_images):
        fig, axs = plt.subplots(1, 2, figsize=(10, 5)) # Create a new figure for each sample
        
        # Ground Truth
        axs[0].imshow(gt_images_display[i])
        axs[0].scatter(gt_landmarks_cpu_scaled[i, :, 0], gt_landmarks_cpu_scaled[i, :, 1], s=10, c='r', marker='.')
        axs[0].set_title("Ground Truth")
        axs[0].axis('off')

        # Prediction
        axs[1].imshow(rendered_images_display[i])
        axs[1].scatter(pred_landmarks_cpu[i, :, 0], pred_landmarks_cpu[i, :, 1], s=10, c='b', marker='.')
        axs[1].set_title("Prediction")
        axs[1].axis('off')

        # Save individual comparison
        img_save_path = f"{save_path_prefix}_sample_{i}.png"
        plt.savefig(img_save_path)
        plt.close(fig) # Close the figure to free memory

    tqdm.write(f"Saved {num_images} validation samples to {os.path.dirname(save_path_prefix)}")


def draw_landmarks_on_images_tensor(images_batch_float, landmarks_batch, color='red', radius=2):
    """
    Draws landmarks on a batch of image tensors.

    Args:
        images_batch_float (torch.Tensor): Batch of images (B, C, H, W), float, range [0, 1].
        landmarks_batch (torch.Tensor): Batch of landmarks (B, N_landmarks, 2), float.
        color (str): Color for the landmarks.
        radius (int): Radius of the landmark points.

    Returns:
        torch.Tensor: Batch of images (B, C, H, W) with landmarks drawn, float, range [0, 1].
    """
    images_batch_uint8 = (images_batch_float.clone() * 255).to(torch.uint8) # Convert to uint8 [0,255]
    
    # Ensure landmarks are on the same device as images and are integer type for drawing
    landmarks_batch_int = landmarks_batch.round().to(dtype=torch.int64, device=images_batch_uint8.device)

    images_with_landmarks_list = []
    for i in range(images_batch_uint8.shape[0]):
        img_uint8 = images_batch_uint8[i] # (C, H, W)
        lms_int = landmarks_batch_int[i]   # (N_landmarks, 2)
        
        # draw_keypoints expects keypoints in (K, 2) format, and landmarks_batch_int[i] is already in this format.
        # It also expects a list of such tensors if drawing on multiple instances within a single image,
        # but here we draw one set of landmarks per image in the batch.
        # So, we provide landmarks_batch_int[i].unsqueeze(0) to make it (1, K, 2) for one instance.
        img_with_lms = torchvision.utils.draw_keypoints(
            image=img_uint8, 
            keypoints=lms_int.unsqueeze(0), # Shape (1, N_landmarks, 2)
            colors=color, 
            radius=radius
        )
        images_with_landmarks_list.append(img_with_lms)
    
    images_with_landmarks_batch_uint8 = torch.stack(images_with_landmarks_list)
    images_with_landmarks_batch_float = images_with_landmarks_batch_uint8.float() / 255.0 # Convert back to float [0,1]
    return images_with_landmarks_batch_float


def plot_landmarks_ascii(landmarks_2d_batch, original_img_width, original_img_height, grid_width=40, grid_height=20, title="Landmarks ASCII Plot"):
    """
    Generates an ASCII representation of 2D landmarks for the first sample in a batch.

    Args:
        landmarks_2d_batch (torch.Tensor): Batch of 2D landmarks (B, N_landmarks, 2).
                                           Coordinates are assumed to be in original image space.
        original_img_width (float): Width of the original image space for landmarks.
        original_img_height (float): Height of the original image space for landmarks.
        grid_width (int): Width of the ASCII character grid.
        grid_height (int): Height of the ASCII character grid.
        title (str): Title for the plot.

    Returns:
        str: A multi-line string representing the ASCII plot.
    """
    if landmarks_2d_batch.numel() == 0:
        return f"{title}:\nNo landmarks to plot.\n"

    landmarks_sample = landmarks_2d_batch[0].cpu().numpy() # Take the first sample (N_landmarks, 2)

    grid = [[' ' for _ in range(grid_width)] for _ in range(grid_height)]

    for x, y in landmarks_sample:
        # Normalize coordinates to grid dimensions
        # Clamp to ensure they are within image bounds before scaling
        norm_x = np.clip(x / original_img_width, 0.0, 1.0)
        norm_y = np.clip(y / original_img_height, 0.0, 1.0)
        
        grid_x = int(norm_x * (grid_width - 1))
        grid_y = int(norm_y * (grid_height - 1))
        
        if 0 <= grid_y < grid_height and 0 <= grid_x < grid_width:
            grid[grid_y][grid_x] = '*'

    output_str = f"{title} (First Sample, {original_img_width}x{original_img_height} space):\n"
    output_str += "+" + "-" * grid_width + "+\n"
    for row in grid:
        output_str += "|" + "".join(row) + "|\n"
    output_str += "+" + "-" * grid_width + "+\n"
    return output_str


def plot_loss_components_ascii(loss_dict, width=60, title="Loss Components"):
    """
    Creates an ASCII bar chart of loss components for debugging.
    
    Args:
        loss_dict (dict): Dictionary of loss components with their values
        width (int): Width of the ASCII chart
        title (str): Title for the chart
        
    Returns:
        str: Multi-line string representing the loss components chart
    """
    if not loss_dict:
        return f"{title}:\nNo loss components to display.\n"
    
    # Extract loss values and convert to float
    loss_items = []
    for key, value in loss_dict.items():
        if key != 'total':  # Skip total to focus on components
            val = value.item() if hasattr(value, 'item') else float(value)
            loss_items.append((key, val))
    
    if not loss_items:
        return f"{title}:\nNo valid loss components found.\n"
    
    # Sort by value for better visualization
    loss_items.sort(key=lambda x: x[1], reverse=True)
    
    # Find max value for scaling
    max_val = max(val for _, val in loss_items) if loss_items else 1.0
    if max_val == 0:
        max_val = 1.0
    
    output_str = f"{title}:\n"
    output_str += "=" * width + "\n"
    
    for name, value in loss_items:
        # Scale bar length
        bar_length = int((value / max_val) * (width - 20)) if max_val > 0 else 0
        bar = "#" * bar_length
        output_str += f"{name:>15}: {bar:<{width-20}} {value:.6f}\n"
    
    output_str += "=" * width + "\n"
    return output_str


def plot_pose_parameters_ascii(pose_dict, width=60, title="Pose Parameters"):
    """
    Creates an ASCII visualization of pose parameters for debugging.
    
    Args:
        pose_dict (dict): Dictionary containing pose parameters
        width (int): Width of the ASCII display
        title (str): Title for the visualization
        
    Returns:
        str: Multi-line string representing pose parameters
    """
    if not pose_dict:
        return f"{title}:\nNo pose parameters to display.\n"
    
    output_str = f"{title}:\n"
    output_str += "=" * width + "\n"
    
    for param_name, param_tensor in pose_dict.items():
        if param_tensor is not None and param_tensor.numel() > 0:
            # Take first sample from batch
            param_vals = param_tensor[0].detach().cpu().numpy() if param_tensor.dim() > 1 else param_tensor.detach().cpu().numpy()
            
            output_str += f"\n{param_name}:\n"
            output_str += f"  Shape: {param_tensor.shape}\n"
            output_str += f"  Values: {param_vals}\n"
            output_str += f"  Range: [{param_vals.min():.4f}, {param_vals.max():.4f}]\n"
            output_str += f"  Mean: {param_vals.mean():.4f}, Std: {param_vals.std():.4f}\n"
            
            # Check for extreme values
            if np.abs(param_vals).max() > 3.0:  # Arbitrary threshold for "extreme"
                output_str += f"  *** WARNING: EXTREME VALUES DETECTED! ***\n"
    
    output_str += "=" * width + "\n"
    return output_str


def validate_landmark_data(gt_landmarks, pred_landmarks, image_size=224, title="Landmark Validation"):
    """
    Validates landmark data for common issues that could cause training problems.
    
    Args:
        gt_landmarks (torch.Tensor): Ground truth landmarks (B, N, 2)
        pred_landmarks (torch.Tensor): Predicted landmarks (B, N, 2)  
        image_size (int): Expected image size for landmarks
        title (str): Title for the validation report
        
    Returns:
        str: Multi-line validation report
    """
    report = f"{title}:\n"
    report += "=" * 60 + "\n"
    
    # Check shapes
    report += f"GT Landmarks shape: {gt_landmarks.shape}\n"
    report += f"Pred Landmarks shape: {pred_landmarks.shape}\n"
    
    if gt_landmarks.shape != pred_landmarks.shape:
        report += "*** ERROR: Shape mismatch between GT and predicted landmarks! ***\n"
    
    # Check value ranges for first sample
    if gt_landmarks.numel() > 0:
        gt_sample = gt_landmarks[0].cpu().numpy()
        report += f"\nGT Landmarks (Sample 0):\n"
        report += f"  X range: [{gt_sample[:, 0].min():.2f}, {gt_sample[:, 0].max():.2f}]\n"
        report += f"  Y range: [{gt_sample[:, 1].min():.2f}, {gt_sample[:, 1].max():.2f}]\n"
        
        # Check if landmarks are outside image bounds
        out_of_bounds_x = np.sum((gt_sample[:, 0] < 0) | (gt_sample[:, 0] > image_size))
        out_of_bounds_y = np.sum((gt_sample[:, 1] < 0) | (gt_sample[:, 1] > image_size))
        report += f"  Out of bounds: X={out_of_bounds_x}, Y={out_of_bounds_y}\n"
        
        if out_of_bounds_x > 0 or out_of_bounds_y > 0:
            report += "  *** WARNING: GT landmarks outside image bounds! ***\n"
    
    if pred_landmarks.numel() > 0:
        pred_sample = pred_landmarks[0].detach().cpu().numpy()
        report += f"\nPred Landmarks (Sample 0):\n"
        report += f"  X range: [{pred_sample[:, 0].min():.2f}, {pred_sample[:, 0].max():.2f}]\n"
        report += f"  Y range: [{pred_sample[:, 1].min():.2f}, {pred_sample[:, 1].max():.2f}]\n"
        
        # Check for clustering (all landmarks in small area - like on one ear)
        x_spread = pred_sample[:, 0].max() - pred_sample[:, 0].min()
        y_spread = pred_sample[:, 1].max() - pred_sample[:, 1].min()
        report += f"  Spread: X={x_spread:.2f}, Y={y_spread:.2f}\n"
        
        if x_spread < 20 or y_spread < 20:  # Arbitrary threshold
            report += "  *** WARNING: Landmarks clustered in small area! ***\n"
        
        # Check for extreme clustering (like all on one ear)
        center_x, center_y = pred_sample[:, 0].mean(), pred_sample[:, 1].mean()
        distances = np.sqrt((pred_sample[:, 0] - center_x)**2 + (pred_sample[:, 1] - center_y)**2)
        max_distance = distances.max()
        report += f"  Max distance from centroid: {max_distance:.2f}\n"
        
        if max_distance < 10:  # Very clustered
            report += "  *** CRITICAL: All landmarks extremely clustered! ***\n"
    
    report += "=" * 60 + "\n"
    return report


def validate_camera_projection(landmarks_3d, camera, image_size=(224, 224), title="Camera Projection Validation"):
    """
    Validates camera projection behavior to detect coordinate system issues.
    
    Args:
        landmarks_3d (torch.Tensor): 3D landmarks (B, N, 3)
        camera: PyTorch3D camera object
        image_size (tuple): Expected output image size
        title (str): Title for validation report
        
    Returns:
        str: Multi-line validation report
    """
    report = f"{title}:\n"
    report += "=" * 60 + "\n"
    
    if landmarks_3d.numel() == 0:
        report += "No 3D landmarks to validate.\n"
        return report
    
    # Take first sample
    lmks_3d_sample = landmarks_3d[0]  # Shape: (N, 3)
    
    report += f"3D Landmarks shape: {landmarks_3d.shape}\n"
    report += f"Camera type: {type(camera).__name__}\n"
    
    # Analyze 3D landmark distribution
    lmks_3d_np = lmks_3d_sample.detach().cpu().numpy()
    for i, axis in enumerate(['X', 'Y', 'Z']):
        values = lmks_3d_np[:, i]
        report += f"3D {axis}: range=[{values.min():.4f}, {values.max():.4f}], mean={values.mean():.4f}\n"
    
    # Project to 2D
    landmarks_2d_proj = camera.transform_points_screen(landmarks_3d, image_size=image_size)[:, :, :2]
    lmks_2d_np = landmarks_2d_proj[0].detach().cpu().numpy()
    
    # Analyze 2D projection
    for i, axis in enumerate(['X', 'Y']):
        values = lmks_2d_np[:, i]
        report += f"2D {axis}: range=[{values.min():.2f}, {values.max():.2f}], mean={values.mean():.2f}\n"
    
    # Check for projection issues
    out_of_bounds = np.sum((lmks_2d_np < 0) | (lmks_2d_np > max(image_size)))
    if out_of_bounds > 0:
        report += f"*** WARNING: {out_of_bounds} landmarks projected outside image bounds! ***\n"
    
    # Check for extreme clustering after projection
    x_spread = lmks_2d_np[:, 0].max() - lmks_2d_np[:, 0].min()
    y_spread = lmks_2d_np[:, 1].max() - lmks_2d_np[:, 1].min()
    if x_spread < 10 or y_spread < 10:
        report += f"*** WARNING: Projected landmarks very clustered (spread: {x_spread:.1f}x{y_spread:.1f})! ***\n"
    
    report += "=" * 60 + "\n"
    return report
