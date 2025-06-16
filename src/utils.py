import torch
import torchvision
import matplotlib
matplotlib.use('Agg') # Set non-interactive backend BEFORE importing pyplot
import matplotlib.pyplot as plt
from tqdm import tqdm # Import tqdm
import numpy as np
import os

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
