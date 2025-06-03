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
