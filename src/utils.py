import torch
import torchvision
import matplotlib.pyplot as plt
from tqdm import tqdm # Import tqdm
import numpy as np
import os

def save_validation_images(gt_images, rendered_images, gt_landmarks, pred_landmarks, save_path_prefix, num_images=4):
    """
    Saves a grid of ground truth vs. prediction images with landmarks.

    Args:
        gt_images (torch.Tensor): Batch of ground truth images (B, C, H, W), normalized.
        rendered_images (torch.Tensor): Batch of rendered images (B, C, H, W).
        gt_landmarks (torch.Tensor): Batch of ground truth 2D landmarks (B, N_landmarks, 2).
        pred_landmarks (torch.Tensor): Batch of predicted 2D landmarks (B, N_landmarks, 2).
        save_path_prefix (str): Base path and filename prefix for saving images (e.g., "outputs/epoch_1_step_500").
                                Each sample will be saved as prefix_sample_idx.png.
        num_images (int): Number of images from the batch to save.
    """
    # Ensure we don't try to save more images than we have
    num_images = min(num_images, gt_images.shape[0])

    # Select subset and move to CPU
    gt_images_cpu = gt_images[:num_images].cpu()
    rendered_images_cpu = rendered_images[:num_images].cpu()
    gt_landmarks_cpu = gt_landmarks[:num_images].cpu().numpy() # (num_images, 68, 2)
    pred_landmarks_cpu = pred_landmarks[:num_images].cpu().numpy() # (num_images, 68, 2)

    # Inverse normalize gt_images for display if they are normalized
    # Assuming standard ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406], device=gt_images_cpu.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=gt_images_cpu.device).view(1, 3, 1, 1)
    gt_images_unnorm = gt_images_cpu * std + mean
    # gt_images_unnorm are now (num_images, C, H, W) in range [0,1]
    # For plotting, permute to (num_images, H, W, C)
    gt_images_display = gt_images_unnorm.permute(0, 2, 3, 1).numpy().clip(0, 1)

    # Scale ground truth landmarks from original image size (e.g., 128x128) to display size (e.g., 224x224)
    # The gt_images were resized from 128 (assumed original for landmarks) to 224 by FaceDataset.
    original_landmark_img_size = 128.0 # The size of images landmarks were detected on
    display_img_size = float(gt_images_display.shape[1]) # Should be 224
    
    if display_img_size != original_landmark_img_size:
        scale_factor = display_img_size / original_landmark_img_size
        gt_landmarks_cpu_scaled = gt_landmarks_cpu * scale_factor
    else:
        gt_landmarks_cpu_scaled = gt_landmarks_cpu


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
