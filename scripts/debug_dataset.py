import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np

# Attempt to import FaceDataset
try:
    from src.dataset import FaceDataset
except ImportError:
    print("Error: Could not import FaceDataset from src.dataset. Make sure the class is defined and accessible.")
    # As a fallback for testing the script structure, define a dummy FaceDataset
    class FaceDataset:
        def __init__(self, image_dir, landmark_dir, transform=None):
            self.image_dir = image_dir
            self.landmark_dir = landmark_dir
            self.transform = transform
            self.image_files = []
            if not os.path.isdir(image_dir):
                raise FileNotFoundError(f"Image directory not found: {image_dir}")
            if not os.path.isdir(landmark_dir):
                raise FileNotFoundError(f"Landmark directory not found: {landmark_dir}")

            # Dummy data for testing script logic if actual data/class is missing
            # In a real scenario, this would load actual file paths
            if "debug" in image_dir: # Minimal data for dummy
                self.image_files = ["dummy.png"]
                # Create dummy image and landmark file for the script to run
                if not os.path.exists(os.path.join(image_dir, "dummy.png")):
                    try:
                        Image.new('RGB', (128, 128), color = 'red').save(os.path.join(image_dir, "dummy.png"))
                    except Exception as e:
                        print(f"Could not create dummy image: {e}")
                if not os.path.exists(os.path.join(landmark_dir, "dummy.npy")):
                    try:
                        np.save(os.path.join(landmark_dir, "dummy.npy"), np.random.rand(68, 2) * 128)
                    except Exception as e:
                        print(f"Could not create dummy landmarks: {e}")


        def __len__(self):
            return len(self.image_files)

        def __getitem__(self, idx):
            if not self.image_files: # No data
                return {
                    'image': torch.empty(0), # empty tensor
                    'gt_landmarks': torch.empty(0) # empty tensor
                }

            # Dummy sample data (replace with actual data loading and preprocessing)
            # This part assumes the real FaceDataset would load an image and landmarks
            # and apply transformations.
            try:
                img_path = os.path.join(self.image_dir, self.image_files[idx])
                landmarks_path = os.path.join(self.landmark_dir, self.image_files[idx].replace('.png', '.npy'))

                image = Image.open(img_path).convert('RGB')
                landmarks = np.load(landmarks_path)
            except FileNotFoundError as e:
                 # This might happen if dummy files were not created successfully
                 print(f"Error loading dummy files: {e}")
                 return {
                    'image': torch.empty(0),
                    'gt_landmarks': torch.empty(0)
                }


            # Apply standard transformations if none are provided (for dummy dataset)
            current_transform = self.transform
            if current_transform is None:
                current_transform = transforms.Compose([
                    transforms.Resize((128, 128)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

            image_tensor = current_transform(image)

            # Ensure landmarks are a tensor
            gt_landmarks = torch.tensor(landmarks, dtype=torch.float32)

            return {'image': image_tensor, 'gt_landmarks': gt_landmarks}

def main():
    image_dir = 'data/ffhq_images_128_debug'
    landmark_dir = 'data/ffhq_landmarks_128_debug'
    output_plot_path = 'debug_dataset_sample.png'

    # Create dummy directories if they don't exist, so the script can run
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(landmark_dir, exist_ok=True)

    print(f"Attempting to load dataset with:")
    print(f"  Image directory: {image_dir}")
    print(f"  Landmark directory: {landmark_dir}")

    try:
        # Assuming default transform for FaceDataset, or it handles None
        dataset = FaceDataset(image_dir=image_dir, landmark_dir=landmark_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the dataset directories exist and contain data.")
        return
    except Exception as e:
        print(f"An unexpected error occurred during dataset instantiation: {e}")
        return

    if len(dataset) == 0:
        print("Dataset loaded, but it is empty. No data to process or visualize.")
        # Check if dummy files were expected but not created/found
        if "debug" in image_dir and not os.listdir(image_dir):
             print(f"The directory {image_dir} is empty. The dummy FaceDataset expects at least one file if 'debug' is in path.")
        return

    print(f"Dataset loaded successfully. Number of samples: {len(dataset)}")

    # Load the first sample
    try:
        sample = dataset[0]
        if sample['image'].numel() == 0 or sample['gt_landmarks'].numel() == 0: # numel() checks if tensor is empty
            print("The first sample is empty. Cannot proceed with visualization.")
            return
    except Exception as e:
        print(f"Error loading sample from dataset: {e}")
        return

    image_tensor = sample['image']
    gt_landmarks = sample['gt_landmarks']

    print("\n--- Image Tensor ---")
    print(f"Shape: {image_tensor.shape}")
    print(f"Dtype: {image_tensor.dtype}")
    if image_tensor.numel() > 0:
        print(f"Min: {image_tensor.min().item():.4f}")
        print(f"Mean: {image_tensor.mean().item():.4f}")
        print(f"Max: {image_tensor.max().item():.4f}")
    else:
        print("Image tensor is empty.")

    print("\n--- Ground Truth Landmarks ---")
    print(f"Shape: {gt_landmarks.shape}")
    print(f"Dtype: {gt_landmarks.dtype}")
    if gt_landmarks.numel() > 0:
        print(f"Min: {gt_landmarks.min().item():.4f}")
        # Calculate mean across all elements for simplicity, or specify axis if needed
        print(f"Mean: {gt_landmarks.mean().item():.4f}")
        print(f"Max: {gt_landmarks.max().item():.4f}")
    else:
        print("Landmarks tensor is empty.")

    # Visualization
    try:
        # Unnormalize the image tensor
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        unnormalized_image_tensor = image_tensor * std + mean
        unnormalized_image_tensor = torch.clamp(unnormalized_image_tensor, 0, 1)

        # Convert to PIL Image
        to_pil = transforms.ToPILImage()
        pil_image = to_pil(unnormalized_image_tensor)

        # Plotting
        plt.figure(figsize=(6, 6))
        plt.imshow(pil_image)

        # Overlay landmarks
        # Assuming landmarks are (x, y). If (x, y, z), take only x, y.
        landmarks_to_plot = gt_landmarks.numpy() # Convert to numpy array
        if landmarks_to_plot.ndim == 2 and landmarks_to_plot.shape[1] >= 2:
            plt.scatter(landmarks_to_plot[:, 0], landmarks_to_plot[:, 1], s=10, c='red', marker='.')
        elif landmarks_to_plot.ndim == 1 and landmarks_to_plot.shape[0] >= 2: # Single landmark
             plt.scatter(landmarks_to_plot[0], landmarks_to_plot[1], s=10, c='red', marker='.')
        else:
            print("Landmarks shape is not suitable for plotting (expected [N, 2] or [N,3]).")

        plt.title("Sample Image with Ground Truth Landmarks")
        plt.xlabel("X-coordinate")
        plt.ylabel("Y-coordinate")
        # Set axes limits based on image dimensions (e.g., 0 to H, 0 to W)
        # Image shape is C, H, W. pil_image.size is W, H
        plt.xlim(0, pil_image.width)
        plt.ylim(pil_image.height, 0) # Invert Y axis for typical image display
        plt.gca().set_aspect('equal', adjustable='box') # Ensure aspect ratio is correct
        plt.savefig(output_plot_path)
        print(f"\nPlot saved to {output_plot_path}")

    except Exception as e:
        print(f"An error occurred during visualization: {e}")

if __name__ == '__main__':
    main()
