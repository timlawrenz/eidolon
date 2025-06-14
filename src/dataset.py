"""
Defines the FaceDataset class for loading images and pre-computed landmarks
from disk.
"""

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import os

class FaceDataset(Dataset):
    def __init__(self, image_dir, landmark_dir, transform=None, image_extension='.png', landmark_extension='.npy'):
        """
        Args:
            image_dir (str): Directory with all the images.
            landmark_dir (str): Directory with all the pre-computed landmark .npy files.
            transform (callable, optional): Optional transform to be applied on an image.
            image_extension (str): Extension of the image files (e.g., '.png', '.jpg').
            landmark_extension (str): Extension of the landmark files (e.g., '.npy').
        """
        self.image_dir = image_dir
        self.landmark_dir = landmark_dir
        self.image_extension = image_extension
        self.landmark_extension = landmark_extension
        
        # Create a list of base filenames that have both an image and a landmark file
        # Assumes filenames (without extension) are the same for corresponding images and landmarks
        try:
            landmark_basenames = {os.path.splitext(f)[0] for f in os.listdir(landmark_dir) if f.endswith(landmark_extension)}
            image_basenames = {os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.endswith(image_extension)}
        except FileNotFoundError as e:
            print(f"Error: Directory not found. Ensure image_dir ('{image_dir}') and landmark_dir ('{landmark_dir}') exist.")
            raise e
            
        self.valid_files = sorted(list(image_basenames.intersection(landmark_basenames)))
        
        if not self.valid_files:
            print(f"Warning: No matching image/landmark pairs found in {image_dir} and {landmark_dir}.")
            print(f"Searched for images like *{image_extension} and landmarks like *{landmark_extension}.")

        # Standard transformations for ResNet-50.
        # Note: Images are assumed to be 128x128 from ffhq_thumbnails_128.
        # ResNet-50 typically expects 224x224. This resize is crucial.
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)), # Resize to ResNet-50 expected input size
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
        print(f"Found {len(self.valid_files)} matching image/landmark pairs.")

    def __len__(self):
        return len(self.valid_files)

    def __getitem__(self, idx):
        base_name = self.valid_files[idx]
        
        # Load image
        img_path = os.path.join(self.image_dir, base_name + self.image_extension)
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            print(f"Error: Image file not found: {img_path}")
            # Return dummy data or raise error, depending on desired handling
            # For now, let's make it obvious if something goes wrong during iteration
            raise
        
        # Load landmark
        landmark_path = os.path.join(self.landmark_dir, base_name + self.landmark_extension)
        try:
            landmarks = torch.from_numpy(np.load(landmark_path)).float()

            # Scale landmarks from original_dim (e.g., 128) to target_dim (e.g., 224)
            # This should correspond to the image resizing dimensions if landmarks are pixel coordinates.
            original_dim = 128.0  # Assuming landmarks were for 128x128 images
            target_dim = 224.0  # Assuming images are resized to 224x224 by self.transform

            if landmarks.numel() > 0: # Ensure landmarks tensor is not empty
                # It's important to know if self.transform resizes the image.
                # If the image is resized from 128 to 224, landmarks need to be scaled.
                # This scaling assumes the Resize transform is (224,224) as defined in __init__
                scale_factor = target_dim / original_dim
                landmarks = landmarks * scale_factor

        except FileNotFoundError:
            print(f"Error: Landmark file not found: {landmark_path}")
            # Return dummy data or raise error
            raise
        
        if self.transform:
            image = self.transform(image)
            
        return {'image': image, 'gt_landmarks': landmarks}
