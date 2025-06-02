"""
Defines the FaceDataset class for loading and transforming face images
directly from a Hugging Face dataset.

This module provides a PyTorch Dataset implementation that wraps a Hugging Face
dataset, applying necessary transformations for model input. It's designed
to be used with PyTorch's DataLoader for efficient batching.
"""

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from datasets import load_dataset # Import load_dataset

class FaceDataset(Dataset):
    def __init__(self, hf_dataset_name, hf_dataset_split='train', transform=None):
        """
        Args:
            hf_dataset_name (str): Name of the dataset on Hugging Face Hub (e.g., "nuwandaa/ffhq128").
            hf_dataset_split (str, optional): The split to use (e.g., "train"). Defaults to 'train'.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        print(f"Loading Hugging Face dataset: {hf_dataset_name}, split: {hf_dataset_split}")
        self.hf_dataset = load_dataset(hf_dataset_name, split=hf_dataset_split)
        print(f"Dataset loaded. Found {len(self.hf_dataset)} images.")
        
        # Define a standard set of transformations if none are provided
        if transform is None:
            self.transform = transforms.Compose([
                # Images from "nuwandaa/ffhq128" are already 128x128.
                # ResNet-50 expects 224x224 inputs.
                transforms.Resize((224, 224)), 
                transforms.ToTensor(),
                # Normalize with ImageNet's mean and std
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        # Access the image from the Hugging Face dataset entry
        # The key for the image column is typically 'image'
        image = self.hf_dataset[idx]['image']
        
        # Ensure image is PIL Image and in RGB format
        if not isinstance(image, Image.Image):
            # This case might not be hit if hf_dataset already yields PIL images
            image = Image.fromarray(image) # Example if it were a numpy array
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        # The image itself is the ground truth for the pixel and perceptual losses.
        # Later, we will also return ground-truth landmarks if available from the dataset.
        sample = {'image': image}
        
        return sample
