
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class EidolonEncoder(nn.Module):
    def __init__(self, num_coeffs):
        """
        The Encoder model that takes an image and outputs FLAME coefficients.

        Args:
            num_coeffs (int): The total number of parameters to predict.
                              (e.g., 100 shape + 50 expression + ... = 227)
        """
        super().__init__()

        # 1. Load a pre-trained ResNet-50 model
        # We use the most up-to-date weights
        weights = ResNet50_Weights.DEFAULT
        self.backbone = resnet50(weights=weights)

        # 2. Get the number of input features for the final layer
        # This is typically 2048 for ResNet-50
        num_bottleneck_features = self.backbone.fc.in_features

        # 3. Replace the final layer (the "head")
        # The original ResNet-50 was for classifying 1000 ImageNet classes.
        # We replace it with a new linear layer that outputs our coefficient vector.
        self.backbone.fc = nn.Linear(num_bottleneck_features, num_coeffs)

    def forward(self, image):
        """
        The forward pass of the model.
        
        Args:
            image (torch.Tensor): A batch of input images (B, C, H, W).
        
        Returns:
            torch.Tensor: The predicted coefficients for the batch (B, num_coeffs).
        """
        return self.backbone(image)
