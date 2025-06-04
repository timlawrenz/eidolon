"""
Defines the loss functions used for training the Eidolon FLAME encoder.

This module includes the `TotalLoss` class, which combines various loss components
such as pixel-wise loss, landmark alignment loss, and regularization losses
to guide the training of the neural network that predicts FLAME parameters.
"""

import torch
import torch.nn as nn

# You might need to load a pre-trained model for perceptual loss
# from torchvision.models import vgg19

class TotalLoss(nn.Module):
    def __init__(self, loss_weights):
        """
        Initializes the TotalLoss module.

        Args:
            loss_weights (dict): A dictionary with weights for each loss component.
                                 e.g., {'pixel': 1.0, 'landmark': 0.001, ...}
        """
        super().__init__()
        self.weights = loss_weights
        
        # 1. Pixel-wise Loss (Image similarity)
        self.pixel_loss = nn.L1Loss()
        
        # 2. Landmark Loss (Geometric alignment)
        # This measures the 2D distance between predicted and ground-truth landmarks
        self.landmark_loss = nn.MSELoss()
        
        # 3. Perceptual Loss (Identity preservation)
        # TODO: Load a pre-trained face recognition model (e.g., VGG, ArcFace)
        # self.perceptual_loss = ... 

        # 4. Regularization Loss (Plausibility of parameters)
        # This penalizes the model for predicting extreme, unrealistic shape/expression params
        # It's often a simple L2 norm (sum of squares) on the coefficient vectors.
        # We can implement this directly in the forward pass.

    def forward(self, pred_coeffs, pred_verts, pred_landmarks_2d, rendered_image, gt_image, gt_landmarks_2d):
        """
        Calculates the total weighted loss.
        
        Args:
            pred_coeffs (dict): Dictionary of predicted coefficients from the encoder.
                                Expected keys: 'shape', 'expression'.
            pred_verts (torch.Tensor): The final deformed vertices from the FLAME model.
            pred_landmarks_2d (torch.Tensor): The 2D projection of the FLAME landmarks.
            rendered_image (torch.Tensor): The image from the PyTorch3D renderer.
            gt_image (torch.Tensor): The ground-truth input image.
            gt_landmarks_2d (torch.Tensor): The ground-truth landmarks from the input image.

        Returns:
            torch.Tensor: The total calculated loss.
            dict: A dictionary containing the individual loss values for logging.
        """
        
        # --- Pixel Loss ---
        loss_pixel = self.pixel_loss(rendered_image, gt_image)

        # --- Landmark Loss ---
        loss_landmark = self.landmark_loss(pred_landmarks_2d, gt_landmarks_2d)
        
        # --- Parameter Regularization Loss ---
        # Penalize large shape and expression params to encourage plausible faces
        loss_reg_shape = torch.tensor(0.0, device=rendered_image.device) # Use a common device
        if 'shape' in pred_coeffs and \
           pred_coeffs['shape'] is not None and \
           pred_coeffs['shape'].numel() > 0 and \
           self.weights.get('reg_shape', 0.0) > 0:
            loss_reg_shape = (pred_coeffs['shape'] ** 2).mean()
        
        loss_reg_expression = torch.tensor(0.0, device=rendered_image.device) # Use a common device
        if 'expression' in pred_coeffs and \
           pred_coeffs['expression'] is not None and \
           pred_coeffs['expression'].numel() > 0 and \
           self.weights.get('reg_expression', 0.0) > 0:
            loss_reg_expression = (pred_coeffs['expression'] ** 2).mean()

        loss_reg_transl = torch.tensor(0.0, device=rendered_image.device)
        if 'transl' in pred_coeffs and \
           pred_coeffs['transl'] is not None and \
           pred_coeffs['transl'].numel() > 0 and \
           self.weights.get('reg_transl', 0.0) > 0:
            loss_reg_transl = (pred_coeffs['transl'] ** 2).mean()

        loss_reg_global_pose = torch.tensor(0.0, device=rendered_image.device)
        if 'global_pose' in pred_coeffs and \
           pred_coeffs['global_pose'] is not None and \
           pred_coeffs['global_pose'].numel() > 0 and \
           self.weights.get('reg_global_pose', 0.0) > 0:
            loss_reg_global_pose = (pred_coeffs['global_pose'] ** 2).mean()

        loss_reg_jaw_pose = torch.tensor(0.0, device=rendered_image.device)
        if 'jaw_pose' in pred_coeffs and \
           pred_coeffs['jaw_pose'] is not None and \
           pred_coeffs['jaw_pose'].numel() > 0 and \
           self.weights.get('reg_jaw_pose', 0.0) > 0:
            loss_reg_jaw_pose = (pred_coeffs['jaw_pose'] ** 2).mean()

        loss_reg_neck_pose = torch.tensor(0.0, device=rendered_image.device)
        if 'neck_pose' in pred_coeffs and \
           pred_coeffs['neck_pose'] is not None and \
           pred_coeffs['neck_pose'].numel() > 0 and \
           self.weights.get('reg_neck_pose', 0.0) > 0:
            loss_reg_neck_pose = (pred_coeffs['neck_pose'] ** 2).mean()

        loss_reg_eye_pose = torch.tensor(0.0, device=rendered_image.device)
        if 'eye_pose' in pred_coeffs and \
           pred_coeffs['eye_pose'] is not None and \
           pred_coeffs['eye_pose'].numel() > 0 and \
           self.weights.get('reg_eye_pose', 0.0) > 0:
            loss_reg_eye_pose = (pred_coeffs['eye_pose'] ** 2).mean()
        
        # --- TODO: Perceptual Loss ---
        # loss_perceptual = self.perceptual_loss(rendered_image, gt_image)
        # For now, let's ensure it's defined if we uncomment its usage later
        loss_perceptual = torch.tensor(0.0, device=rendered_image.device)


        # --- Total Weighted Loss ---
        total_loss = (
            self.weights.get('pixel', 1.0) * loss_pixel +
            self.weights.get('landmark', 1.0) * loss_landmark +
            self.weights.get('reg_shape', 1.0) * loss_reg_shape +
            self.weights.get('reg_expression', 1.0) * loss_reg_expression +
            self.weights.get('reg_transl', 1.0) * loss_reg_transl +
            self.weights.get('reg_global_pose', 1.0) * loss_reg_global_pose +
            self.weights.get('reg_jaw_pose', 1.0) * loss_reg_jaw_pose +
            self.weights.get('reg_neck_pose', 1.0) * loss_reg_neck_pose +
            self.weights.get('reg_eye_pose', 1.0) * loss_reg_eye_pose
            # + self.weights.get('perceptual', 1.0) * loss_perceptual # Keep commented for now
        )
        
        loss_dict = {
            'total': total_loss,
            'pixel': loss_pixel,
            'landmark': loss_landmark,
            'reg_shape': loss_reg_shape,
            'reg_expression': loss_reg_expression,
            'reg_transl': loss_reg_transl,
            'reg_global_pose': loss_reg_global_pose,
            'reg_jaw_pose': loss_reg_jaw_pose,
            'reg_neck_pose': loss_reg_neck_pose,
            'reg_eye_pose': loss_reg_eye_pose,
            # 'perceptual': loss_perceptual # Keep commented for now
        }
        
        # If perceptual loss is to be included, uncomment its addition to total_loss
        # and its entry in loss_dict.
        if 'perceptual' in self.weights and self.weights['perceptual'] > 0:
            # Placeholder for actual perceptual loss calculation
            # loss_perceptual = self.perceptual_loss(rendered_image, gt_image) # This would call the actual function
            # total_loss += self.weights['perceptual'] * loss_perceptual
            loss_dict['perceptual'] = loss_perceptual # Add to dict if used

        return total_loss, loss_dict
