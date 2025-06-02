
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np
import pickle

# Helper function for LBS (will be part of the TODO for full FLAME)
def lbs(betas, pose, v_template, shapedirs, posedirs, J_regressor, parents, lbs_weights, dtype=torch.float32):
    # Placeholder for LBS: returns posedirs applied with zero pose for now
    # TODO: Implement full LBS
    # For now, we just return a zero deformation due to pose to make the model runnable
    batch_size = betas.shape[0]
    device = betas.device
    return torch.zeros_like(v_template).unsqueeze(0).repeat(batch_size, 1, 1)


class EidolonEncoder(nn.Module):
    """
    An encoder model that processes an input image using a pre-trained ResNet-50
    backbone and outputs a vector of coefficients, intended to represent
    FLAME model parameters (shape, expression, pose, etc.).
    """
    def __init__(self, num_coeffs):
        """
        Initializes the EidolonEncoder.

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

class FLAME(nn.Module):
    """
    Placeholder for the FLAME PyTorch model.
    This model will take FLAME parameters (shape, expression, pose, etc.)
    and output 3D vertices and 3D landmarks.
    """
    def __init__(self, flame_model_path, landmark_embedding_path, n_shape, n_exp):
        super().__init__()

        with open(flame_model_path, 'rb') as f:
            flame_model_data = pickle.load(f, encoding='latin1')

        # FLAME components
        # The .pkl file seems to load directly as NumPy arrays, not chumpy objects.
        v_template_np = flame_model_data['v_template']
        self.register_buffer('v_template', torch.tensor(v_template_np, dtype=torch.float32))
        num_vertices = v_template_np.shape[0]

        # Shapedirs: Expected shape is (num_vertices, 3, total_shape_coeffs)
        shapedirs_data_np = flame_model_data['shapedirs']
        self.register_buffer('shapedirs', torch.tensor(shapedirs_data_np[:, :, :n_shape], dtype=torch.float32))
        
        # Expressedirs: Check if exists
        if 'expressedirs' in flame_model_data and flame_model_data['expressedirs'] is not None:
             expressedirs_data_np = flame_model_data['expressedirs']
             self.register_buffer('expressedirs', torch.tensor(expressedirs_data_np[:, :, :n_exp], dtype=torch.float32))
        else:
            print("Warning: 'expressedirs' not found or is None in FLAME model. Expression parameters will have no effect.")
            self.register_buffer('expressedirs', torch.zeros((num_vertices, 3, n_exp), dtype=torch.float32, device=self.v_template.device))

        # Posedirs: Expected shape is (num_vertices, 3, total_pose_blendshapes)
        posedirs_data_np = flame_model_data['posedirs']
        self.register_buffer('posedirs', torch.tensor(posedirs_data_np, dtype=torch.float32)) # Use all pose blendshapes
        
        # J_regressor: Typically a sparse matrix
        j_regressor_data = flame_model_data['J_regressor']
        if hasattr(j_regressor_data, 'toarray'): # Check for sparse matrix
            j_regressor_np = j_regressor_data.toarray()
        else: # Assume it's already a NumPy array
            j_regressor_np = j_regressor_data
        self.register_buffer('J_regressor', torch.tensor(j_regressor_np, dtype=torch.float32))
        
        # LBS weights
        lbs_weights_np = flame_model_data['weights']
        self.register_buffer('lbs_weights', torch.tensor(lbs_weights_np, dtype=torch.float32))
        
        # Faces (typically already a NumPy array)
        faces_np = flame_model_data['f'].astype(np.int64)
        self.register_buffer('faces_idx', torch.tensor(faces_np, dtype=torch.long))
        
        # Kinematic tree (parents of joints)
        # The FLAME pkl might store parents differently, e.g., flame_model_data['parent']
        # For now, LBS is a TODO, so parents are not immediately critical.
        # We'll use a placeholder if not found, but a real LBS implementation needs correct parents.
        if 'kintree_table' in flame_model_data:
             parents = flame_model_data['kintree_table'][0].astype(np.int64)
             parents[0] = -1 # Root joint has no parent
             self.register_buffer('parents', torch.tensor(parents, dtype=torch.long))
        elif 'parent' in flame_model_data: # Some FLAME versions might use this key
             parents = flame_model_data['parent'].astype(np.int64)
             parents[0] = -1 
             self.register_buffer('parents', torch.tensor(parents, dtype=torch.long))
        else:
            print("Warning: Joint parent information ('kintree_table' or 'parent') not found in FLAME model. LBS will be affected.")
            # Placeholder for number of joints, typically 16 for FLAME (including global)
            # This needs to match J_regressor.shape[0]
            num_joints = self.J_regressor.shape[0] if hasattr(self, 'J_regressor') else 16 
            self.register_buffer('parents', torch.full((num_joints,), -1, dtype=torch.long))


        # Load 3D landmark embedding
        landmark_data = np.load(landmark_embedding_path, allow_pickle=True)
        self.register_buffer('landmark_vertex_ids', torch.tensor(landmark_data['vertex_ids'], dtype=torch.long))


    def forward(self, shape_params=None, expression_params=None, pose_params=None, 
                  eye_pose_params=None, jaw_pose_params=None, neck_pose_params=None, transl=None, detail_params=None):
        """
        Forward pass of the FLAME model.

        Args:
            shape_params (torch.Tensor): Shape parameters (B, N_shape).
            expression_params (torch.Tensor): Expression parameters (B, N_expression).
            pose_params (torch.Tensor): Global pose parameters (B, N_global_pose).
            eye_pose_params (torch.Tensor): Eye pose parameters (B, N_eye_pose).
            jaw_pose_params (torch.Tensor): Jaw pose parameters (B, N_jaw_pose).
            neck_pose_params (torch.Tensor): Neck pose parameters (B, N_neck_pose).
            transl (torch.Tensor): Translation parameters (B, 3).
            detail_params (torch.Tensor): Detail/texture parameters (B, N_detail), currently unused.


        Returns:
            pred_verts (torch.Tensor): Predicted 3D vertices (B, N_verts, 3).
            pred_landmarks_3d (torch.Tensor): Predicted 3D landmarks (B, N_landmarks, 3).
        """
        batch_size = shape_params.shape[0]
        device = shape_params.device

        # 1. Shape deformation
        # v_template: (N_verts, 3)
        # shapedirs: (N_verts, 3, N_shape_coeffs)
        # shape_params: (B, N_shape_coeffs)
        # shape_blendshape = torch.einsum('bl,vcl->bvc', shape_params, self.shapedirs)
        # More robust way if shapedirs is (N_verts * 3, N_shape_coeffs)
        # For (N_verts, 3, N_shape_coeffs)
        shape_offset = torch.einsum('bS,VCS->bVC', shape_params, self.shapedirs).contiguous()
        v_shaped = self.v_template.unsqueeze(0).repeat(batch_size, 1, 1) + shape_offset

        # 2. Expression deformation
        # expressedirs: (N_verts, 3, N_expr_coeffs)
        # expression_params: (B, N_expr_coeffs)
        expression_offset = torch.einsum('bE,VCE->bVC', expression_params, self.expressedirs).contiguous()
        v_expressed = v_shaped + expression_offset
        
        # 3. Pose deformation (LBS)
        # TODO: Implement full LBS using pose_params, jaw_pose_params, neck_pose_params, eye_pose_params
        # This involves:
        # - Converting all pose parameters (global, jaw, neck, eyes) to rotation matrices.
        # - Applying these to the template joints (J_regressor @ v_template).
        # - Calculating the transformation for each vertex using lbs_weights and posedirs.
        # For now, we use a placeholder that returns zero pose deformation.
        
        # Concatenate all pose parameters for the LBS function (if it expects them combined)
        # The LBS function would need to handle the specific structure of these poses.
        # Example: full_pose = torch.cat([pose_params, jaw_pose_params, neck_pose_params, eye_pose_params], dim=1)
        
        # For the placeholder LBS, we pass dummy betas and pose.
        # The actual LBS would use the various pose_params.
        pose_blendshapes = lbs(shape_params, pose_params, self.v_template, self.shapedirs, 
                               self.posedirs, self.J_regressor, self.parents, self.lbs_weights,
                               dtype=self.v_template.dtype)
        
        pred_verts_posed = v_expressed + pose_blendshapes # With placeholder LBS, pose_blendshapes is zero

        # 4. Apply global translation
        if transl is not None:
            pred_verts = pred_verts_posed + transl.unsqueeze(1)
        else:
            pred_verts = pred_verts_posed

        # 5. Calculate 3D landmarks
        pred_landmarks_3d = pred_verts[:, self.landmark_vertex_ids, :]
        
        return pred_verts, pred_landmarks_3d
