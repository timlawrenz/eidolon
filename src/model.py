
import torch
import torch.nn as nn
import torch.nn.functional as F # Added for DECA's LBS components
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np
import pickle
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_axis_angle # For pose conversion

# --- LBS Helper Functions ---

def batch_rodrigues(rot_vecs, epsilon=1e-8):
    ''' Converts a batch of axis-angle vectors to rotation matrices.
    Args:
        rot_vecs: (B, 3) axis-angle vectors.
    Returns:
        (B, 3, 3) rotation matrices.
    '''
    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device
    dtype = rot_vecs.dtype

    angle = torch.norm(rot_vecs + epsilon, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.cos(angle).unsqueeze(1)
    sin = torch.sin(angle).unsqueeze(1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mats = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mats

# DECA's helper function for batch_rigid_transform
def transform_mat(R, t):
    ''' Creates a batch of transformation matrices
        Args:
            - R: Bx3x3 array of a batch of rotation matrices
            - t: Bx3x1 array of a batch of translation vectors
        Returns:
            - T: Bx4x4 Transformation matrix
    '''
    # No padding left or right, only add an extra row
    return torch.cat([F.pad(R, [0, 0, 0, 1]),
                      F.pad(t, [0, 0, 0, 1], value=1)], dim=2)

def batch_rigid_transform(rot_mats, joints, parents, dtype=torch.float32):
    """
    Applies rigid transformations to the joints based on the kinematic tree.
    Args:
        rot_mats (torch.Tensor): Batch of rotation matrices (B, J, 3, 3).
        joints (torch.Tensor): Batch of initial joint locations (B, J, 3).
        parents (torch.Tensor): Parent indices for each joint (J).
    Returns:
        A_global (torch.Tensor): Batch of global transformation matrices (B, J, 4, 4).
    """
    # Args: (Kept from original for reference, DECA's args are compatible)
    #   rot_mats (torch.Tensor): Batch of rotation matrices for LBS joints (B, num_lbs_joints, 3, 3).
    #   joints (torch.Tensor): Batch of LBS joint locations (B, num_lbs_joints, 3).
    #                          These are typically the rest-pose joint locations.
    #   parents (torch.Tensor): Parent indices for LBS joints (num_lbs_joints), root parent is -1.
    #                           DECA's code expects parents[0] to be a valid index for transform_chain,
    #                           but the loop `for i in range(1, parents.shape[0])` means parents[0]
    #                           is used for transform_chain[parents[0]] only if parents[0] is a parent of another joint.
    #                           Effectively, parents[0] is the root. My parents_lbs is [-1,0,1,1,1].
    #                           DECA's `rel_joints[:, 1:] -= joints[:, parents[1:]]` uses parents[1:].
    #   dtype (torch.dtype): Data type for new tensors.

    # DECA's batch_rigid_transform implementation:
    # joints is (B, N, 3), rot_mats is (B, N, 3, 3), parents is (N)
    _joints = torch.unsqueeze(joints, dim=-1) # (B, N, 3, 1)

    _rel_joints = _joints.clone()
    # parents[0] is -1 (root), so parents[1:] are valid indices [0,1,1,1] for the 5 LBS joints.
    # This correctly calculates relative joint positions for children joints.
    if parents.shape[0] > 1: # Ensure there are child joints
        _rel_joints[:, 1:] -= _joints[:, parents[1:]]

    # transforms_mat is (B, N, 4, 4)
    transforms_mat = transform_mat(
        rot_mats.reshape(-1, 3, 3), # view is also fine here
        _rel_joints.reshape(-1, 3, 1)).reshape(-1, joints.shape[1], 4, 4)

    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, parents.shape[0]):
        # parents[i] is the index of the parent joint.
        # transform_chain[parents[i]] is the global transform of the parent.
        curr_res = torch.matmul(transform_chain[parents[i]],
                                transforms_mat[:, i])
        transform_chain.append(curr_res)

    transforms = torch.stack(transform_chain, dim=1) # (B, N, 4, 4), these are G_j

    # The last column of the transformations contains the posed joints
    posed_joints = transforms[:, :, :3, 3] # J_transformed

    # Calculate skinning matrices A_j = G_j @ inv(T_j_rest)
    # where T_j_rest = [I | J_j_rest; 0 | 1]
    # inv(T_j_rest) = [I | -J_j_rest; 0 | 1]
    # So, A_j_rot = G_j_rot
    #     A_j_trans = G_j_trans - G_j_rot @ J_j_rest
    # This is equivalent to DECA's: rel_transforms = transforms - F.pad(torch.matmul(transforms, joints_homogen), ...)
    
    # joints_homogen is (B, N, 4, 1)
    joints_homogen = F.pad(_joints, [0, 0, 0, 1]) # Pad the 3x1 joint vectors to 4x1

    # rel_transforms are the skinning matrices A
    # G_j - [0 | G_j @ J_j_rest_homo]
    # This results in: rel_transforms_rot = G_j_rot
    #                 rel_transforms_trans = G_j_trans - (G_j @ J_j_rest_homo)_trans
    #                                      = G_j_trans - (G_j_rot @ J_j_rest + G_j_trans)
    #                                      = - G_j_rot @ J_j_rest
    # This is [R | -R*J_rest], which is the standard skinning matrix.
    rel_transforms = transforms - F.pad(
        torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0])

    return posed_joints, rel_transforms


def lbs(v_shaped_expressed, 
        global_pose_params_6d, # (B, 6)
        neck_pose_params_ax,   # (B, 3) axis-angle
        jaw_pose_params_ax,    # (B, 3) axis-angle
        eye_pose_params_ax,    # (B, 6) axis-angle (left_eye_ax, right_eye_ax)
        J_transformed_rest_lbs, # (B, num_lbs_joints, 3) Pre-computed rest positions of LBS joints
        parents_lbs, 
        lbs_weights, 
        posedirs, 
        dtype=torch.float32,
        debug_print: bool = False):
    """
    Performs Linear Blend Skinning (LBS).
    Args:
        v_shaped_expressed (torch.Tensor): Vertices after shape and expression (B, N_verts, 3).
        global_pose_params_6d (torch.Tensor): Global pose parameters (B, 6).
        neck_pose_params_ax (torch.Tensor): Neck pose axis-angle (B, 3).
        jaw_pose_params_ax (torch.Tensor): Jaw pose axis-angle (B, 3).
        eye_pose_params_ax (torch.Tensor): Eye poses axis-angle (B, 6).
        J_transformed_rest_lbs (torch.Tensor): Pre-computed rest joint locations for LBS joints (B, num_lbs_joints, 3).
        parents_lbs (torch.Tensor): Parent indices for the LBS joints.
        lbs_weights (torch.Tensor): LBS weights.
        posedirs (torch.Tensor): Pose-dependent blendshapes.
    Returns:
        v_posed (torch.Tensor): Posed vertices.
    """
    if debug_print:
        print(f"--- ENTERING LBS FUNCTION (batch_size={v_shaped_expressed.shape[0]}) ---") # VERY FIRST LINE
        print(f"--- DEBUG lbs input: v_shaped_expressed[0] Stats ---")
        if v_shaped_expressed.numel() > 0 and v_shaped_expressed.shape[0] > 0:
            print(f"  Shape: {v_shaped_expressed.shape}")
            print(f"  X: mean={v_shaped_expressed[0, :, 0].mean().item():.4f}, std={v_shaped_expressed[0, :, 0].std().item():.4f}, "
                  f"min={v_shaped_expressed[0, :, 0].min().item():.4f}, max={v_shaped_expressed[0, :, 0].max().item():.4f}")
            print(f"  Y: mean={v_shaped_expressed[0, :, 1].mean().item():.4f}, std={v_shaped_expressed[0, :, 1].std().item():.4f}, "
                  f"min={v_shaped_expressed[0, :, 1].min().item():.4f}, max={v_shaped_expressed[0, :, 1].max().item():.4f}")
            print(f"  Z: mean={v_shaped_expressed[0, :, 2].mean().item():.4f}, std={v_shaped_expressed[0, :, 2].std().item():.4f}, "
                  f"min={v_shaped_expressed[0, :, 2].min().item():.4f}, max={v_shaped_expressed[0, :, 2].max().item():.4f}")
        else:
            print("  v_shaped_expressed is empty or has zero batch size.")
        print(f"----------------------------------------------------")

    batch_size = v_shaped_expressed.shape[0]
    device = v_shaped_expressed.device

    # 1. Convert pose parameters to rotation matrices for LBS joints
    # Global Pose (from 6D to 3x3 Rotation Matrix)
    if global_pose_params_6d.shape[1] == 6:
        global_rot_mat = rotation_6d_to_matrix(global_pose_params_6d) # (B, 3, 3)
    elif global_pose_params_6d.shape[1] == 3: # If global pose is already axis-angle
        print("Warning: Global pose is 3D (axis-angle), converting to matrix. Consider 6D output from encoder.")
        global_rot_mat = batch_rodrigues(global_pose_params_6d)
    else:
        print(f"Warning: Global pose_params have unexpected shape {global_pose_params_6d.shape}. Expected 3 or 6. Using identity for global rotation.")
        global_rot_mat = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).repeat(batch_size, 1, 1)

    # Calculate neck, jaw, and eye rotation matrices normally
    neck_rot_mat = batch_rodrigues(neck_pose_params_ax)
    jaw_rot_mat = batch_rodrigues(jaw_pose_params_ax)
    eye_l_rot_mat = batch_rodrigues(eye_pose_params_ax[:, :3])
    eye_r_rot_mat = batch_rodrigues(eye_pose_params_ax[:, 3:])

    # Stack rotation matrices for the 5 main LBS joints: global, neck, jaw, left_eye, right_eye
    rot_mats_lbs = torch.stack([
        global_rot_mat, neck_rot_mat, jaw_rot_mat, eye_l_rot_mat, eye_r_rot_mat
    ], dim=1) # (B, 5, 3, 3)

    # 2. Rest-pose LBS joint locations (J_transformed_rest_lbs) are now passed as an argument.
    #    This tensor is pre-computed in FLAME.forward using self.J_regressor and self.v_template.
    #    self.J_regressor is expected to be for the 5 LBS joints.

    if debug_print:
        print(f"--- DEBUG lbs: J_transformed_rest_lbs[0] Stats ---")
        print(f"  Shape: {J_transformed_rest_lbs.shape}")
        if J_transformed_rest_lbs.numel() > 0 and J_transformed_rest_lbs.shape[0] > 0:
            print(f"  X: mean={J_transformed_rest_lbs[0, :, 0].mean().item():.4f}, std={J_transformed_rest_lbs[0, :, 0].std().item():.4f}")
            print(f"  Y: mean={J_transformed_rest_lbs[0, :, 1].mean().item():.4f}, std={J_transformed_rest_lbs[0, :, 1].std().item():.4f}")
            print(f"  Z: mean={J_transformed_rest_lbs[0, :, 2].mean().item():.4f}, std={J_transformed_rest_lbs[0, :, 2].std().item():.4f}")
        print(f"--- DEBUG lbs: rot_mats_lbs[0] Stats (first 3x3 of 5) ---")
        print(f"  Shape: {rot_mats_lbs.shape}")
        if rot_mats_lbs.numel() > 0 and rot_mats_lbs.shape[0] > 0:
            print(rot_mats_lbs[0, 0, :, :]) # Print the first global rotation matrix for the first sample
        print(f"----------------------------------------------------")

    # 3. Get global joint transformations A_global (B, num_lbs_joints, 4, 4)
    #    using the pre-computed rest_pose LBS joint locations.
    #    J_transformed_rest_lbs is (B, num_lbs_joints, 3)
    #    rot_mats_lbs is (B, num_lbs_joints, 3, 3)
    #    parents_lbs is (num_lbs_joints)
    
    # batch_rigid_transform (DECA's version) returns:
    # posed_joints (J_transformed_deca): (B, num_lbs_joints, 3)
    # rel_transforms (A_global_deca): (B, num_lbs_joints, 4, 4) - These are the skinning matrices
    _, A_global = batch_rigid_transform(rot_mats_lbs, J_transformed_rest_lbs, parents_lbs, dtype=dtype)
    # We use A_global (rel_transforms) for skinning. posed_joints is not directly used here for skinning v_shaped_expressed.

    if debug_print:
        print(f"--- DEBUG lbs: A_global[0] Stats (first 4x4 of 5) ---")
        print(f"  Shape: {A_global.shape}")
        if A_global.numel() > 0 and A_global.shape[0] > 0:
            print(A_global[0, 0, :, :]) # Print the first skinning matrix for the first sample
            # Check for NaNs or Infs in A_global
            if torch.isnan(A_global).any() or torch.isinf(A_global).any():
                print("CRITICAL WARNING: NaNs or Infs found in A_global!")
        print(f"-------------------------------------------------")

    # 4. Calculate pose-corrective blendshapes (posedirs)
    # Standard FLAME posedirs (36 features) are typically driven by neck, jaw, left eye, and right eye rotations.
    num_joints_for_posedirs = 4 # Neck, Jaw, Left Eye, Right Eye
    # rot_mats_lbs is stacked as [global, neck, jaw, eye_l, eye_r]
    # Select indices 1 (neck), 2 (jaw), 3 (eye_l), 4 (eye_r).
    rot_mats_subset_for_posedirs = rot_mats_lbs[:, 1:1+num_joints_for_posedirs, :, :] # (B, 4, 3, 3)

    ident = torch.eye(3, device=device, dtype=dtype).unsqueeze(0) # (1,3,3)
    # pose_feature_vector_from_4_joints will be (B, 4*9=36)
    pose_feature_vector_from_4_joints = (rot_mats_subset_for_posedirs - ident).view(batch_size, -1) 
    
    # posedirs (loaded and permuted in FLAME.__init__) has shape (V, P, 3).
    # P (number of pose features) is posedirs.shape[1].
    # For the loaded flame2023.pkl, P is 36.
    num_features_expected_by_posedirs = posedirs.shape[1]
    
    current_pose_feature_vector = pose_feature_vector_from_4_joints # (B, 36)

    if current_pose_feature_vector.shape[1] != num_features_expected_by_posedirs:
        # This warning should ideally not trigger if num_features_expected_by_posedirs is 36.
        print(f"Warning: The calculated pose feature vector (from {num_joints_for_posedirs} joints: neck, jaw, eyes) "
              f"has {current_pose_feature_vector.shape[1]} features, "
              f"but the 'posedirs' tensor expects {num_features_expected_by_posedirs} features. "
              f"This mismatch will result in a zero pose-corrective blendshape effect. "
              f"Ensure the 'posedirs' data matches the {num_joints_for_posedirs} driving joints if a non-zero effect is desired.")
        # To prevent a runtime error in einsum and apply a zero effect due to mismatch:
        current_pose_feature_vector = torch.zeros(batch_size, num_features_expected_by_posedirs, device=device, dtype=dtype)
            
    # Einsum: current_pose_feature_vector (B, P_expected), posedirs (V, P_expected, C) -> pose_blendshapes (B, V, C)
    # Here C=3 (for x,y,z offsets), P_expected is num_features_expected_by_posedirs.
    pose_blendshapes = torch.einsum('BP,VPC->BVC', current_pose_feature_vector, posedirs) # Actual calculation

    # Apply posedirs to v_shaped_expressed before skinning
    v_to_skin = v_shaped_expressed + pose_blendshapes

    if debug_print:
        print(f"--- DEBUG lbs: v_to_skin[0] Stats ---")
        print(f"  Shape: {v_to_skin.shape}")
        if v_to_skin.numel() > 0 and v_to_skin.shape[0] > 0:
            print(f"  X: mean={v_to_skin[0, :, 0].mean().item():.4f}, std={v_to_skin[0, :, 0].std().item():.4f}")
            print(f"  Y: mean={v_to_skin[0, :, 1].mean().item():.4f}, std={v_to_skin[0, :, 1].std().item():.4f}")
            print(f"  Z: mean={v_to_skin[0, :, 2].mean().item():.4f}, std={v_to_skin[0, :, 2].std().item():.4f}")
            if torch.isnan(v_to_skin).any() or torch.isinf(v_to_skin).any():
                print("CRITICAL WARNING: NaNs or Infs found in v_to_skin!")
        print(f"---------------------------------------")

    # 5. Transform vertices by LBS
    T = torch.einsum('VJ,BJHW->BVHW', lbs_weights, A_global) # lbs_weights (N_verts, num_lbs_joints)
    # Use v_to_skin for skinning
    v_homo = torch.cat([v_to_skin, torch.ones(batch_size, v_to_skin.shape[1], 1, device=device, dtype=dtype)], dim=2)
    v_posed_lbs = torch.einsum('BVHW,BVW->BVH', T, v_homo)[:, :, :3]
    
    # v_posed is now the result of skinning the posedirs-corrected mesh
    v_posed = v_posed_lbs

    if debug_print:
        print(f"--- DEBUG lbs: v_posed[0] (output) Stats ---")
        print(f"  Shape: {v_posed.shape}")
        if v_posed.numel() > 0 and v_posed.shape[0] > 0:
            print(f"  X: mean={v_posed[0, :, 0].mean().item():.4f}, std={v_posed[0, :, 0].std().item():.4f}")
            print(f"  Y: mean={v_posed[0, :, 1].mean().item():.4f}, std={v_posed[0, :, 1].std().item():.4f}")
            print(f"  Z: mean={v_posed[0, :, 2].mean().item():.4f}, std={v_posed[0, :, 2].std().item():.4f}")
            if torch.isnan(v_posed).any() or torch.isinf(v_posed).any():
                print("CRITICAL WARNING: NaNs or Infs found in v_posed output!")
        print(f"--------------------------------------------")
    return v_posed


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

        # Freeze all parameters in the backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        # 2. Get the number of input features for the final layer
        # This is typically 2048 for ResNet-50
        num_bottleneck_features = self.backbone.fc.in_features

        # 3. Replace the final layer (the "head")
        # The original ResNet-50 was for classifying 1000 ImageNet classes.
        # We replace it with a new linear layer that outputs our coefficient vector.
        self.backbone.fc = nn.Linear(num_bottleneck_features, num_coeffs)

        # Initialize the biases of the new fully connected layer
        # This aims to start predictions closer to a neutral/canonical face
        with torch.no_grad():
            # Define the expected number of coefficients for each parameter type
            # These should match the definitions in train.py (NUM_SHAPE_COEFFS, etc.)
            # For num_coeffs = 227:
            n_shape = 100
            n_expr = 0 # Must match NUM_EXPRESSION_COEFFS in train.py (currently 0)
            n_global_pose = 6 # 6D rotation
            n_jaw_pose = 3    # axis-angle
            n_eye_pose = 6    # axis-angle
            n_neck_pose = 3   # axis-angle
            n_transl = 3
            # n_detail = 56 (remaining, NUM_COEFFS - sum_of_above)

            # Initialize all biases to zero first
            self.backbone.fc.bias.fill_(0.0)

            # Set specific biases for global pose to encourage identity rotation
            # For 6D rotation (representing the first two columns of a 3x3 matrix),
            # identity is [1,0,0] and [0,1,0]. So, 6D vector is (1,0,0,0,1,0).
            current_idx = n_shape + n_expr
            # Global pose params (6D)
            if num_coeffs >= current_idx + n_global_pose:
                # Indices for the 6D global pose parameters
                # R_col1_x, R_col1_y, R_col1_z, R_col2_x, R_col2_y, R_col2_z
                # Identity: col1=(1,0,0), col2=(0,1,0)
                self.backbone.fc.bias[current_idx + 0] = 1.0 # R_col1_x
                self.backbone.fc.bias[current_idx + 1] = 0.0 # R_col1_y
                self.backbone.fc.bias[current_idx + 2] = 0.0 # R_col1_z
                self.backbone.fc.bias[current_idx + 3] = 0.0 # R_col2_x
                self.backbone.fc.bias[current_idx + 4] = 1.0 # R_col2_y
                self.backbone.fc.bias[current_idx + 5] = 0.0 # R_col2_z
            
            # Other parameters (shape, expression, jaw/neck/eye poses, translation, detail)
            # will have their biases remain at 0.0, encouraging neutral initial values.
            # This means:
            # - Zero shape offset (average shape)
            # - Zero expression offset
            # - Zero jaw, neck, eye pose (identity rotation for axis-angle)
            # - Zero translation
            # - Zero detail parameters

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
    def __init__(self, flame_model_path, deca_landmark_embedding_path, n_shape, n_exp): # Changed parameter name
        super().__init__()

        with open(flame_model_path, 'rb') as f:
            flame_model_data = pickle.load(f, encoding='latin1')
        
        print(f"DEBUG: FLAME model data keys and types:")
        for key, value in flame_model_data.items():
            type_info = f"  Key: '{key}', Type: {type(value)}"
            if hasattr(value, 'r'): # For chumpy objects
                type_info += f", .r Type: {type(value.r)}"
                if hasattr(value.r, 'shape'):
                    type_info += f", .r Shape: {value.r.shape}"
            elif hasattr(value, 'shape'): # For numpy arrays or tensors
                 type_info += f", Shape: {value.shape}"
            print(type_info)

        # --- Debugging expressedirs ---
        if 'expressedirs' in flame_model_data:
            print(f"DEBUG: 'expressedirs' key found. Type: {type(flame_model_data['expressedirs'])}")
            expressedirs_content = flame_model_data['expressedirs']
            if expressedirs_content is not None:
                if hasattr(expressedirs_content, 'shape'):
                    print(f"DEBUG: 'expressedirs' content shape: {expressedirs_content.shape}")
                if hasattr(expressedirs_content, 'r'): # For chumpy objects
                    if hasattr(expressedirs_content.r, 'shape'):
                        print(f"DEBUG: 'expressedirs.r' content shape: {expressedirs_content.r.shape}")
                    else:
                        print(f"DEBUG: 'expressedirs.r' exists but has no shape attribute.")
            else:
                print("DEBUG: 'expressedirs' key found, but its content is None.")
        else:
            print("DEBUG: 'expressedirs' key NOT found in flame_model_data.")
        # --- End Debugging expressedirs ---

        # FLAME components
        # Conditionally access .r to get NumPy arrays from potential chumpy objects
        
        v_template_data = flame_model_data['v_template']
        v_template_np = v_template_data.r if hasattr(v_template_data, 'r') else v_template_data
        self.register_buffer('v_template', torch.tensor(v_template_np, dtype=torch.float32))
        num_vertices = v_template_np.shape[0]

        print(f"--- DEBUG: self.v_template Stats (in FLAME.__init__) ---")
        print(f"  Shape: {self.v_template.shape}")
        print(f"  X: mean={self.v_template[:, 0].mean().item():.4f}, std={self.v_template[:, 0].std().item():.4f}, "
              f"min={self.v_template[:, 0].min().item():.4f}, max={self.v_template[:, 0].max().item():.4f}")
        print(f"  Y: mean={self.v_template[:, 1].mean().item():.4f}, std={self.v_template[:, 1].std().item():.4f}, "
              f"min={self.v_template[:, 1].min().item():.4f}, max={self.v_template[:, 1].max().item():.4f}")
        print(f"  Z: mean={self.v_template[:, 2].mean().item():.4f}, std={self.v_template[:, 2].std().item():.4f}, "
              f"min={self.v_template[:, 2].min().item():.4f}, max={self.v_template[:, 2].max().item():.4f}")
        print(f"----------------------------------------------------------")

        # Shapedirs: Expected shape after .r is (num_vertices, 3, total_shape_coeffs)
        shapedirs_data = flame_model_data['shapedirs']
        shapedirs_np = shapedirs_data.r if hasattr(shapedirs_data, 'r') else shapedirs_data
        self.register_buffer('shapedirs', torch.tensor(shapedirs_np[:, :, :n_shape], dtype=torch.float32))
        
        # Expressedirs: Check if exists and access .r if it does
        if 'expressedirs' in flame_model_data and flame_model_data['expressedirs'] is not None:
             expressedirs_data = flame_model_data['expressedirs']
             expressedirs_np = expressedirs_data.r if hasattr(expressedirs_data, 'r') else expressedirs_data
             self.register_buffer('expressedirs', torch.tensor(expressedirs_np[:, :, :n_exp], dtype=torch.float32))
        else:
            print("Warning: 'expressedirs' not found or is None in FLAME model. Expression parameters will have no effect.")
            self.register_buffer('expressedirs', torch.zeros((num_vertices, 3, n_exp), dtype=torch.float32, device=self.v_template.device))

        # Posedirs: Expected shape after .r is (num_vertices, 3, total_pose_blendshapes)
        posedirs_data = flame_model_data['posedirs']
        posedirs_np = posedirs_data.r if hasattr(posedirs_data, 'r') else posedirs_data
        # posedirs_np is often (num_vertices, 3, total_pose_blendshapes) e.g. (V, 3, 207)
        # For einsum 'BP,VPC->BVC', P is the feature dim, C is coordinate dim (3 for x,y,z)
        # So, we need posedirs to be (V, P, 3). Permute if necessary.
        # Assuming posedirs_np from pickle is (V, 3, P_dim)
        if posedirs_np.shape[1] == 3 and posedirs_np.shape[2] != 3: # Likely (V, 3, P_dim)
            posedirs_permuted_np = np.transpose(posedirs_np, (0, 2, 1)) # Convert to (V, P_dim, 3)
            print(f"Permuted posedirs from {posedirs_np.shape} to {posedirs_permuted_np.shape}")
        else: # Assuming it's already (V, P_dim, 3) or other configuration
            posedirs_permuted_np = posedirs_np
            print(f"Using posedirs with shape: {posedirs_permuted_np.shape} (no permutation applied or shape is not (V,3,P))")
        self.register_buffer('posedirs', torch.tensor(posedirs_permuted_np, dtype=torch.float32))
        
        # J_regressor: Typically a sparse matrix
        j_regressor_data = flame_model_data['J_regressor']
        if hasattr(j_regressor_data, 'toarray'): # Check for sparse matrix (e.g., scipy.sparse)
            j_regressor_np = j_regressor_data.toarray()
        elif hasattr(j_regressor_data, 'r'): # Check for chumpy object
            j_regressor_np = j_regressor_data.r
        else: # Assume it's already a NumPy array
            j_regressor_np = j_regressor_data
        self.register_buffer('J_regressor', torch.tensor(j_regressor_np, dtype=torch.float32))
        
        # LBS weights
        lbs_weights_data = flame_model_data['weights']
        lbs_weights_np = lbs_weights_data.r if hasattr(lbs_weights_data, 'r') else lbs_weights_data
        self.register_buffer('lbs_weights', torch.tensor(lbs_weights_np, dtype=torch.float32))
        
        # Faces (typically already a NumPy array in the pkl)
        faces_data = flame_model_data['f']
        faces_np = faces_data.astype(np.int64) if isinstance(faces_data, np.ndarray) else np.array(faces_data, dtype=np.int64)
        self.register_buffer('faces_idx', torch.tensor(faces_np, dtype=torch.long))
        
        # Kinematic tree (parents of joints)
        # self.parents_full_skeleton is for the full 16-joint skeleton if needed elsewhere.
        if 'kintree_table' in flame_model_data:
             parents_full_np = flame_model_data['kintree_table'][0].astype(np.int64)
             parents_full_np[0] = -1 # Root joint has no parent
             self.register_buffer('parents_full_skeleton', torch.tensor(parents_full_np, dtype=torch.long))
        elif 'parent' in flame_model_data: 
             parents_full_np = flame_model_data['parent'].astype(np.int64)
             parents_full_np[0] = -1 
             self.register_buffer('parents_full_skeleton', torch.tensor(parents_full_np, dtype=torch.long))
        else:
            print("Warning: Full skeleton parent information ('kintree_table' or 'parent') not found in FLAME model.")
            # This might be okay if only the 5-joint LBS is used.
            self.register_buffer('parents_full_skeleton', torch.empty(0, dtype=torch.long))

        # Define parents for the 5 LBS joints (global, neck, jaw, left_eye, right_eye)
        # This assumes a simplified hierarchy for these 5 joints for LBS purposes.
        # Global (0) is root. Neck (1) is child of Global. Jaw (2) is child of Neck.
        # Left Eye (3) and Right Eye (4) are children of Neck (simplified head).
        parents_lbs_np = np.array([-1, 0, 1, 1, 1], dtype=np.int64) # Corresponds to global, neck, jaw, eyeL, eyeR
        self.register_buffer('parents_lbs', torch.tensor(parents_lbs_np, dtype=torch.long))

        # Assert consistency in the number of LBS joints
        num_lbs_joints = self.parents_lbs.shape[0] # This will be 5
        assert self.J_regressor.shape[0] == num_lbs_joints, \
            f"J_regressor joint count ({self.J_regressor.shape[0]}) from pkl does not match " \
            f"expected LBS joint count ({num_lbs_joints}) defined by parents_lbs. " \
            f"The J_regressor in flame2023.pkl is expected to be (5, num_vertices)."
        assert self.lbs_weights.shape[1] == num_lbs_joints, \
            f"lbs_weights joint count ({self.lbs_weights.shape[1]}) does not match " \
            f"expected LBS joint count ({num_lbs_joints})."

        # --- Load DECA Landmark Data for 68 points ---
        # The argument `landmark_embedding_path` is now `deca_landmark_embedding_path`
        self.using_barycentric_landmarks = False
        NUM_EXPECTED_LANDMARKS = 68
        try:
            # deca_landmark_embedding_path is now the correct variable name from the signature
            deca_lmk_data_container = np.load(deca_landmark_embedding_path, allow_pickle=True)
            # Check if the loaded data is a scalar object array containing a dict
            if deca_lmk_data_container.shape == () and deca_lmk_data_container.dtype == object:
                deca_lmk_data = deca_lmk_data_container.item() 
            elif isinstance(deca_lmk_data_container, dict): # If it's already a dict (e.g. from a .npz file loaded as dict)
                deca_lmk_data = deca_lmk_data_container
            else:
                raise ValueError(f"DECA landmark file {deca_landmark_embedding_path} does not contain an expected dictionary structure.")


            # Use the 'full' keys for 68 landmarks from DECA embedding
            if 'full_lmk_faces_idx' in deca_lmk_data and 'full_lmk_bary_coords' in deca_lmk_data:
                lmk_faces_idx_68_np = deca_lmk_data['full_lmk_faces_idx']
                lmk_bary_coords_68_np = deca_lmk_data['full_lmk_bary_coords']

                # Squeeze if they have an unnecessary leading dimension of 1
                if lmk_faces_idx_68_np.ndim == 2 and lmk_faces_idx_68_np.shape[0] == 1:
                    lmk_faces_idx_68_np = lmk_faces_idx_68_np.squeeze(0)
                if lmk_bary_coords_68_np.ndim == 3 and lmk_bary_coords_68_np.shape[0] == 1:
                    lmk_bary_coords_68_np = lmk_bary_coords_68_np.squeeze(0)
                
                if lmk_faces_idx_68_np.shape == (NUM_EXPECTED_LANDMARKS,) and \
                   lmk_bary_coords_68_np.shape == (NUM_EXPECTED_LANDMARKS, 3):
                    self.register_buffer('landmark_face_idx', torch.tensor(lmk_faces_idx_68_np, dtype=torch.long))
                    self.register_buffer('landmark_b_coords', torch.tensor(lmk_bary_coords_68_np, dtype=torch.float32))
                    self.using_barycentric_landmarks = True
                    print(f"Successfully loaded 68 barycentric landmarks from DECA embedding: {deca_landmark_embedding_path}.")
                else:
                    print(f"Warning: DECA 'full_lmk_faces_idx' or 'full_lmk_bary_coords' have unexpected shapes after squeeze. "
                          f"Faces shape: {lmk_faces_idx_68_np.shape}, Coords shape: {lmk_bary_coords_68_np.shape}")
            else:
                print(f"Warning: Keys 'full_lmk_faces_idx' or 'full_lmk_bary_coords' not found in DECA embedding: {deca_landmark_embedding_path}.")

        except Exception as e:
            print(f"ERROR loading or processing DECA landmark embedding from {deca_landmark_embedding_path}: {e}")
        
        if not self.using_barycentric_landmarks:
            print("Critical Warning: Could not load 68-point barycentric landmarks from DECA embedding. "
                  "Landmark prediction will be zeros. Ensure the DECA file and keys are correct.")
            # Ensure dummy buffers exist if barycentric loading fails, to prevent errors in forward pass
            self.register_buffer('landmark_face_idx', torch.empty(0, dtype=torch.long)) 
            self.register_buffer('landmark_b_coords', torch.empty(0, dtype=torch.float32))
            # Also ensure landmark_vertex_ids is empty or non-existent if not using vertex based
            if hasattr(self, 'landmark_vertex_ids'): # Clean up if a previous logic path set this
                del self.landmark_vertex_ids 
            self.register_buffer('landmark_vertex_ids', torch.empty(0, dtype=torch.long)) # For the forward pass logic
        else: # Debug print if barycentric landmarks were successfully loaded
            print("--- DEBUG: Barycentric Landmark Data ---")
            print(f"self.landmark_face_idx shape: {self.landmark_face_idx.shape}")
            if self.landmark_face_idx.numel() > 0: # Check if not empty before min/max
                print(f"self.landmark_face_idx min: {self.landmark_face_idx.min()}, max: {self.landmark_face_idx.max()}") # Max should be < num_faces
                print(f"self.landmark_face_idx[:5]: {self.landmark_face_idx[:5]}")
            else:
                print("self.landmark_face_idx is empty.")

            print(f"self.landmark_b_coords shape: {self.landmark_b_coords.shape}")
            if self.landmark_b_coords.numel() > 0: # Check if not empty
                print(f"self.landmark_b_coords min: {self.landmark_b_coords.min()}, max: {self.landmark_b_coords.max()}") # Should be between 0 and 1
                print(f"self.landmark_b_coords[:5]:\n{self.landmark_b_coords[:5]}")
                print(f"Sum of barycentric coords for first 5 landmarks:\n{torch.sum(self.landmark_b_coords[:5], dim=1)}") # Should be close to 1.0
            else:
                print("self.landmark_b_coords is empty.")
            print("--- END DEBUG ---")


    def forward(self, shape_params=None, expression_params=None, pose_params=None, 
                  eye_pose_params=None, jaw_pose_params=None, neck_pose_params=None, transl=None, detail_params=None,
                  debug_print: bool = False):
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
        # Concatenate relevant pose parameters for LBS.
        # FLAME LBS typically uses global rotation, jaw, neck, and eye rotations.
        # The order and exact number of parameters depend on the LBS implementation.
        # Assuming pose_params (global), jaw_pose_params, neck_pose_params, eye_pose_params are axis-angle (3 params each).
        # Global pose from deconstruct_flame_coeffs is (B,6). If it's 6D log-matrix, it needs conversion to 3D axis-angle first for rodrigues.
        # This is a major simplification point. Assuming pose_params is (B,3) for global axis-angle for now.
        # If pose_params is (B,6) (e.g. from SMPL), it needs to be handled.
        # For FLAME, it's common to have global (3), jaw (3), neck (3), eyes (6). Total 15.
        # The deconstruction in train.py gives:
        # pose_params (global, 6), jaw (3), neck (3), eyes (6).
        # The LBS function expects axis-angle. If global_pose is 6D, it needs conversion.
        # For now, let's assume the first 3 of global_pose are axis-angle.
        
        # This is a placeholder for how poses are combined.
        # A proper FLAME layer would handle this internally based on its design.
        
        # The lbs function now takes individual pose parameters and handles conversions internally.
        
        # Calculate joint locations for LBS based on v_expressed (shaped + expressed vertices)
        # self.J_regressor is (num_lbs_joints, num_vertices)
        # v_expressed is (B, num_vertices, 3)
        # J_for_lbs_batched will be (B, num_lbs_joints, 3)
        J_for_lbs_batched = torch.einsum('JV,BVC->BJC', self.J_regressor, v_expressed)

        # --- DEBUGGING LBS: Temporarily neutralize non-global poses before passing to lbs ---
        # The lbs function itself also has internal debugging to force identity rotations
        # for non-global parts and zero posedirs. This ensures inputs are also zeroed out.
        # debug_neck_pose = torch.zeros_like(neck_pose_params) # Removed for training
        # debug_jaw_pose = torch.zeros_like(jaw_pose_params) # Removed for training
        # debug_eye_pose = torch.zeros_like(eye_pose_params) # Removed for training
        # --- END DEBUGGING LBS ---

        pred_verts_posed = lbs(
            v_shaped_expressed=v_expressed,
            global_pose_params_6d=pose_params,         # Keep this active
            neck_pose_params_ax=neck_pose_params,      # Use predicted neck pose
            jaw_pose_params_ax=jaw_pose_params,        # Use predicted jaw pose
            eye_pose_params_ax=eye_pose_params,        # Use predicted eye pose
            J_transformed_rest_lbs=J_for_lbs_batched,  # Pass LBS joint locations from v_expressed
            parents_lbs=self.parents_lbs,
            lbs_weights=self.lbs_weights,
            posedirs=self.posedirs, # lbs internal debugging will zero out posedirs effect
            dtype=self.v_template.dtype,
            debug_print=debug_print
        )
        
        # 4. Apply global translation
        if transl is not None:
            pred_verts = pred_verts_posed + transl.unsqueeze(1)
        else:
            pred_verts = pred_verts_posed

        # 5. Calculate 3D landmarks
        NUM_EXPECTED_LANDMARKS = 68 # Define for clarity within forward pass as well
        if self.using_barycentric_landmarks:
            # Get the vertices for each landmark face
            # pred_verts is (B, N_verts, 3)
            # self.faces_idx is (N_faces, 3) contains vertex indices for each triangle
            # self.landmark_face_idx is (N_landmarks=68,) contains indices of triangles

            landmark_triangles_verts_idx = self.faces_idx[self.landmark_face_idx] # (N_landmarks, 3)

            pred_landmarks_3d_list = []
            for b in range(batch_size):
                current_pred_verts = pred_verts[b] # (N_verts, 3)
                # Get the 3 vertices for each landmark triangle: (N_landmarks, 3, 3)
                tri_verts = current_pred_verts[landmark_triangles_verts_idx]
                # Interpolate using barycentric coordinates: (N_landmarks, 3)
                pred_landmarks_3d_sample = torch.einsum('ijk,ik->ij', tri_verts, self.landmark_b_coords)
                pred_landmarks_3d_list.append(pred_landmarks_3d_sample)
            pred_landmarks_3d = torch.stack(pred_landmarks_3d_list, dim=0) # (B, N_landmarks=68, 3)

        elif hasattr(self, 'landmark_vertex_ids') and self.landmark_vertex_ids.numel() > 0:
            pred_landmarks_3d = pred_verts[:, self.landmark_vertex_ids, :]
            # This slicing should already ensure 68 points if __init__ logic is correct
            if pred_landmarks_3d.shape[1] != NUM_EXPECTED_LANDMARKS:
                 print(f"Warning: Using 'landmark_vertex_ids' results in {pred_landmarks_3d.shape[1]} landmarks, "
                       f"but expected {NUM_EXPECTED_LANDMARKS}. This will cause a loss calculation error if not aligned with GT.")
                 # Fallback to a dummy tensor of correct shape if mismatch to prevent crash, though loss will be wrong.
                 # This case should ideally not be hit if __init__ correctly prepares landmark_vertex_ids.
                 pred_landmarks_3d = torch.zeros(batch_size, NUM_EXPECTED_LANDMARKS, 3, device=device)
        else:
            print("Error: No valid landmark definition (barycentric or vertex_ids) loaded in FLAME model for 68 points.")
            pred_landmarks_3d = torch.zeros(batch_size, NUM_EXPECTED_LANDMARKS, 3, device=device)
        
        return pred_verts, pred_landmarks_3d
