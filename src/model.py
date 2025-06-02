
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np
import pickle

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
    # TODO: Implement the full kinematic chain transformation.
    # This involves iterating through joints from root to leaves,
    # composing local transformations with parent's global transformation.
    #
    # For now, returning identity transformations for each joint.
    # This means no actual posing will occur from LBS joint rotations.
    batch_size = rot_mats.shape[0]
    num_joints = joints.shape[1]
    device = rot_mats.device

    # Placeholder: Return identity transformations
    A_global = torch.eye(4, dtype=dtype, device=device).unsqueeze(0).unsqueeze(0).repeat(batch_size, num_joints, 1, 1)
    # To make it a valid (though incorrect) transformation, set translation part from joints
    A_global[:, :, :3, 3] = joints 
    return A_global


def lbs(v_shaped_expressed, pose_params_all, J_regressor, parents, lbs_weights, posedirs, dtype=torch.float32):
    """
    Performs Linear Blend Skinning (LBS).
    Args:
        v_shaped_expressed (torch.Tensor): Vertices after shape and expression (B, N_verts, 3).
        pose_params_all (torch.Tensor): Concatenated pose parameters for all relevant joints (B, J*3).
                                     (e.g., global, jaw, neck, etc. as axis-angle)
        J_regressor (torch.Tensor): Joint regressor matrix.
        parents (torch.Tensor): Parent indices for each joint.
        lbs_weights (torch.Tensor): LBS weights.
        posedirs (torch.Tensor): Pose-dependent blendshapes.
    Returns:
        v_posed (torch.Tensor): Posed vertices.
    """
    batch_size = v_shaped_expressed.shape[0]
    device = v_shaped_expressed.device

    # 1. Calculate initial joint locations J from v_shaped_expressed
    # J_regressor: (num_joints, num_vertices)
    # v_shaped_expressed: (B, num_vertices, 3)
    # J: (B, num_joints, 3)
    J = torch.einsum('JV,BVC->BJC', J_regressor, v_shaped_expressed)

    # 2. Convert pose_params_all to rotation matrices
    # pose_params_all is (B, num_total_pose_coeffs), e.g., (B, 18) for 6 joints (global, jaw, neck, 2 eyes)
    # Reshape to (B * num_joints_posed, 3) for batch_rodrigues
    # num_joints_posed should correspond to the joints whose rotations are in pose_params_all
    # For FLAME, this typically includes global, jaw, neck, and eyes.
    # Let's assume pose_params_all is (B, num_flame_main_joints * 3)
    num_flame_main_joints = lbs_weights.shape[1] # Should be 5 for FLAME (global, neck, jaw, L_eye, R_eye)
    
    # Ensure pose_params_all has the correct number of parameters for these joints
    # Example: if 5 joints, then 5*3=15 pose parameters.
    # The current deconstruction in train.py gives 18 (global=6, jaw=3, neck=3, eyes=6).
    # This needs careful alignment. For now, assume pose_params_all is (B, num_flame_main_joints, 3)
    # This part requires careful handling of how pose_params from encoder map to FLAME joints.
    # For this placeholder, we'll assume pose_params_all is correctly shaped (B, num_flame_main_joints, 3)
    # If pose_params_all is (B, 18), and we need (B,5,3) for 5 joints, a split/map is needed.
    # For now, let's take the first 5*3=15 parameters if pose_params_all is longer.
    
    if pose_params_all.shape[1] > num_flame_main_joints * 3:
        # This is a temporary hack if pose_params_all has more than needed for LBS joints
        # e.g. if global pose is 6D (rot_mat_log) but rodrigues needs 3D axis-angle
        # Or if eye poses are included but not used for mesh LBS in the same way.
        # For FLAME, global (3), neck (3), jaw (3) are primary for LBS. Eyes are separate.
        # Let's assume pose_params_all contains [global_rot(3), neck_rot(3), jaw_rot(3), eye_L(3), eye_R(3)]
        # Total 15 parameters for 5 joints.
        # If train.py provides 18 (e.g. 6D global), this needs adjustment.
        # For now, assuming it's (B, 15) and reshaping to (B, 5, 3)
        # This is a MAJOR simplification/placeholder.
        pose_params_for_rodrigues = pose_params_all[:, :(num_flame_main_joints * 3)].reshape(batch_size * num_flame_main_joints, 3)
    elif pose_params_all.shape[1] == num_flame_main_joints * 3:
        pose_params_for_rodrigues = pose_params_all.reshape(batch_size * num_flame_main_joints, 3)
    else:
        # Fallback if dimensions don't match, LBS will be effectively identity
        print(f"Warning: pose_params_all shape {pose_params_all.shape} not compatible with {num_flame_main_joints} LBS joints. Using identity rotations for LBS.")
        pose_params_for_rodrigues = torch.zeros(batch_size * num_flame_main_joints, 3, device=device, dtype=dtype)

    rot_mats = batch_rodrigues(pose_params_for_rodrigues)
    rot_mats = rot_mats.view(batch_size, num_flame_main_joints, 3, 3)

    # 3. Get global joint transformations A_global (B, J, 4, 4)
    A_global = batch_rigid_transform(rot_mats, J, parents, dtype=dtype)
    # For FLAME, J_transformed (from J_regressor) are the rest pose joints.
    # The pose is applied to these.

    # 4. Transform vertices by LBS
    # lbs_weights: (N_verts, num_joints)
    # A_global: (B, num_joints, 4, 4)
    # T: (B, N_verts, 4, 4)
    T = torch.einsum('VJ,BJHW->BVHW', lbs_weights, A_global)

    v_homo = torch.cat([v_shaped_expressed, torch.ones(batch_size, v_shaped_expressed.shape[1], 1, device=device, dtype=dtype)], dim=2)
    v_posed_lbs = torch.einsum('BVHW,BVW->BVH', T, v_homo)[:, :, :3] # Keep only x,y,z

    # 5. Add pose-corrective blendshapes (posedirs)
    # posedirs: (N_verts, 3, num_pose_blendshape_coeffs)
    # num_pose_blendshape_coeffs is typically (num_joints_for_posedirs - 1) * 9
    # For FLAME, posedirs are (5023, 3, 27) -> 3 joints * 9 components each (e.g. jaw, neck, global effects)
    # We need a pose_feature_vector from rot_mats (excluding identity for root)
    # This is highly model-specific. For now, a placeholder.
    # TODO: Create correct pose_feature_vector for FLAME's posedirs.
    # This vector should be (B, num_pose_blendshape_coeffs), e.g. (B, 27) for FLAME.
    # It's derived from (rot_mats - Identity) of relevant joints.
    num_pose_blendshape_coeffs = posedirs.shape[2]
    pose_feature_vector = torch.zeros(batch_size, num_pose_blendshape_coeffs, device=device, dtype=dtype) # Placeholder

    pose_blendshapes = torch.einsum('BP,VCP->BVC', pose_feature_vector, posedirs)
    
    v_posed = v_posed_lbs + pose_blendshapes
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
    def __init__(self, flame_model_path, deca_landmark_embedding_path, n_shape, n_exp): # Changed parameter name
        super().__init__()

        with open(flame_model_path, 'rb') as f:
            flame_model_data = pickle.load(f, encoding='latin1')

        # FLAME components
        # Conditionally access .r to get NumPy arrays from potential chumpy objects
        
        v_template_data = flame_model_data['v_template']
        v_template_np = v_template_data.r if hasattr(v_template_data, 'r') else v_template_data
        self.register_buffer('v_template', torch.tensor(v_template_np, dtype=torch.float32))
        num_vertices = v_template_np.shape[0]

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
        self.register_buffer('posedirs', torch.tensor(posedirs_np, dtype=torch.float32)) # Use all pose blendshapes
        
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
        # For now, we'll take the first 3 of global_pose, and then jaw, neck, and eyes.
        # This makes 3 (global) + 3 (jaw) + 3 (neck) + 6 (eyes) = 15 parameters for 5 joints.
        # This part is highly dependent on the specific LBS implementation details.
        if pose_params.shape[1] == 6: # Assuming 6D global pose, take first 3 as simplified axis-angle
            print("Warning: Using first 3 params of 6D global_pose as axis-angle for LBS placeholder.")
            global_pose_axis_angle = pose_params[:, :3]
        else: # Assuming it's already 3D axis-angle
            global_pose_axis_angle = pose_params

        # This forms a (B, 15) tensor, then reshaped to (B, 5, 3) inside lbs if needed.
        # This order (global, neck, jaw, eyeL, eyeR) is common for FLAME lbs_weights.
        # The eye rotations primarily affect eyeballs, not the facial mesh via LBS weights/posedirs.
        # For facial mesh LBS, usually global, neck, jaw are key.
        # The `lbs_weights` has 5 columns, so 5 joints.
        # The `posedirs` has 27 columns, so (3 joints - 1)*9 or similar.
        # This indicates a mismatch or simplification in the current placeholder.
        
        # For the LBS function, we need to provide pose parameters for the joints
        # that lbs_weights and posedirs correspond to.
        # Let's assume pose_params_all for LBS should be (B, num_lbs_joints * 3)
        # For FLAME, lbs_weights has 5 joints. So, (B, 15).
        # Global (3), Neck (3), Jaw (3), EyeL (3), EyeR (3)
        # Eye poses are split: eye_pose_params is (B,6) -> eye_L (B,3), eye_R (B,3)
        
        pose_params_for_lbs = torch.cat([
            global_pose_axis_angle, # (B,3)
            neck_pose_params,       # (B,3)
            jaw_pose_params,        # (B,3)
            eye_pose_params[:, :3], # Left eye (B,3)
            eye_pose_params[:, 3:]  # Right eye (B,3)
        ], dim=1) # Results in (B, 15)

        pred_verts_posed = lbs(v_expressed, pose_params_for_lbs,
                               self.J_regressor, self.parents, self.lbs_weights, self.posedirs,
                               dtype=self.v_template.dtype)
        
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
