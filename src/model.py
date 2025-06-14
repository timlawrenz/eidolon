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
    _joints = torch.unsqueeze(joints, dim=-1)
    _rel_joints = _joints.clone()
    if parents.shape[0] > 1:
        _rel_joints[:, 1:] -= _joints[:, parents[1:]]

    transforms_mat = transform_mat(
        rot_mats.reshape(-1, 3, 3),
        _rel_joints.reshape(-1, 3, 1)).reshape(-1, joints.shape[1], 4, 4)

    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, parents.shape[0]):
        curr_res = torch.matmul(transform_chain[parents[i]],
                                transforms_mat[:, i])
        transform_chain.append(curr_res)

    transforms = torch.stack(transform_chain, dim=1)
    posed_joints = transforms[:, :, :3, 3]
    R = transforms[:, :, :3, :3]
    t_posed = posed_joints.unsqueeze(-1)
    t = t_posed - torch.matmul(R, _joints)
    rel_transforms = transform_mat(R.reshape(-1, 3, 3), t.reshape(-1, 3, 1))
    rel_transforms = rel_transforms.reshape(-1, joints.shape[1], 4, 4)
    return posed_joints, rel_transforms


def lbs(v_shaped_expressed, 
        global_pose_params_6d,
        neck_pose_params_ax,
        jaw_pose_params_ax,
        eye_pose_params_ax,
        J_transformed_rest_lbs,
        parents_lbs, 
        lbs_weights, 
        posedirs, 
        dtype=torch.float32,
        debug_print: bool = False):
    if debug_print:
        print(f"--- ENTERING LBS FUNCTION (batch_size={v_shaped_expressed.shape[0]}) ---")
        # ... (rest of debug prints for lbs inputs)
    batch_size = v_shaped_expressed.shape[0]
    device = v_shaped_expressed.device

    if global_pose_params_6d.shape[1] == 6:
        global_rot_mat = rotation_6d_to_matrix(global_pose_params_6d)
    elif global_pose_params_6d.shape[1] == 3:
        global_rot_mat = batch_rodrigues(global_pose_params_6d)
    else:
        global_rot_mat = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).repeat(batch_size, 1, 1)

    neck_rot_mat = batch_rodrigues(neck_pose_params_ax)
    jaw_rot_mat = batch_rodrigues(jaw_pose_params_ax)
    eye_l_rot_mat = batch_rodrigues(eye_pose_params_ax[:, :3])
    eye_r_rot_mat = batch_rodrigues(eye_pose_params_ax[:, 3:])

    rot_mats_lbs = torch.stack([
        global_rot_mat, neck_rot_mat, jaw_rot_mat, eye_l_rot_mat, eye_r_rot_mat
    ], dim=1)

    _, A_global = batch_rigid_transform(rot_mats_lbs, J_transformed_rest_lbs, parents_lbs, dtype=dtype)
    
    num_joints_for_posedirs = 4
    rot_mats_subset_for_posedirs = rot_mats_lbs[:, 1:1+num_joints_for_posedirs, :, :]
    ident = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)
    pose_feature_vector_from_4_joints = (rot_mats_subset_for_posedirs - ident).view(batch_size, -1)
    num_features_expected_by_posedirs = posedirs.shape[1]
    current_pose_feature_vector = pose_feature_vector_from_4_joints
    if current_pose_feature_vector.shape[1] != num_features_expected_by_posedirs:
        current_pose_feature_vector = torch.zeros(batch_size, num_features_expected_by_posedirs, device=device, dtype=dtype)
            
    pose_blendshapes = torch.einsum('BP,VPC->BVC', current_pose_feature_vector, posedirs)
    v_to_skin = v_shaped_expressed + pose_blendshapes
    T = torch.einsum('VJ,BJHW->BVHW', lbs_weights, A_global)
    v_homo = torch.cat([v_to_skin, torch.ones(batch_size, v_to_skin.shape[1], 1, device=device, dtype=dtype)], dim=2)
    v_posed_lbs = torch.einsum('BVHW,BVW->BVH', T, v_homo)[:, :, :3]
    v_posed = v_posed_lbs
    # ... (rest of debug prints for lbs outputs)
    return v_posed


class EidolonEncoder(nn.Module):
    def __init__(self, num_coeffs):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT
        self.backbone = resnet50(weights=weights)
        for param in self.backbone.parameters():
            param.requires_grad = False
        num_bottleneck_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_bottleneck_features, num_coeffs)
        with torch.no_grad():
            n_shape = 100
            n_expr = 0
            n_global_pose = 6
            n_jaw_pose = 3
            n_eye_pose = 6
            n_neck_pose = 3
            self.backbone.fc.bias.fill_(0.0)
            current_idx = n_shape + n_expr
            if num_coeffs >= current_idx + n_global_pose:
                self.backbone.fc.bias[current_idx + 0] = 1.0
                self.backbone.fc.bias[current_idx + 4] = 1.0
    def forward(self, image):
        return self.backbone(image)

class FLAME(nn.Module):
    def __init__(self, flame_model_path, deca_landmark_embedding_path, n_shape, n_exp):
        super().__init__()

        with open(flame_model_path, 'rb') as f:
            flame_model_data = pickle.load(f, encoding='latin1')
        
        # ... (omitting existing debug prints for flame_model_data keys and expressedirs for brevity) ...
        
        v_template_data = flame_model_data['v_template']
        v_template_np = v_template_data.r if hasattr(v_template_data, 'r') else v_template_data
        self.register_buffer('v_template', torch.tensor(v_template_np, dtype=torch.float32))
        num_vertices = v_template_np.shape[0]

        # ... (omitting v_template debug prints) ...

        shapedirs_data = flame_model_data['shapedirs']
        shapedirs_np = shapedirs_data.r if hasattr(shapedirs_data, 'r') else shapedirs_data
        self.register_buffer('shapedirs', torch.tensor(shapedirs_np[:, :, :n_shape], dtype=torch.float32))
        
        if 'expressedirs' in flame_model_data and flame_model_data['expressedirs'] is not None:
             expressedirs_data = flame_model_data['expressedirs']
             expressedirs_np = expressedirs_data.r if hasattr(expressedirs_data, 'r') else expressedirs_data
             self.register_buffer('expressedirs', torch.tensor(expressedirs_np[:, :, :n_exp], dtype=torch.float32))
        else:
            self.register_buffer('expressedirs', torch.zeros((num_vertices, 3, n_exp), dtype=torch.float32, device=self.v_template.device))

        posedirs_data = flame_model_data['posedirs']
        posedirs_np = posedirs_data.r if hasattr(posedirs_data, 'r') else posedirs_data
        if posedirs_np.shape[1] == 3 and posedirs_np.shape[2] != 3:
            posedirs_permuted_np = np.transpose(posedirs_np, (0, 2, 1))
        else:
            posedirs_permuted_np = posedirs_np
        self.register_buffer('posedirs', torch.tensor(posedirs_permuted_np, dtype=torch.float32))
        
        j_regressor_data = flame_model_data['J_regressor']
        if hasattr(j_regressor_data, 'toarray'):
            j_regressor_np = j_regressor_data.toarray()
        elif hasattr(j_regressor_data, 'r'):
            j_regressor_np = j_regressor_data.r
        else:
            j_regressor_np = j_regressor_data
        self.register_buffer('J_regressor', torch.tensor(j_regressor_np, dtype=torch.float32))
        
        lbs_weights_data = flame_model_data['weights']
        lbs_weights_np = lbs_weights_data.r if hasattr(lbs_weights_data, 'r') else lbs_weights_data
        self.register_buffer('lbs_weights', torch.tensor(lbs_weights_np, dtype=torch.float32))
        
        faces_data = flame_model_data['f']
        faces_np = faces_data.astype(np.int64) if isinstance(faces_data, np.ndarray) else np.array(faces_data, dtype=np.int64)
        self.register_buffer('faces_idx', torch.tensor(faces_np, dtype=torch.long))
        
        if 'kintree_table' in flame_model_data:
             parents_full_np = flame_model_data['kintree_table'][0].astype(np.int64)
             parents_full_np[0] = -1
             self.register_buffer('parents_full_skeleton', torch.tensor(parents_full_np, dtype=torch.long))
        elif 'parent' in flame_model_data: 
             parents_full_np = flame_model_data['parent'].astype(np.int64)
             parents_full_np[0] = -1 
             self.register_buffer('parents_full_skeleton', torch.tensor(parents_full_np, dtype=torch.long))
        else:
            self.register_buffer('parents_full_skeleton', torch.empty(0, dtype=torch.long))

        num_lbs_joints = self.lbs_weights.shape[1]
        parents_lbs_np = self.parents_full_skeleton.cpu().numpy()[:num_lbs_joints]
        self.register_buffer('parents_lbs', torch.tensor(parents_lbs_np, dtype=torch.long))
        # ... (omitting LBS joint count asserts) ...

        # --- MODIFIED LANDMARK LOADING LOGIC ---
        self.using_barycentric_landmarks = False
        NUM_EXPECTED_LANDMARKS = 68
        deca_lmk_data = None # Initialize to ensure it's defined in case of early exception

        try:
            deca_lmk_data_container = np.load(deca_landmark_embedding_path, allow_pickle=True)
            if isinstance(deca_lmk_data_container, np.lib.npyio.NpzFile):
                print("DEBUG FLAME.__init__: Loaded landmark data is an NpzFile.")
                deca_lmk_data = deca_lmk_data_container
            elif isinstance(deca_lmk_data_container, np.ndarray) and deca_lmk_data_container.shape == ():
                print("DEBUG FLAME.__init__: Loaded landmark data is a 0-dim array, attempting .item().")
                deca_lmk_data = deca_lmk_data_container.item()
                if not isinstance(deca_lmk_data, dict):
                     raise ValueError(f"FLAME.__init__: Loaded .npy object from {deca_landmark_embedding_path} did not contain a dict.")
            elif isinstance(deca_lmk_data_container, dict):
                 print("DEBUG FLAME.__init__: Loaded landmark data is already a dictionary.")
                 deca_lmk_data = deca_lmk_data_container
            else:
                raise ValueError(f"FLAME.__init__: Landmark file {deca_landmark_embedding_path} loaded into unexpected type: {type(deca_lmk_data_container)}.")

            media_pipe_face_key = 'lmk_face_idx'
            media_pipe_bary_key = 'lmk_b_coords'
            deca_face_key = 'full_lmk_faces_idx'
            deca_bary_key = 'full_lmk_bary_coords'
            used_face_key = None
            used_bary_key = None

            if media_pipe_face_key in deca_lmk_data and media_pipe_bary_key in deca_lmk_data:
                print(f"DEBUG FLAME.__init__: Found MediaPipe landmark keys: '{media_pipe_face_key}', '{media_pipe_bary_key}'.")
                used_face_key = media_pipe_face_key
                used_bary_key = media_pipe_bary_key
            elif deca_face_key in deca_lmk_data and deca_bary_key in deca_lmk_data:
                print(f"DEBUG FLAME.__init__: MediaPipe keys not found. Found DECA landmark keys: '{deca_face_key}', '{deca_bary_key}'.")
                used_face_key = deca_face_key
                used_bary_key = deca_bary_key
            else:
                print(f"Warning FLAME.__init__: Neither MediaPipe nor DECA standard barycentric landmark keys found in {deca_landmark_embedding_path}.")

            if used_face_key and used_bary_key:
                lmk_faces_idx_68_np = deca_lmk_data[used_face_key]
                lmk_bary_coords_68_np = deca_lmk_data[used_bary_key]
                if lmk_faces_idx_68_np.ndim == 2 and lmk_faces_idx_68_np.shape[0] == 1:
                    lmk_faces_idx_68_np = lmk_faces_idx_68_np.squeeze(0)
                if lmk_bary_coords_68_np.ndim == 3 and lmk_bary_coords_68_np.shape[0] == 1:
                    lmk_bary_coords_68_np = lmk_bary_coords_68_np.squeeze(0)
                if lmk_faces_idx_68_np.ndim == 2 and lmk_faces_idx_68_np.shape[0] == 1 and lmk_faces_idx_68_np.shape[1] == NUM_EXPECTED_LANDMARKS:
                    lmk_faces_idx_68_np = lmk_faces_idx_68_np.squeeze(0)

                if lmk_faces_idx_68_np.shape == (NUM_EXPECTED_LANDMARKS,) and \
                   lmk_bary_coords_68_np.shape == (NUM_EXPECTED_LANDMARKS, 3):
                    self.register_buffer('landmark_face_idx', torch.tensor(lmk_faces_idx_68_np, dtype=torch.long))
                    self.register_buffer('landmark_b_coords', torch.tensor(lmk_bary_coords_68_np, dtype=torch.float32))
                    self.using_barycentric_landmarks = True
                    print(f"Successfully loaded {NUM_EXPECTED_LANDMARKS} barycentric landmarks using keys '{used_face_key}', '{used_bary_key}'.")
                else:
                    print(f"Warning FLAME.__init__: Barycentric landmark data (keys: '{used_face_key}', '{used_bary_key}') have unexpected shapes. Not using.")
        except Exception as e:
            print(f"ERROR loading or processing barycentric landmark data from {deca_landmark_embedding_path}: {e}")
        
        if not self.using_barycentric_landmarks:
            print("DEBUG FLAME.__init__: Barycentric landmark loading failed or criteria not met. Attempting vertex ID fallback.")
            vertex_ids_key = 'landmark_indices'
            if deca_lmk_data is not None and vertex_ids_key in deca_lmk_data:
                print(f"DEBUG FLAME.__init__: Found key '{vertex_ids_key}'. Attempting to use for landmarks.")
                landmark_vertex_ids_np = deca_lmk_data[vertex_ids_key]
                if isinstance(landmark_vertex_ids_np, np.ndarray):
                    if landmark_vertex_ids_np.ndim == 2 and landmark_vertex_ids_np.shape[0] == 1:
                        landmark_vertex_ids_np = landmark_vertex_ids_np.squeeze(0)
                    if landmark_vertex_ids_np.shape == (NUM_EXPECTED_LANDMARKS,):
                        if np.all(landmark_vertex_ids_np >= 0) and np.all(landmark_vertex_ids_np < num_vertices):
                            self.register_buffer('landmark_vertex_ids', torch.tensor(landmark_vertex_ids_np, dtype=torch.long))
                            print(f"Successfully loaded {NUM_EXPECTED_LANDMARKS} landmark vertex IDs using key '{vertex_ids_key}'.")
                            # Ensure barycentric attributes are empty if we successfully use vertex_ids
                            if hasattr(self, 'landmark_face_idx'): self.landmark_face_idx = torch.empty(0, dtype=torch.long, device=self.v_template.device)
                            if hasattr(self, 'landmark_b_coords'): self.landmark_b_coords = torch.empty(0, dtype=torch.float32, device=self.v_template.device)
                        else:
                            print(f"Warning FLAME.__init__: '{vertex_ids_key}' values out of range for model vertices.")
                    else:
                        print(f"Warning FLAME.__init__: '{vertex_ids_key}' shape is {landmark_vertex_ids_np.shape}, expected ({NUM_EXPECTED_LANDMARKS},).")
                else:
                     print(f"Warning FLAME.__init__: Data for '{vertex_ids_key}' is not a numpy array.")
            elif deca_lmk_data is not None:
                print(f"DEBUG FLAME.__init__: Fallback key '{vertex_ids_key}' not found. Available keys: {list(deca_lmk_data.keys())}")
            # else: deca_lmk_data was None, initial load must have failed.

        # Final status print and buffer initialization for safety
        if self.using_barycentric_landmarks:
            print("INFO FLAME.__init__: Using BARYCENTRIC landmarks.")
            # Ensure vertex_ids buffer is empty if barycentric is used
            if not hasattr(self, 'landmark_vertex_ids'): self.register_buffer('landmark_vertex_ids', torch.empty(0, dtype=torch.long))
            else: self.landmark_vertex_ids = torch.empty(0, dtype=torch.long, device=self.v_template.device)
        elif hasattr(self, 'landmark_vertex_ids') and self.landmark_vertex_ids.numel() == NUM_EXPECTED_LANDMARKS:
            print("INFO FLAME.__init__: Using VERTEX ID landmarks.")
            # Ensure barycentric buffers are empty
            if not hasattr(self, 'landmark_face_idx'): self.register_buffer('landmark_face_idx', torch.empty(0, dtype=torch.long))
            else: self.landmark_face_idx = torch.empty(0, dtype=torch.long, device=self.v_template.device)
            if not hasattr(self, 'landmark_b_coords'): self.register_buffer('landmark_b_coords', torch.empty(0, dtype=torch.float32))
            else: self.landmark_b_coords = torch.empty(0, dtype=torch.float32, device=self.v_template.device)
        else:
            print("CRITICAL WARNING FLAME.__init__: NO VALID 68-point LANDMARK DEFINITION LOADED (barycentric or vertex IDs).")
            if not hasattr(self, 'landmark_face_idx'): self.register_buffer('landmark_face_idx', torch.empty(0, dtype=torch.long))
            if not hasattr(self, 'landmark_b_coords'): self.register_buffer('landmark_b_coords', torch.empty(0, dtype=torch.float32))
            if not hasattr(self, 'landmark_vertex_ids'): self.register_buffer('landmark_vertex_ids', torch.empty(0, dtype=torch.long))
        # --- END OF MODIFIED LANDMARK LOADING ---

    def forward(self, shape_params=None, expression_params=None, pose_params=None, 
                  eye_pose_params=None, jaw_pose_params=None, neck_pose_params=None, transl=None, detail_params=None,
                  debug_print: bool = False):
        batch_size = shape_params.shape[0]
        device = shape_params.device
        shape_offset = torch.einsum('bS,VCS->bVC', shape_params, self.shapedirs).contiguous()
        v_shaped = self.v_template.unsqueeze(0).repeat(batch_size, 1, 1) + shape_offset
        expression_offset = torch.einsum('bE,VCE->bVC', expression_params, self.expressedirs).contiguous()
        v_expressed = v_shaped + expression_offset
        J_for_lbs_batched = torch.einsum('JV,BVC->BJC', self.J_regressor, v_expressed)
        pred_verts_posed = lbs(
            v_shaped_expressed=v_expressed,
            global_pose_params_6d=pose_params,
            neck_pose_params_ax=neck_pose_params,
            jaw_pose_params_ax=jaw_pose_params,
            eye_pose_params_ax=eye_pose_params,
            J_transformed_rest_lbs=J_for_lbs_batched,
            parents_lbs=self.parents_lbs,
            lbs_weights=self.lbs_weights,
            posedirs=self.posedirs,
            dtype=self.v_template.dtype,
            debug_print=debug_print
        )
        if transl is not None:
            pred_verts = pred_verts_posed + transl.unsqueeze(1)
        else:
            pred_verts = pred_verts_posed

        NUM_EXPECTED_LANDMARKS = 68
        if self.using_barycentric_landmarks and self.landmark_face_idx.numel() > 0: # Ensure buffers are not empty
            landmark_triangles_verts_idx = self.faces_idx[self.landmark_face_idx]
            pred_landmarks_3d_list = []
            for b in range(batch_size):
                current_pred_verts = pred_verts[b]
                tri_verts = current_pred_verts[landmark_triangles_verts_idx]
                pred_landmarks_3d_sample = torch.einsum('ijk,ik->ij', tri_verts, self.landmark_b_coords)
                pred_landmarks_3d_list.append(pred_landmarks_3d_sample)
            pred_landmarks_3d = torch.stack(pred_landmarks_3d_list, dim=0)
        elif hasattr(self, 'landmark_vertex_ids') and self.landmark_vertex_ids.numel() == NUM_EXPECTED_LANDMARKS:
            pred_landmarks_3d = pred_verts[:, self.landmark_vertex_ids, :]
        else: # Fallback if no valid landmark definition is found
            print("Error in FLAME.forward: No valid landmark definition (barycentric or vertex_ids) for 68 points. Returning zeros.")
            pred_landmarks_3d = torch.zeros(batch_size, NUM_EXPECTED_LANDMARKS, 3, device=device, dtype=pred_verts.dtype)
        
        return pred_verts, pred_landmarks_3d
