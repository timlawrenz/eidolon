import torch
import torch.nn as nn
import torch.nn.functional as F # Added for DECA's LBS components
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np
import pickle
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_axis_angle # For pose conversion

# --- LBS Helper Functions ---

def batch_rodrigues(rot_vecs, epsilon=1e-8):
    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device
    dtype = rot_vecs.dtype
    angle = torch.norm(rot_vecs + epsilon, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle
    cos = torch.cos(angle).unsqueeze(1)
    sin = torch.sin(angle).unsqueeze(1)
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))
    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mats = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mats

def transform_mat(R, t):
    return torch.cat([F.pad(R, [0, 0, 0, 1]),
                      F.pad(t, [0, 0, 0, 1], value=1)], dim=2)

def batch_rigid_transform(rot_mats, joints, parents, dtype=torch.float32):
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
        global_pose_params_6d, neck_pose_params_ax, jaw_pose_params_ax, eye_pose_params_ax,
        J_transformed_rest_lbs, parents_lbs, lbs_weights, posedirs,
        dtype=torch.float32, debug_print: bool = False):
    if debug_print:
        print(f"--- ENTERING LBS FUNCTION (batch_size={v_shaped_expressed.shape[0]}) ---")
        if v_shaped_expressed.numel() > 0 and v_shaped_expressed.shape[0] > 0:
            print(f"  DEBUG lbs input: v_shaped_expressed[0] Stats - Shape: {v_shaped_expressed.shape}")
            for i_axis, axis_name in enumerate(["X", "Y", "Z"]):
                print(f"    {axis_name}: mean={v_shaped_expressed[0, :, i_axis].mean().item():.4f}, std={v_shaped_expressed[0, :, i_axis].std().item():.4f}, "
                      f"min={v_shaped_expressed[0, :, i_axis].min().item():.4f}, max={v_shaped_expressed[0, :, i_axis].max().item():.4f}")
        else: print("  v_shaped_expressed is empty or has zero batch size.")
        print(f"----------------------------------------------------")

    batch_size = v_shaped_expressed.shape[0]
    device = v_shaped_expressed.device
    if global_pose_params_6d.shape[1] == 6:
        global_rot_mat = rotation_6d_to_matrix(global_pose_params_6d)
    elif global_pose_params_6d.shape[1] == 3:
        print("Warning: Global pose is 3D (axis-angle), converting to matrix.")
        global_rot_mat = batch_rodrigues(global_pose_params_6d)
    else:
        print(f"Warning: Global pose_params have unexpected shape {global_pose_params_6d.shape}. Using identity.")
        global_rot_mat = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).repeat(batch_size, 1, 1)
    neck_rot_mat = batch_rodrigues(neck_pose_params_ax)
    jaw_rot_mat = batch_rodrigues(jaw_pose_params_ax)
    eye_l_rot_mat = batch_rodrigues(eye_pose_params_ax[:, :3])
    eye_r_rot_mat = batch_rodrigues(eye_pose_params_ax[:, 3:])
    rot_mats_lbs = torch.stack([global_rot_mat, neck_rot_mat, jaw_rot_mat, eye_l_rot_mat, eye_r_rot_mat], dim=1)

    if debug_print:
        if J_transformed_rest_lbs.numel() > 0 and J_transformed_rest_lbs.shape[0] > 0:
            print(f"--- DEBUG lbs: J_transformed_rest_lbs[0] Stats - Shape: {J_transformed_rest_lbs.shape} ---")
        if rot_mats_lbs.numel() > 0 and rot_mats_lbs.shape[0] > 0:
             print(f"--- DEBUG lbs: rot_mats_lbs[0,0] (Global Rot Mat Sample 0) ---\n{rot_mats_lbs[0, 0, :, :]}")
        print(f"----------------------------------------------------")
    _, A_global = batch_rigid_transform(rot_mats_lbs, J_transformed_rest_lbs, parents_lbs, dtype=dtype)
    if debug_print:
        if A_global.numel() > 0 and A_global.shape[0] > 0:
            print(f"--- DEBUG lbs: A_global[0,0] (First Skinning Matrix Sample 0) ---\n{A_global[0, 0, :, :]}")
            if torch.isnan(A_global).any() or torch.isinf(A_global).any(): print("CRITICAL WARNING: NaNs or Infs found in A_global!")
        print(f"-------------------------------------------------")

    num_joints_for_posedirs = 4
    rot_mats_subset_for_posedirs = rot_mats_lbs[:, 1:1+num_joints_for_posedirs, :, :]
    ident = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)
    pose_feature_vector_from_4_joints = (rot_mats_subset_for_posedirs - ident).view(batch_size, -1)
    num_features_expected_by_posedirs = posedirs.shape[1]
    current_pose_feature_vector = pose_feature_vector_from_4_joints
    if current_pose_feature_vector.shape[1] != num_features_expected_by_posedirs:
        print(f"Warning: Pose feature vector shape mismatch. Using zeros for pose_blendshapes.")
        current_pose_feature_vector = torch.zeros(batch_size, num_features_expected_by_posedirs, device=device, dtype=dtype)
    pose_blendshapes = torch.einsum('BP,VPC->BVC', current_pose_feature_vector, posedirs)
    v_to_skin = v_shaped_expressed + pose_blendshapes
    if debug_print:
        if v_to_skin.numel() > 0 and v_to_skin.shape[0] > 0:
            print(f"--- DEBUG lbs: v_to_skin[0] Stats - Shape: {v_to_skin.shape} ---")
            if torch.isnan(v_to_skin).any() or torch.isinf(v_to_skin).any(): print("CRITICAL WARNING: NaNs or Infs found in v_to_skin!")
        print(f"---------------------------------------")
    T = torch.einsum('VJ,BJHW->BVHW', lbs_weights, A_global)
    v_homo = torch.cat([v_to_skin, torch.ones(batch_size, v_to_skin.shape[1], 1, device=device, dtype=dtype)], dim=2)
    v_posed_lbs = torch.einsum('BVHW,BVW->BVH', T, v_homo)[:, :, :3]
    v_posed = v_posed_lbs
    if debug_print:
        if v_posed.numel() > 0 and v_posed.shape[0] > 0:
            print(f"--- DEBUG lbs: v_posed[0] (output) Stats - Shape: {v_posed.shape} ---")
            if torch.isnan(v_posed).any() or torch.isinf(v_posed).any(): print("CRITICAL WARNING: NaNs or Infs found in v_posed output!")
        print(f"--------------------------------------------")
    return v_posed

class EidolonEncoder(nn.Module):
    def __init__(self, num_coeffs):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT
        self.backbone = resnet50(weights=weights)
        for param in self.backbone.parameters(): param.requires_grad = False
        num_bottleneck_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_bottleneck_features, num_coeffs)
        with torch.no_grad():
            n_shape, n_expr, n_global_pose, n_jaw_pose, n_eye_pose, n_neck_pose = 100, 0, 6, 3, 6, 3
            self.backbone.fc.bias.fill_(0.0)
            current_idx = n_shape + n_expr
            if num_coeffs >= current_idx + n_global_pose:
                self.backbone.fc.bias[current_idx + 0] = 1.0
                self.backbone.fc.bias[current_idx + 4] = 1.0
    def forward(self, image): return self.backbone(image)

class FLAME(nn.Module):
    def __init__(self, flame_model_path, deca_landmark_embedding_path, n_shape, n_exp):
        super().__init__()
        with open(flame_model_path, 'rb') as f: flame_model_data = pickle.load(f, encoding='latin1')
        v_template_data = flame_model_data['v_template']
        v_template_np = v_template_data.r if hasattr(v_template_data, 'r') else v_template_data
        self.register_buffer('v_template', torch.tensor(v_template_np, dtype=torch.float32))
        num_vertices = v_template_np.shape[0]
        shapedirs_data = flame_model_data['shapedirs']
        shapedirs_np = shapedirs_data.r if hasattr(shapedirs_data, 'r') else shapedirs_data
        self.register_buffer('shapedirs', torch.tensor(shapedirs_np[:, :, :n_shape], dtype=torch.float32))
        if 'expressedirs' in flame_model_data and flame_model_data['expressedirs'] is not None:
             expressedirs_data = flame_model_data['expressedirs']
             expressedirs_np = expressedirs_data.r if hasattr(expressedirs_data, 'r') else expressedirs_data
             self.register_buffer('expressedirs', torch.tensor(expressedirs_np[:, :, :n_exp], dtype=torch.float32))
        else:
            print("Warning: 'expressedirs' not found or is None. Expression parameters will have no effect.")
            self.register_buffer('expressedirs', torch.zeros((num_vertices, 3, n_exp), dtype=torch.float32, device=self.v_template.device))
        posedirs_data = flame_model_data['posedirs']
        posedirs_np = posedirs_data.r if hasattr(posedirs_data, 'r') else posedirs_data
        if posedirs_np.ndim == 3 and posedirs_np.shape[1] == 3 and posedirs_np.shape[2] != 3:
            posedirs_permuted_np = np.transpose(posedirs_np, (0, 2, 1))
        else: posedirs_permuted_np = posedirs_np
        self.register_buffer('posedirs', torch.tensor(posedirs_permuted_np, dtype=torch.float32))
        j_regressor_data = flame_model_data['J_regressor']
        j_regressor_np = j_regressor_data.toarray() if hasattr(j_regressor_data, 'toarray') else (j_regressor_data.r if hasattr(j_regressor_data, 'r') else j_regressor_data)
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
            print("Warning: Full skeleton parent information not found.")
            self.register_buffer('parents_full_skeleton', torch.empty(0, dtype=torch.long))
        num_lbs_joints = self.lbs_weights.shape[1]
        parents_lbs_np = self.parents_full_skeleton.cpu().numpy()[:num_lbs_joints] if self.parents_full_skeleton.numel() >= num_lbs_joints else np.array([-1,0,0,0,0],dtype=np.int64)
        self.register_buffer('parents_lbs', torch.tensor(parents_lbs_np, dtype=torch.long))

        self.using_barycentric_landmarks = False
        NUM_EXPECTED_LANDMARKS = 68
        try:
            deca_lmk_data = None
            if deca_landmark_embedding_path.endswith('.pkl'):
                with open(deca_landmark_embedding_path, 'rb') as f:
                    deca_lmk_data = pickle.load(f, encoding='latin1')
                print("DEBUG FLAME.__init__: Loaded landmark data from .pkl file.")
            else: # Fallback for .npz, .npy
                deca_lmk_data_container = np.load(deca_landmark_embedding_path, allow_pickle=True)
                if isinstance(deca_lmk_data_container, np.lib.npyio.NpzFile):
                    print("DEBUG FLAME.__init__: Loaded landmark data is an NpzFile.")
                    deca_lmk_data = deca_lmk_data_container
                elif isinstance(deca_lmk_data_container, np.ndarray) and deca_lmk_data_container.shape == ():
                    print("DEBUG FLAME.__init__: Loaded landmark data is a 0-dim array, attempting .item().")
                    deca_lmk_data = deca_lmk_data_container.item()
                elif isinstance(deca_lmk_data_container, dict):
                     print("DEBUG FLAME.__init__: Loaded landmark data is already a dictionary.")
                     deca_lmk_data = deca_lmk_data_container
                else:
                    raise ValueError(f"FLAME.__init__: Landmark file {deca_landmark_embedding_path} loaded into unexpected type: {type(deca_lmk_data_container)}.")
            
            if not isinstance(deca_lmk_data, dict):
                raise ValueError(f"FLAME.__init__: Landmark data from {deca_landmark_embedding_path} did not resolve to a dictionary.")

            # Check for different landmark embedding standards (e.g., DECA, MediaPipe)
            deca_face_key, deca_bary_key = 'full_lmk_faces_idx', 'full_lmk_bary_coords'
            mediapipe_face_key, mediapipe_bary_key = 'lmk_face_idx', 'lmk_b_coords'
            used_face_key, used_bary_key = None, None

            if deca_face_key in deca_lmk_data and deca_bary_key in deca_lmk_data:
                print(f"DEBUG FLAME.__init__: Found DECA landmark keys: '{deca_face_key}', '{deca_bary_key}'.")
                used_face_key, used_bary_key = deca_face_key, deca_bary_key
            elif mediapipe_face_key in deca_lmk_data and mediapipe_bary_key in deca_lmk_data:
                print(f"DEBUG FLAME.__init__: DECA keys not found. Found MediaPipe keys: '{mediapipe_face_key}', '{mediapipe_bary_key}'.")
                used_face_key, used_bary_key = mediapipe_face_key, mediapipe_bary_key
            
            if used_face_key and used_bary_key:
                lmk_faces_idx_np = deca_lmk_data[used_face_key]
                lmk_bary_coords_np = deca_lmk_data[used_bary_key]
                if lmk_faces_idx_np.ndim == 2 and lmk_faces_idx_np.shape[0] == 1: lmk_faces_idx_np = lmk_faces_idx_np.squeeze(0)
                if lmk_bary_coords_np.ndim == 3 and lmk_bary_coords_np.shape[0] == 1: lmk_bary_coords_np = lmk_bary_coords_np.squeeze(0)

                if lmk_faces_idx_np.shape == (NUM_EXPECTED_LANDMARKS,) and lmk_bary_coords_np.shape == (NUM_EXPECTED_LANDMARKS, 3):
                    self.register_buffer('landmark_face_idx', torch.tensor(lmk_faces_idx_np, dtype=torch.long))
                    self.register_buffer('landmark_b_coords', torch.tensor(lmk_bary_coords_np, dtype=torch.float32))
                    self.using_barycentric_landmarks = True
                    print(f"Successfully loaded {NUM_EXPECTED_LANDMARKS} barycentric landmarks using keys '{used_face_key}', '{used_bary_key}'.")
                else:
                    print(f"Warning FLAME.__init__: Landmark data (keys '{used_face_key}', '{used_bary_key}') have shapes {lmk_faces_idx_np.shape} and {lmk_bary_coords_np.shape},"
                          f" but expected ({NUM_EXPECTED_LANDMARKS},) and ({NUM_EXPECTED_LANDMARKS}, 3).")
            else:
                print(f"Warning FLAME.__init__: Neither DECA nor MediaPipe standard landmark keys found in {deca_landmark_embedding_path}. Keys: {list(deca_lmk_data.keys())}")
        except Exception as e:
            print(f"ERROR loading or processing landmark embedding from {deca_landmark_embedding_path}: {e}")
            import traceback; traceback.print_exc()

        # The fallback to 'landmark_indices' is no longer needed as we are using a known-good static embedding.

        if not self.using_barycentric_landmarks:
            print(f"Critical Warning: Final check - Could not load {NUM_EXPECTED_LANDMARKS}-point landmarks.")
            # Ensure buffers exist even if loading fails, to prevent errors in forward pass.
            if not hasattr(self, 'landmark_face_idx'): self.register_buffer('landmark_face_idx', torch.empty(0, dtype=torch.long))
            if not hasattr(self, 'landmark_b_coords'): self.register_buffer('landmark_b_coords', torch.empty(0, dtype=torch.float32))
        # We no longer use landmark_vertex_ids, so no need for an else-if here.


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
            v_shaped_expressed=v_expressed, global_pose_params_6d=pose_params,
            neck_pose_params_ax=neck_pose_params, jaw_pose_params_ax=jaw_pose_params,
            eye_pose_params_ax=eye_pose_params, J_transformed_rest_lbs=J_for_lbs_batched,
            parents_lbs=self.parents_lbs, lbs_weights=self.lbs_weights,
            posedirs=self.posedirs, dtype=self.v_template.dtype, debug_print=debug_print
        )
        if transl is not None: pred_verts = pred_verts_posed + transl.unsqueeze(1)
        else: pred_verts = pred_verts_posed

        NUM_EXPECTED_LANDMARKS = 68
        if self.using_barycentric_landmarks and self.landmark_face_idx.numel() == NUM_EXPECTED_LANDMARKS and self.landmark_b_coords.numel() == NUM_EXPECTED_LANDMARKS * 3:
            landmark_triangles_verts_idx = self.faces_idx[self.landmark_face_idx]
            pred_landmarks_3d_list = []
            for b in range(batch_size):
                current_pred_verts = pred_verts[b]
                tri_verts = current_pred_verts[landmark_triangles_verts_idx]
                pred_landmarks_3d_sample = torch.einsum('ijk,ik->ij', tri_verts, self.landmark_b_coords)
                pred_landmarks_3d_list.append(pred_landmarks_3d_sample)
            pred_landmarks_3d = torch.stack(pred_landmarks_3d_list, dim=0)
        else:
            print(f"Error: No valid {NUM_EXPECTED_LANDMARKS}-point landmark definition available in FLAME model.")
            pred_landmarks_3d = torch.zeros(batch_size, NUM_EXPECTED_LANDMARKS, 3, device=device)
        return pred_verts, pred_landmarks_3d
