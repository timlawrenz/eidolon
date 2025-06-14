import torch
import os
import sys
import numpy as np
import pickle

# --- Path Setup ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# --- Ensure REAL FLAME is imported ---
try:
    from src.model import FLAME
except ImportError:
    print("Error: src.model.FLAME not found. This script requires the actual FLAME model implementation.")
    print(f"Please ensure 'src/model.py' exists and FLAME class is defined, and that '{project_root}' is in PYTHONPATH if running from elsewhere.")
    sys.exit(1)

# --- .obj Saving Function ---
def save_obj(filepath, vertices, faces=None):
    assert vertices.ndim == 2 and vertices.shape[1] == 3, "Vertices must be of shape (N, 3)"
    if faces is not None:
        assert faces.ndim == 2 and faces.shape[1] == 3, "Faces must be of shape (F, 3)"
        assert faces.dtype == torch.long or faces.dtype == np.int64 or faces.dtype == np.int32 or faces.dtype == torch.int32, f"Faces dtype must be integer, got {faces.dtype}"

    with open(filepath, 'w') as f:
        for v_idx in range(vertices.shape[0]):
            v = vertices[v_idx]
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        if faces is not None:
            for face_idx in range(faces.shape[0]):
                face = faces[face_idx]
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    print(f"Saved to {filepath}")

def main():
    flame_model_path = 'data/flame_model/flame2023.pkl'
    # This is the path for the landmark embedding file, as referenced in the README.
    # The FLAME class __init__ may expect this argument as `deca_landmark_embedding_path`.
    # Using a standard 68-point landmark embedding, e.g., from DECA.
    landmark_embedding_file_path = 'data/flame_model/deca_landmark_embedding.npz'
    NUM_EXPECTED_LANDMARKS_SCRIPT = 68

    print("--- Inspecting Landmark Embedding File ---")
    print(f"Loading landmark embedding file from: {landmark_embedding_file_path}")

    # Dummy file creation is not necessary for this test. We need the real file.
    if not os.path.exists(landmark_embedding_file_path):
        print(f"Error: Landmark embedding file not found at {landmark_embedding_file_path}")
        print("Please acquire a 68-point FLAME landmark embedding, such as 'deca_landmark_embedding.npz' from the DECA project, and place it in that path.")
        sys.exit(1)

    try:
        if landmark_embedding_file_path.endswith('.pkl'):
            with open(landmark_embedding_file_path, 'rb') as f:
                data_embed = pickle.load(f, encoding='latin1')
        else: # For .npz, .npy
            data_embed = np.load(landmark_embedding_file_path, allow_pickle=True)
    except Exception as e:
        print(f"Error loading {landmark_embedding_file_path}: {e}")
        sys.exit(1)

    data_dict = None
    if isinstance(data_embed, np.lib.npyio.NpzFile):
        data_dict = data_embed
    elif isinstance(data_embed, np.ndarray) and data_embed.shape == ():
        data_dict = data_embed.item()
        if not isinstance(data_dict, dict):
            print(f"Error: Loaded .npy object from {landmark_embedding_file_path} did not contain a dict.")
            sys.exit(1)
    elif isinstance(data_embed, dict):
         data_dict = data_embed
    else:
        print(f"Error: Loaded landmark data not in expected dict-like format. Type: {type(data_embed)}")
        sys.exit(1)

    print(f"Available keys in landmark file: {list(data_dict.keys())}")

    keys_to_inspect = {
        'full_lmk_faces_idx': "DECA Face Indices",
        'full_lmk_bary_coords': "DECA Barycentric Coords",
        'lmk_face_idx': "MediaPipe Face Indices",
        'lmk_b_coords': "MediaPipe Barycentric Coords",
    }
    for key, desc in keys_to_inspect.items():
        if key in data_dict:
            data = data_dict[key]
            print(f"\n--- Stats for '{key}' ({desc}) ---")
            print(f"Shape: {data.shape}, Dtype: {data.dtype}")
            if data.size > 0:
                is_coord = "coords" in key.lower() or "b_coords" in key.lower()
                min_val, mean_val, max_val = np.min(data), np.mean(data), np.max(data)
                if is_coord:
                    print(f"Min: {min_val:.4f}, Mean: {mean_val:.4f}, Max: {max_val:.4f}")
                else:
                    print(f"Min: {min_val}, Mean: {mean_val:.4f}, Max: {max_val}")
                print(f"First 5 elements/rows:\n{data[:5]}")
        else:
            print(f"\nKey '{key}' not found in landmark file.")

    print("\n--- FLAME Model Initialization ---")
    n_shape = 100
    n_exp = 0

    if not os.path.exists(flame_model_path):
        print(f"Warning: FLAME model file not found at {flame_model_path}. Creating dummy.")
        os.makedirs(os.path.dirname(flame_model_path), exist_ok=True)
        # Ensure dummy model has v_template for landmark_indices validation in FLAME.__init__
        dummy_flame_pickle = {"v_template": np.random.rand(5023,3).astype(np.float32)}
        with open(flame_model_path, 'wb') as f: pickle.dump(dummy_flame_pickle, f)

    try:
        flame_model = FLAME(
            flame_model_path=flame_model_path,
            deca_landmark_embedding_path=landmark_embedding_file_path, # Corrected argument name
            n_shape=n_shape,
            n_exp=n_exp # Corrected argument name
        )
        print("FLAME model loaded successfully.")
    except Exception as e:
        print(f"An unexpected error occurred during FLAME model initialization: {e}")
        import traceback; traceback.print_exc(); sys.exit(1)

    shape_params = torch.zeros(1, n_shape)
    expression_params = torch.zeros(1, n_exp if n_exp > 0 else 0)
    pose_params = torch.zeros(1, 6, dtype=torch.float32)
    eye_pose_params = torch.zeros(1, 6)
    jaw_pose_params = torch.zeros(1, 3)
    neck_pose_params = torch.zeros(1, 3)
    transl = torch.zeros(1, 3)

    print("\nRunning FLAME forward pass...")
    try:
        pred_verts, pred_landmarks_3d = flame_model.forward(
            shape_params=shape_params, expression_params=expression_params, pose_params=pose_params,
            eye_pose_params=eye_pose_params, jaw_pose_params=jaw_pose_params,
            neck_pose_params=neck_pose_params, transl=transl, debug_print=True
        )
        print("FLAME forward pass completed.")
    except Exception as e:
        print(f"Error during FLAME forward pass: {e}"); import traceback; traceback.print_exc(); sys.exit(1)

    if pred_landmarks_3d is not None and pred_landmarks_3d.numel() > 0:
        lmk_sample = pred_landmarks_3d[0].detach().cpu()
        print(f"\n--- Predicted 3D Landmarks (Sample 0 from FLAME) ---")
        print(f"Shape: {lmk_sample.shape}")
        if lmk_sample.ndim == 2 and lmk_sample.shape[0] > 0 and lmk_sample.shape[1] == 3:
            for i, axis in enumerate(["X", "Y", "Z"]):
                print(f"{axis}: min={lmk_sample[:,i].min():.4f}, mean={lmk_sample[:,i].mean():.4f}, max={lmk_sample[:,i].max():.4f}")

    os.makedirs('output', exist_ok=True)
    verts_path = 'output/debug_flame_vertices.obj'
    lmks_path = 'output/debug_flame_landmarks.obj'
    if pred_verts is not None and pred_verts.numel() > 0:
        save_obj(verts_path, pred_verts[0].detach().cpu(), getattr(flame_model, 'faces_idx', None).cpu() if hasattr(flame_model, 'faces_idx') else None)
    if pred_landmarks_3d is not None and pred_landmarks_3d.ndim == 3 and pred_landmarks_3d.shape[2] == 3:
        save_obj(lmks_path, pred_landmarks_3d[0].detach().cpu())

    if hasattr(flame_model, 'using_barycentric_landmarks') and flame_model.using_barycentric_landmarks and \
       hasattr(flame_model, 'landmark_face_idx') and flame_model.landmark_face_idx.numel() > 0 :
        print("\n--- Manual Landmark 0 Calculation vs. FLAME Output (Barycentric) ---")
        # ... (Manual barycentric calculation logic - kept same as previous version, ensure attributes exist before access)
        try:
            faces_idx_np = flame_model.faces_idx.cpu().numpy().astype(np.int64)
            lmk_face_idx_np = flame_model.landmark_face_idx.cpu().numpy().astype(np.int64)
            lmk_b_coords_np = flame_model.landmark_b_coords.cpu().numpy()
            pred_verts_np = pred_verts[0].detach().cpu().numpy()
            idx_for_lmk0 = 0

            if not (0 <= idx_for_lmk0 < lmk_face_idx_np.shape[0] and 0 <= idx_for_lmk0 < lmk_b_coords_np.shape[0]):
                 print(f"ERROR: Landmark index {idx_for_lmk0} out of bounds for loaded landmark data.")
            else:
                face_for_lmk0 = lmk_face_idx_np[idx_for_lmk0]
                if not (0 <= face_for_lmk0 < faces_idx_np.shape[0]):
                    print(f"ERROR: face_for_lmk0 index {face_for_lmk0} is out of bounds for faces_idx_np.")
                else:
                    verts_indices_for_lmk0_face = faces_idx_np[face_for_lmk0]
                    if np.any(verts_indices_for_lmk0_face < 0) or np.any(verts_indices_for_lmk0_face >= pred_verts_np.shape[0]):
                        print(f"ERROR: Vertex indices {verts_indices_for_lmk0_face} are out of bounds for pred_verts_np.")
                    else:
                        triangle_vertices_for_lmk0 = pred_verts_np[verts_indices_for_lmk0_face]
                        b_coords_for_lmk0 = lmk_b_coords_np[idx_for_lmk0]
                        calculated_lmk0_pos_np = np.einsum('i,ij->j', b_coords_for_lmk0, triangle_vertices_for_lmk0)
                        flame_output_lmk0_pos_np = pred_landmarks_3d[0, idx_for_lmk0].detach().cpu().numpy()
                        diff = np.abs(calculated_lmk0_pos_np - flame_output_lmk0_pos_np)
                        print(f"Manually calculated Lmk0: {calculated_lmk0_pos_np}, FLAME Lmk0: {flame_output_lmk0_pos_np}, Diff Sum: {np.sum(diff):.6e}")
        except Exception as e: print(f"Error during manual barycentric check: {e}")

    elif hasattr(flame_model, 'landmark_vertex_ids') and flame_model.landmark_vertex_ids.numel() > 0:
        print("\n--- FLAME using Vertex ID based landmarks. Manual barycentric check skipped. ---")
        # You could add a simple check for vertex 0 if desired:
        # print(f"Lmk0 (Vertex ID): {flame_model.landmark_vertex_ids[0].item()}, Pos: {pred_verts[0, flame_model.landmark_vertex_ids[0].item()].detach().cpu().numpy()}")
        # print(f"FLAME output Lmk0: {pred_landmarks_3d[0,0].detach().cpu().numpy()}")

    print("\n--- Inspection ---") # ... (rest of inspection messages)

if __name__ == '__main__':
    main()
