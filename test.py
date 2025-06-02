import numpy as np
landmark_data_path = './data/flame_model/mediapipe_landmark_embedding.npz'
landmark_data = np.load(landmark_data_path, allow_pickle=True)
print(f"Available keys: {list(landmark_data.keys())}")
if 'lmk_face_idx' in landmark_data:
    print(f"Shape of 'lmk_face_idx': {landmark_data['lmk_face_idx'].shape}") # Should be (68,) or (68, some_other_dim)
    print(f"Shape of 'lmk_b_coords': {landmark_data['lmk_b_coords'].shape}") # Should be (68, 3)

