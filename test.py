import pickle
import numpy as np

# Path to the new static embedding pickle file
static_embedding_path = './data/flame_model/flame_static_embedding.pkl' 

try:
    with open(static_embedding_path, 'rb') as f:
        static_embedding_data = pickle.load(f, encoding='latin1')

    print(f"Successfully loaded {static_embedding_path}")

    # --- Inspection Steps ---
    if isinstance(static_embedding_data, dict):
        print("\nFile contains a dictionary. Available keys:", list(static_embedding_data.keys()))
        
        key_to_check = None
        
        # Check for barycentric coordinates first (preferred for 68 landmarks)
        if 'lmk_face_idx' in static_embedding_data and 'lmk_b_coords' in static_embedding_data:
            print("\nFound 'lmk_face_idx' and 'lmk_b_coords'. Inspecting for 68-point barycentric landmarks...")
            lmk_face_idx = static_embedding_data['lmk_face_idx']
            lmk_b_coords = static_embedding_data['lmk_b_coords']
            
            if isinstance(lmk_face_idx, np.ndarray) and isinstance(lmk_b_coords, np.ndarray):
                print(f"  Shape of 'lmk_face_idx': {lmk_face_idx.shape}")
                print(f"  Shape of 'lmk_b_coords': {lmk_b_coords.shape}")
                if lmk_face_idx.shape == (68,) and lmk_b_coords.shape == (68, 3):
                    print("  SUCCESS! 'lmk_face_idx' and 'lmk_b_coords' seem to define 68 barycentric landmarks.")
                    # You would use these in your FLAME model
                else:
                    print("  Shapes for 'lmk_face_idx'/'lmk_b_coords' do not match (68,) and (68,3).")
            else:
                print("  'lmk_face_idx' or 'lmk_b_coords' are not NumPy arrays.")

        # If barycentric not found or not matching, look for direct vertex indices for 68 landmarks
        print("\nSearching for direct 68-point vertex indices...")
        possible_vertex_id_keys = ['static_landmark_vertex_ids', 'vertex_ids_68', 'landmark_vertex_ids', 'landmark_indices']
        found_68_vertex_key = None

        for k in possible_vertex_id_keys:
            if k in static_embedding_data:
                print(f"  Inspecting potential key: '{k}'")
                try:
                    data_array = np.array(static_embedding_data[k])
                    print(f"    Shape of data['{k}']: {data_array.shape}")
                    if data_array.shape == (68,):
                        print(f"    SUCCESS! Key '{k}' looks like 68 landmark vertex indices!")
                        found_68_vertex_key = k
                        break 
                except Exception as e_shape:
                    print(f"    Could not get shape for key '{k}': {e_shape}")
        
        if found_68_vertex_key:
            landmark_indices = np.array(static_embedding_data[found_68_vertex_key])
            print(f"\nUsing key '{found_68_vertex_key}' for 68 vertex indices.")
            print(f"  First 5 indices: {landmark_indices[:5]}")
            # Further checks:
            if np.issubdtype(landmark_indices.dtype, np.integer):
                print("  Data type is integer (good).")
            else:
                print("  Warning: Data type is not integer.")
            if landmark_indices.min() >= 0 and landmark_indices.max() < 5023: # Assuming ~5023 FLAME vertices
                print("  Indices are within typical FLAME vertex range (good).")
            else:
                print(f"  Warning: Indices range ({landmark_indices.min()}-{landmark_indices.max()}) might be outside typical FLAME vertex range.")
        elif not ('lmk_face_idx' in static_embedding_data and static_embedding_data['lmk_face_idx'].shape == (68,)): # if barycentric wasn't found either
            print("\nNo direct 68-point vertex index key found among common names.")
            print("Please inspect all keys and their shapes manually if needed:")
            for k, v in static_embedding_data.items():
                if isinstance(v, np.ndarray):
                    print(f"  Key: '{k}', Shape: {v.shape}, Dtype: {v.dtype}")
                else:
                    print(f"  Key: '{k}', Type: {type(v)}")


    elif isinstance(static_embedding_data, np.ndarray):
        print(f"\nFile directly contains a NumPy array with shape: {static_embedding_data.shape}")
        if static_embedding_data.shape == (68,):
            print("SUCCESS! This NumPy array could be the 68 landmark indices!")
            landmark_indices = static_embedding_data
            # Perform similar checks as above for dtype and range
        else:
            print("Shape is not (68,). This might be something else.")
    else:
        print(f"\nFile contains data of type: {type(static_embedding_data)}. Please inspect manually.")

except FileNotFoundError:
    print(f"ERROR: {static_embedding_path} not found. Check your path.")
except Exception as e:
    print(f"An error occurred: {e}")

