import numpy as np

try:
    # Update this path to where you saved the downloaded file
    deca_landmark_path = './data/flame_model/deca_landmark_embedding.npy' 
    landmark_data = np.load(deca_landmark_path, allow_pickle=True) # Added allow_pickle=True

    print(f"Successfully loaded: {deca_landmark_path}")
    print(f"Raw loaded data type: {type(landmark_data)}")
    print(f"Raw loaded data shape: {landmark_data.shape}")
    print(f"Raw loaded data dtype: {landmark_data.dtype}")

    if landmark_data.shape == () and landmark_data.dtype == object:
        print("\nFile contains a scalar object. Extracting item...")
        actual_data = landmark_data.item()
        print(f"Extracted item type: {type(actual_data)}")

        if isinstance(actual_data, dict):
            print("Extracted item is a dictionary. Available keys:", list(actual_data.keys()))
            
            key_to_check = None
            found_68_vertex_key = None

            # Check for barycentric coordinates first
            if 'lmk_face_idx' in actual_data and 'lmk_b_coords' in actual_data:
                print("\nFound 'lmk_face_idx' and 'lmk_b_coords'. Inspecting for 68-point barycentric landmarks...")
                lmk_face_idx = actual_data['lmk_face_idx']
                lmk_b_coords = actual_data['lmk_b_coords']
                
                if isinstance(lmk_face_idx, np.ndarray) and isinstance(lmk_b_coords, np.ndarray):
                    print(f"  Shape of 'lmk_face_idx': {lmk_face_idx.shape}")
                    print(f"  Shape of 'lmk_b_coords': {lmk_b_coords.shape}")
                    if lmk_face_idx.shape == (68,) and lmk_b_coords.shape == (68, 3):
                        print("  SUCCESS! 'lmk_face_idx' and 'lmk_b_coords' seem to define 68 barycentric landmarks.")
                    else:
                        print("  Shapes for 'lmk_face_idx'/'lmk_b_coords' do not match (68,) and (68,3).")
                else:
                    print("  'lmk_face_idx' or 'lmk_b_coords' are not NumPy arrays.")

            # Search for direct 68-point vertex indices
            print("\nSearching for direct 68-point vertex indices in the extracted dictionary...")
            possible_vertex_id_keys = ['static_landmark_vertex_ids', 'vertex_ids_68', 'landmark_vertex_ids', 'landmark_indices', 'lmk_idx', 'ids', 'v_idx_68']

            for k in possible_vertex_id_keys:
                if k in actual_data:
                    print(f"  Inspecting potential key: '{k}'")
                    try:
                        data_array = np.array(actual_data[k])
                        print(f"    Shape of data['{k}']: {data_array.shape}")
                        if data_array.ndim == 1 and data_array.shape[0] == 68:
                            print(f"    SUCCESS! Key '{k}' looks like 68 landmark vertex indices!")
                            found_68_vertex_key = k
                            break 
                    except Exception as e_shape:
                        print(f"    Could not get shape for key '{k}': {e_shape}")
            
            if found_68_vertex_key:
                landmark_indices = np.array(actual_data[found_68_vertex_key])
                print(f"\nUsing key '{found_68_vertex_key}' for 68 vertex indices.")
                print(f"  Min index value: {np.min(landmark_indices)}")
                print(f"  Max index value: {np.max(landmark_indices)}")
                print(f"  First 5 indices: {landmark_indices[:5]}")
                if np.issubdtype(landmark_indices.dtype, np.integer):
                    print("  Data type is integer (good).")
                else:
                    print(f"  Warning: Data type is {landmark_indices.dtype}. Expected integer.")
            else:
                print("\nNo direct 68-point vertex index key found among common names in the dictionary.")
                print("Please inspect all dictionary keys and their shapes manually if needed:")
                for k_dict, v_dict in actual_data.items():
                    if isinstance(v_dict, np.ndarray):
                        print(f"  Dict Key: '{k_dict}', Shape: {v_dict.shape}, Dtype: {v_dict.dtype}")
                    else:
                        print(f"  Dict Key: '{k_dict}', Type: {type(v_dict)}")
        
        elif isinstance(actual_data, np.ndarray): # If the item itself is a NumPy array
            print("Extracted item is a NumPy array.")
            print(f"  Shape: {actual_data.shape}")
            print(f"  Dtype: {actual_data.dtype}")
            if actual_data.ndim == 1 and actual_data.shape[0] == 68:
                print("  SUCCESS! This NumPy array (the extracted item) looks like 68 landmark vertex indices.")
                # Perform sanity checks as above
            else:
                print("  This NumPy array (the extracted item) does not have shape (68,).")
        else:
            print(f"Extracted item is of type: {type(actual_data)}. Please inspect manually.")

    elif landmark_data.ndim == 1 and landmark_data.shape[0] == 68: # Original check if it was a direct array
        print("\nSUCCESS! This looks like a 1D array of 68 landmark vertex indices.")
        print(f"Min index value: {np.min(landmark_data)}")
        print(f"Max index value: {np.max(landmark_data)}")
        print(f"First 5 indices: {landmark_data[:5]}")
        if np.issubdtype(landmark_data.dtype, np.integer):
            print("Element data type is integer, which is expected for vertex indices.")
        else:
            print(f"Warning: Element data type is {landmark_data.dtype}. Expected integer for vertex indices.")
    else:
        print("\nThis file does not appear to be a direct 1D array of 68 indices, nor a scalar object containing a dictionary or suitable array. Further inspection needed.")

except FileNotFoundError:
    print(f"ERROR: File not found at {deca_landmark_path}. Please check the path.")
except Exception as e:
    print(f"An error occurred: {e}")

