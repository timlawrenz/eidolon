import numpy as np

try:
    # Update this path to where you saved the downloaded file
    deca_landmark_path = './data/flame_model/deca_landmark_embedding.npy' 
    landmark_data = np.load(deca_landmark_path)

    print(f"Successfully loaded: {deca_landmark_path}")
    print(f"Data type: {type(landmark_data)}")
    print(f"Data shape: {landmark_data.shape}")
    print(f"Data dtype: {landmark_data.dtype}") # Should ideally be integer

    if landmark_data.ndim == 1 and landmark_data.shape[0] == 68:
        print("\nSUCCESS! This looks like a 1D array of 68 landmark vertex indices.")
        # Perform some sanity checks on the values
        print(f"Min index value: {np.min(landmark_data)}")
        print(f"Max index value: {np.max(landmark_data)}") # Should be < number of FLAME vertices (approx 5023)
        print(f"First 5 indices: {landmark_data[:5]}")

        # Ensure values are integers (or can be safely cast to int)
        if np.issubdtype(landmark_data.dtype, np.integer):
            print("Element data type is integer, which is expected for vertex indices.")
        else:
            print(f"Warning: Element data type is {landmark_data.dtype}. Expected integer for vertex indices.")

    elif isinstance(landmark_data, np.lib.npyio.NpzFile): # If it was an .npz archive by mistake
        print("\nThis is an .npz archive. Available keys:", list(landmark_data.keys()))
        print("Please inspect the keys to find the array of 68 indices.")
        # Example: if a key 'vertex_indices_68' exists:
        # if 'vertex_indices_68' in landmark_data:
        #    lmk_indices = landmark_data['vertex_indices_68']
        #    print(f"Shape of 'vertex_indices_68': {lmk_indices.shape}")
        #    if lmk_indices.shape == (68,):
        #        print("SUCCESS! Found 68 indices under key 'vertex_indices_68'.")

    else:
        print("\nThis file does not appear to be a direct 1D array of 68 indices. Further inspection needed if it's a dictionary or different structure.")

except FileNotFoundError:
    print(f"ERROR: File not found at {deca_landmark_path}. Please check the path.")
except Exception as e:
    print(f"An error occurred: {e}")

