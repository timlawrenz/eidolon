import torch
import numpy as np
import pickle # Use the pickle library for .pkl files
import matplotlib.pyplot as plt

# --- 1. Define the path to your FLAME model file ---
# UPDATE THIS PATH to the actual .pkl file you found
flame_model_path = './data/flame_model/flame2023.pkl' 

# In older versions of NumPy, `np.int` was an alias (often for `np.int_` or Python's `int`).
# Pickle files created with such versions may expect `np.int` to exist.
# We temporarily create this alias before loading the pickle file.
_original_np_int = getattr(np, 'int', None) # Save the original np.int if it exists
np.int = np.int_ # Create the alias np.int pointing to np.int_

try:
    # Load the model using pickle
    with open(flame_model_path, 'rb') as f:
        flame_model = pickle.load(f, encoding='latin1')
    print("FLAME 2023 model loaded successfully.")
except FileNotFoundError:
    print(f"ERROR: FLAME model not found at {flame_model_path}")
    # Note: if this path is taken, flame_model will not be defined,
    # and subsequent code relying on it will fail.
finally:
    # Restore np.int to its original state to avoid potential side effects.
    if _original_np_int is not None:
        np.int = _original_np_int # Restore the original attribute
    elif hasattr(np, 'int'): # If we defined np.int and it wasn't there originally
        del np.int # Remove the alias we created

# --- 2. Extract and Prepare Key Components ---
# The keys in the FLAME model dictionary are different from BFM's
# v_template is the average face shape (the equivalent of BFM's shapeMU)
mean_shape = torch.from_numpy(flame_model['v_template']).float()

# f contains the face triangles (the equivalent of BFM's tl)
# It's already 0-indexed, so we don't need to subtract 1
triangles = torch.from_numpy(flame_model['f'].astype(np.int64))

# Get the number of vertices and faces
num_vertices = mean_shape.shape[0]
num_triangles = triangles.shape[0]
print(f"Data parsed: {num_vertices} vertices, {num_triangles} triangles.")

# --- 3. Create a Generic Texture ---
# The base FLAME model doesn't come with a mean texture map like BFM.
# We will create a simple, uniform gray color for now.
# The goal is to confirm the SHAPE is rendering correctly.
# We create one color (e.g., gray) and repeat it for all vertices.
generic_color = torch.tensor([0.7, 0.7, 0.7]) # A nice medium gray
vertex_colors = generic_color.view(1, 3).expand(num_vertices, 3)
