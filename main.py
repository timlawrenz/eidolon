import torch
import numpy as np
import pickle # Use the pickle library for .pkl files
import matplotlib.pyplot as plt
from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh import TexturesVertex
from pytorch3d.renderer import (
    look_at_view_transform, FoVPerspectiveCameras, PointLights, RasterizationSettings,
    MeshRenderer, MeshRasterizer, SoftPhongShader
)

# --- 1. Define the path to your FLAME model file ---
# UPDATE THIS PATH to the actual .pkl file you found
flame_model_path = './data/flame_model/flame2023.pkl' 

# The chumpy library, used by the FLAME model pickle, relies on older NumPy aliases.
# Pinning NumPy to version 1.23.5 in requirements.txt ensures these aliases are
# available, allowing the pickle file to load correctly.

try:
    # Load the model using pickle
    with open(flame_model_path, 'rb') as f:
        flame_model = pickle.load(f, encoding='latin1')
    print("FLAME 2023 model loaded successfully.")
except FileNotFoundError:
    print(f"ERROR: FLAME model not found at {flame_model_path}")
    # Note: if this path is taken, flame_model will not be defined,
    # and subsequent code relying on it will fail.
# except Exception as e: # Optionally, catch other unpickling errors
#     print(f"An error occurred during unpickling: {e}")

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

# Part B

# We use our new variables to create the mesh.
# The texture is now our generic gray color per vertex.
# We need to add a batch dimension for PyTorch3D -> (1, num_vertices, 3)
verts_rgb = vertex_colors.unsqueeze(0)
textures = TexturesVertex(verts_features=verts_rgb)

# Create the Meshes object
average_face_mesh = Meshes(
    verts=[mean_shape], # Use the mean_shape directly
    faces=[triangles],
    textures=textures
)
print("PyTorch3D Meshes object created with FLAME 2023 model.")

# --- Select a device ---
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# --- Set up the renderer (camera, lights, etc.) ---
R, T = look_at_view_transform(dist=2.7, elev=0, azim=0)
cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
raster_settings = RasterizationSettings(image_size=512, blur_radius=0.0, faces_per_pixel=1)
lights = PointLights(device=device, location=[[0.0, 0.0, 3.0]])
shader = SoftPhongShader(device=device, cameras=cameras, lights=lights)

renderer = MeshRenderer(
    rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
    shader=shader
)

# --- Move mesh to the correct device and render! ---
average_face_mesh = average_face_mesh.to(device)
images = renderer(average_face_mesh)

# --- Visualize the output ---
plt.figure(figsize=(8, 8))
plt.imshow(images[0, ..., :3].cpu().numpy())
plt.axis("off")
plt.title("Success! The Rendered Average Face")
plt.show()
