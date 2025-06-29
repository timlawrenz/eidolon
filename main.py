import torch
import numpy as np
# --- Monkey patch for old numpy data in pkl file ---
# This is a temporary workaround to load a pickle file created with an older
# version of numpy, which has deprecated several type aliases.
np.bool = np.bool_
np.int = np.int_
np.float = np.float_
np.complex = np.complex_
np.object = np.object_
np.unicode = np.unicode_
np.str = np.str_
import matplotlib.pyplot as plt
from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh import TexturesVertex
from pytorch3d.renderer import (
    look_at_view_transform, FoVPerspectiveCameras, PointLights, RasterizationSettings,
    MeshRenderer, MeshRasterizer, SoftPhongShader
)
from src.model import EidolonEncoder, FLAME # Import the new encoder and FLAME
from src.utils import deconstruct_flame_coeffs, apply_coordinate_system_correction # Import the coefficient deconstructor

# --- 1. Load FLAME Model ---
flame_model_path = './data/flame_model/flame2023.pkl'
landmark_path = './data/flame_model/deca_landmark_embedding.npz'
# These must match the values used during training of the loaded model!
NUM_SHAPE_COEFFS = 100
NUM_EXPRESSION_COEFFS = 0

try:
    flame_model = FLAME(flame_model_path, landmark_path, NUM_SHAPE_COEFFS, NUM_EXPRESSION_COEFFS)
    print("FLAME 2023 model loaded successfully via FLAME class.")
except Exception as e:
    print(f"ERROR: Could not instantiate FLAME model from {flame_model_path}")
    print(f"Please ensure all model assets are downloaded as per the README. Error: {e}")
    # Exit or handle error appropriately
    exit()

# --- 2. Extract and Prepare Key Components ---
# Get the mean shape and face indices from the FLAME model instance
mean_shape = flame_model.v_template.clone()
triangles = flame_model.faces_idx.clone()

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
# Apply coordinate system correction to vertices for rendering to match camera system.
# Note: `apply_coordinate_system_correction` expects a batch dimension.
mean_shape_for_render = apply_coordinate_system_correction(mean_shape.unsqueeze(0)).squeeze(0)
average_face_mesh = Meshes(
    verts=[mean_shape_for_render],
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
rendered_average_face = renderer(average_face_mesh)
print("Rendered the average FLAME face.")

# --- 4. Full End-to-End Inference Test ---
print("\n--- Running Full Inference Test ---")
try:
    # --- Define Coefficient counts (must match training & inference.py) ---
    NUM_GLOBAL_POSE_COEFFS = 6
    NUM_JAW_POSE_COEFFS = 3
    NUM_EYE_POSE_COEFFS = 6
    NUM_NECK_POSE_COEFFS = 3
    NUM_TRANSLATION_COEFFS = 3
    num_total_coeffs = 227 # Total number of FLAME parameters
    NUM_DETAIL_COEFFS = num_total_coeffs - (NUM_SHAPE_COEFFS + NUM_EXPRESSION_COEFFS + NUM_GLOBAL_POSE_COEFFS + \
                                            NUM_JAW_POSE_COEFFS + NUM_EYE_POSE_COEFFS + NUM_NECK_POSE_COEFFS + \
                                            NUM_TRANSLATION_COEFFS)

    # --- Load Encoder with Pre-trained Weights ---
    encoder = EidolonEncoder(num_coeffs=num_total_coeffs).to(device)
    encoder_path = 'eidolon_encoder_stage_3.pth'
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    encoder.eval()
    print(f"Loaded trained encoder from '{encoder_path}'")

    # --- Move FLAME model to device ---
    flame_model.to(device)
    flame_model.eval()

    # --- Create a dummy input image (not used if bypassing encoder) ---
    dummy_image = torch.randn(1, 3, 224, 224).to(device)
    print("Created a dummy random image (for encoder input if not bypassed).")

    # --- Run Full Inference Test ---
    with torch.no_grad():
        # 1. Predict coefficients from the image
        # pred_coeffs_vec = encoder(dummy_image)

        # --- Create Neutral Coefficients for LBS/FLAME Debugging ---
        # To isolate issues in the FLAME model from the encoder, we can feed it a
        # known, neutral set of coefficients instead of using the encoder's output.
        print("DEBUG: Bypassing encoder and using hardcoded neutral coefficients.")
        pred_coeffs_vec = torch.zeros(1, num_total_coeffs, device=device)
        # The neutral pose is an identity rotation. For 6D rotation representation,
        # this corresponds to [1, 0, 0, 0, 1, 0] in the flattened 3x2 matrix.
        pose_start_idx = NUM_SHAPE_COEFFS + NUM_EXPRESSION_COEFFS
        pred_coeffs_vec[0, pose_start_idx + 0] = 1.0 # R[0,0]
        pred_coeffs_vec[0, pose_start_idx + 4] = 1.0 # R[1,1]

        # 2. Deconstruct coefficients into a dictionary
        pred_coeffs_dict = deconstruct_flame_coeffs(
            pred_coeffs_vec,
            NUM_SHAPE_COEFFS, NUM_EXPRESSION_COEFFS, NUM_GLOBAL_POSE_COEFFS,
            NUM_JAW_POSE_COEFFS, NUM_EYE_POSE_COEFFS, NUM_NECK_POSE_COEFFS,
            NUM_TRANSLATION_COEFFS, NUM_DETAIL_COEFFS
        )

        # 3. Generate mesh vertices using the FLAME model
        pred_verts, _ = flame_model(
            shape_params=pred_coeffs_dict['shape_params'],
            expression_params=pred_coeffs_dict['expression_params'],
            pose_params=pred_coeffs_dict['pose_params'],
            jaw_pose_params=pred_coeffs_dict['jaw_pose_params'],
            eye_pose_params=pred_coeffs_dict['eye_pose_params'],
            neck_pose_params=pred_coeffs_dict['neck_pose_params'],
            transl=pred_coeffs_dict['transl']
        )
    print("Inference pipeline complete (encoder -> coeffs -> FLAME -> vertices).")

    # --- Render Predicted Mesh ---
    # We can reuse the `textures` and `renderer` from the average face rendering
    # Apply coordinate system correction to vertices for rendering to match camera system
    pred_verts_for_render = apply_coordinate_system_correction(pred_verts)

    # The Meshes class expects a list of tensors for verts and faces.
    # We convert the predicted verts tensor to a list to match the faces format.
    pred_mesh = Meshes(
        verts=list(pred_verts_for_render),
        faces=[triangles.to(device)],
        textures=textures.to(device)
    )

    rendered_predicted_face = renderer(pred_mesh)
    print("Rendered predicted mesh from dummy image.")

    # --- Visualize Both Outputs Side-by-Side ---
    plt.figure(figsize=(10, 5))
    
    # Plot Average Face
    plt.subplot(1, 2, 1)
    plt.imshow(rendered_average_face[0, ..., :3].cpu().numpy())
    plt.title("Average Face Shape")
    plt.axis("off")

    # Plot Predicted Face
    plt.subplot(1, 2, 2)
    plt.imshow(rendered_predicted_face[0, ..., :3].cpu().numpy())
    plt.title("Predicted Face from Dummy Image")
    plt.axis("off")
    
    plt.suptitle("main.py: End-to-End Test Result")
    output_image_path = "main_test_result.png"
    plt.savefig(output_image_path)
    plt.close() # Free up memory
    print(f"Full inference test passed. Output image saved to '{output_image_path}'.")

except Exception as e:
    print(f"An error occurred during the full inference test: {e}")
    import traceback
    traceback.print_exc()
