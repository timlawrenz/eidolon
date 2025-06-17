"""
Script to run inference with a trained EidolonEncoder model on a single image.
"""

import torch
from torchvision import transforms
from PIL import Image
import argparse
import os
import sys

# --- 3D  Rendering Imports ---
from torchvision.utils import save_image
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform, FoVPerspectiveCameras, PointLights, RasterizationSettings,
    MeshRenderer, MeshRasterizer, SoftPhongShader, TexturesVertex
)

# --- Path Setup ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# --- Project Imports ---
from src.model import EidolonEncoder, FLAME
from src.utils import save_obj, deconstruct_flame_coeffs

def run_inference():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run FLAME inference on a single image.")
    parser.add_argument('--image_path', type=str, required=True, help="Path to the input image.")
    parser.add_argument('--model_path', type=str, default='eidolon_encoder_stage_3.pth', help="Path to the trained encoder weights.")
    parser.add_argument('--output_path', type=str, default='output/inference_result.obj', help="Path to save the output .obj mesh.")
    parser.add_argument('--output_image_path', type=str, default='output/inference_render.png', help="Path to save the rendered output image.")
    args = parser.parse_args()

    # --- Config ---
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # These must match the values used during training of the loaded model!
    NUM_SHAPE_COEFFS = 100
    NUM_EXPRESSION_COEFFS = 0
    NUM_GLOBAL_POSE_COEFFS = 6
    NUM_JAW_POSE_COEFFS = 3
    NUM_EYE_POSE_COEFFS = 6
    NUM_NECK_POSE_COEFFS = 3
    NUM_TRANSLATION_COEFFS = 3
    NUM_DETAIL_COEFFS = 106
    NUM_COEFFS = NUM_SHAPE_COEFFS + NUM_EXPRESSION_COEFFS + NUM_GLOBAL_POSE_COEFFS + \
                 NUM_JAW_POSE_COEFFS + NUM_EYE_POSE_COEFFS + NUM_NECK_POSE_COEFFS + \
                 NUM_TRANSLATION_COEFFS + NUM_DETAIL_COEFFS

    # --- Load Models ---
    print(f"Loading trained encoder from: {args.model_path}")
    encoder = EidolonEncoder(num_coeffs=NUM_COEFFS).to(DEVICE)
    encoder.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
    encoder.eval()

    # Initialize FLAME model
    flame_model_path = 'data/flame_model/flame2023.pkl'
    landmark_path = 'data/flame_model/deca_landmark_embedding.npz'
    if not (os.path.exists(flame_model_path) and os.path.exists(landmark_path)):
        print("Error: FLAME model or landmark embedding not found.")
        print("Please follow the setup instructions in README.md to download the necessary assets.")
        sys.exit(1)
    flame = FLAME(flame_model_path, landmark_path, NUM_SHAPE_COEFFS, NUM_EXPRESSION_COEFFS).to(DEVICE)
    flame.eval()

    # --- Preprocess Image ---
    print(f"Loading and preprocessing image: {args.image_path}")
    input_image = Image.open(args.image_path).convert('RGB')
    # Use the same transforms as in the dataset
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0).to(DEVICE) # Add batch dimension

    # --- Run Inference ---
    print("Running inference...")
    with torch.no_grad():
        pred_coeffs_vec = encoder(input_batch)
        pred_coeffs_dict = deconstruct_flame_coeffs(
            pred_coeffs_vec,
            NUM_SHAPE_COEFFS, NUM_EXPRESSION_COEFFS, NUM_GLOBAL_POSE_COEFFS,
            NUM_JAW_POSE_COEFFS, NUM_EYE_POSE_COEFFS, NUM_NECK_POSE_COEFFS,
            NUM_TRANSLATION_COEFFS, NUM_DETAIL_COEFFS
        )
        pred_verts, _ = flame(
            shape_params=pred_coeffs_dict['shape_params'],
            expression_params=pred_coeffs_dict['expression_params'],
            pose_params=pred_coeffs_dict['pose_params'],
            jaw_pose_params=pred_coeffs_dict['jaw_pose_params'],
            eye_pose_params=pred_coeffs_dict['eye_pose_params'],
            neck_pose_params=pred_coeffs_dict['neck_pose_params'],
            transl=pred_coeffs_dict['transl'],
            use_posedirs=True # Use pose-dependent blendshapes for best quality
        )

    # --- Render Mesh to Image ---
    print("Rendering predicted mesh...")
    # Set up renderer. We use a fixed camera as the pose is predicted by the model.
    # Explicitly define camera extrinsics for a standard view.
    # This corresponds to a camera at the origin looking down the -Z axis.
    R = torch.eye(3).unsqueeze(0).to(DEVICE)
    T = torch.tensor([[0, 0, 3.0]]).to(DEVICE)
    cameras = FoVPerspectiveCameras(device=DEVICE, R=R, T=T, fov=12.0)
    raster_settings = RasterizationSettings(image_size=512, blur_radius=0.0, faces_per_pixel=1)
    lights = PointLights(device=DEVICE, location=[[0.0, 0.0, 3.0]])
    shader = SoftPhongShader(device=DEVICE, cameras=cameras, lights=lights)
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=shader
    )

    # Create a Meshes object for rendering
    num_vertices = pred_verts.shape[1]
    generic_color = torch.tensor([0.7, 0.7, 0.7], device=DEVICE) # Medium gray
    vertex_colors = generic_color.view(1, 3).expand(num_vertices, 3)
    verts_rgb = vertex_colors.unsqueeze(0) # (1, V, 3)
    textures = TexturesVertex(verts_features=verts_rgb)
    
    pred_mesh = Meshes(
        verts=pred_verts,
        faces=flame.faces_idx.unsqueeze(0),
        textures=textures
    )

    rendered_images = renderer(pred_mesh) # Shape: (1, H, W, 4)
    # Convert to image format (C, H, W) and remove alpha channel
    rendered_image_for_save = rendered_images[0, ..., :3].permute(2, 0, 1)

    # --- Save Output ---
    # Ensure output directories exist
    output_obj_dir = os.path.dirname(args.output_path)
    if output_obj_dir:
        os.makedirs(output_obj_dir, exist_ok=True)
    
    output_img_dir = os.path.dirname(args.output_image_path)
    if output_img_dir:
        os.makedirs(output_img_dir, exist_ok=True)
        
    # Save the 3D mesh and the rendered image
    save_obj(args.output_path, pred_verts[0], flame.faces_idx)
    save_image(rendered_image_for_save, args.output_image_path)
    print(f"Saved rendered image to {args.output_image_path}")
    print("Inference complete.")

if __name__ == '__main__':
    run_inference()
