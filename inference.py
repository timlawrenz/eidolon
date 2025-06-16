"""
Script to run inference with a trained EidolonEncoder model on a single image.
"""

import torch
from torchvision import transforms
from PIL import Image
import argparse
import os
import sys

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
    parser.add_argument('--model_path', type=str, default='eidolon_encoder_v1_30_epochs.pth', help="Path to the trained encoder weights.")
    parser.add_argument('--output_path', type=str, default='output/inference_result.obj', help="Path to save the output .obj mesh.")
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
            transl=pred_coeffs_dict['transl']
        )

    # --- Save Output ---
    output_dir = os.path.dirname(args.output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    save_obj(args.output_path, pred_verts[0], flame.faces_idx)
    print("Inference complete.")

if __name__ == '__main__':
    run_inference()
