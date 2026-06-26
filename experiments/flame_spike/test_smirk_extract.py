import sys
import os
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "smirk"))
from src.smirk_encoder import SmirkEncoder

def test_shape_encoder():
    print("Loading SmirkEncoder (Shape only)...")
    model = SmirkEncoder()
    checkpoint = torch.load("experiments/flame_spike/smirk/pretrained_models/SMIRK_em1.pt", map_location="cpu")
    
    encoder_state_dict = {}
    for k, v in checkpoint.items():
        if k.startswith("smirk_encoder."):
            encoder_state_dict[k.replace("smirk_encoder.", "")] = v
            
    model.load_state_dict(encoder_state_dict, strict=False)
    model.eval()
    
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        outputs = model(x)
        
    print("Keys in output:", outputs.keys())
    shape_beta = outputs.get('shape_params', None)
    if shape_beta is not None:
        print("Shape (beta) shape:", shape_beta.shape)
        
if __name__ == "__main__":
    test_shape_encoder()
