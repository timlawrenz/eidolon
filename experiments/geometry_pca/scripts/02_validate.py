#!/usr/bin/env python3
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geometry_pca.loader import load_matrix
from geometry_pca.fit import load_encoder
from geometry_pca.validate import scree_plot, recon_error, traversal_plot

def main():
    enc_path = "output/encoder.npz"
    if not os.path.exists(enc_path):
        print(f"Encoder not found at {enc_path}. Run 01_fit_encoder.py first.")
        return

    encoder = load_encoder(enc_path)
    out_dir = "output"
    
    print("Generating scree plot...")
    scree_plot(encoder, os.path.join(out_dir, "scree.png"))
    
    print("Loading validation set (500 samples)...")
    M = load_matrix(limit=500, drop_low_conf=True)
    
    print("Calculating reconstruction errors...")
    errs = recon_error(encoder, M, os.path.join(out_dir, "recon_error.png"))
    
    print("Generating ±3σ traversal plots for C1-C5...")
    for i in range(min(5, encoder["components"].shape[0])):
        traversal_plot(encoder, i, os.path.join(out_dir, f"traversal_C{i+1}.png"))
        
    metrics = {
        "retained_variance": float(sum(encoder["explained_variance_ratio"])),
        "recon_rmse": errs
    }
    
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
        
    print("Validation complete. Check output/ for plots and metrics.")

if __name__ == "__main__":
    main()
