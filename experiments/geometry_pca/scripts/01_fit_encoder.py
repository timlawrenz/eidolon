#!/usr/bin/env python3
import argparse
import os
import sys
import numpy as np

# Add parent dir to path so we can import geometry_pca
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geometry_pca.loader import load_matrix
from geometry_pca.gpa import gpa_align
from geometry_pca.fit import fit_encoder, save_encoder

def main():
    parser = argparse.ArgumentParser(description="Fit geometry PCA encoder on FFHQ-Stratum.")
    parser.add_argument("--limit", type=int, default=10000, help="Max samples to load")
    parser.add_argument("--k", type=int, default=50, help="Number of PCA components to keep")
    parser.add_argument("--out", type=str, default="output/encoder.npz", help="Output path")
    parser.add_argument("--conf-thresh", type=float, default=0.5, help="Confidence threshold to drop occluded faces")
    args = parser.parse_args()

    print(f"Loading up to {args.limit} samples (conf >= {args.conf_thresh})...")
    M = load_matrix(limit=args.limit, drop_low_conf=True, conf_threshold=args.conf_thresh)
    
    N = M.shape[0]
    print(f"Loaded {N} valid faces. Running GPA alignment...")
    
    aligned_M, mean_shape = gpa_align(M)
    
    print(f"Fitting PCA with k={args.k} components...")
    encoder = fit_encoder(aligned_M, mean_shape, k=args.k)
    
    evr = encoder["explained_variance_ratio"]
    total_var = np.sum(evr)
    print(f"Retained {total_var*100:.2f}% of variance at k={args.k}.")
    
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_encoder(encoder, args.out)
    print(f"Encoder saved to {args.out}")

if __name__ == "__main__":
    main()
