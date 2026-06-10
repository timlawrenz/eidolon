#!/usr/bin/env python3
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geometry_pca.loader import load_face_keypoints, iter_sample_ids
from geometry_pca.fit import load_encoder
from geometry_pca.gpa import align_single

def main():
    enc_path = "output/encoder.npz"
    if not os.path.exists(enc_path):
        print(f"Encoder not found at {enc_path}.")
        return

    encoder = load_encoder(enc_path)
    out_dir = "output"
    
    pca_mean = encoder["pca_mean"]
    comp1 = encoder["components"][0]
    sigma1 = encoder["whiten_sigma"][0]
    mu1 = encoder["whiten_mu"][0]
    
    print("1. Generating separate frames for C1 traversal...")
    zs = [-3.0, -1.5, 0.0, 1.5, 3.0]
    for i, z in enumerate(zs):
        score = z * sigma1 + mu1
        shape_flat = pca_mean + score * comp1
        shape = shape_flat.reshape(-1, 2)
        
        plt.figure(figsize=(5, 5))
        plt.scatter(shape[:, 0], -shape[:, 1], c='black', s=20)
        plt.title(f'C1 Traversal: {z}σ')
        plt.axis('equal')
        plt.grid(True, linestyle=':', alpha=0.5)
        
        # Lock axis limits so they don't bounce around between frames
        plt.xlim(-0.3, 0.3)
        plt.ylim(-0.3, 0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"c1_frame_{i+1}_z{z}.png"))
        plt.close()
        
    print("2. Finding dataset outliers for C1...")
    ids = []
    shapes = []
    
    # Load 2000 samples to find extremes
    for sid in iter_sample_ids(2000):
        try:
            coords, conf = load_face_keypoints(sid, return_conf=True)
            if np.mean(conf) >= 0.5:
                ids.append(sid)
                shapes.append(coords)
        except FileNotFoundError:
            continue
            
    shapes = np.stack(shapes)
    
    # Align and project
    aligned = np.array([align_single(s, encoder["gpa_mean"]) for s in shapes])
    flat = aligned.reshape(len(shapes), -1)
    centered = flat - pca_mean
    scores = centered @ encoder["components"].T
    
    c1_scores = scores[:, 0]
    
    # Get indices of the 3 lowest and 3 highest scores
    bot_idx = np.argsort(c1_scores)[:3]
    top_idx = np.argsort(c1_scores)[-3:]
    
    # Plot the raw, unaligned shapes of these extremes
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle("Raw Landmarks of C1 Extremes (Min vs Max)")
    
    for j, idx in enumerate(bot_idx):
        s = shapes[idx]
        axes[0, j].scatter(s[:, 0], -s[:, 1], c='blue', s=10)
        axes[0, j].set_title(f"Min C1 (Score: {c1_scores[idx]:.2f})\nID: {ids[idx]}")
        axes[0, j].axis('equal')
        axes[0, j].grid(True, alpha=0.3)
        
    for j, idx in enumerate(top_idx):
        s = shapes[idx]
        axes[1, j].scatter(s[:, 0], -s[:, 1], c='red', s=10)
        axes[1, j].set_title(f"Max C1 (Score: {c1_scores[idx]:.2f})\nID: {ids[idx]}")
        axes[1, j].axis('equal')
        axes[1, j].grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "c1_outliers_raw.png"))
    plt.close()
    
    print("Diagnostic generation complete.")

if __name__ == "__main__":
    main()
