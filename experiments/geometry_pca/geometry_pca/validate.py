import json
import numpy as np
import matplotlib.pyplot as plt
from geometry_pca.gpa import gpa_align

def scree_plot(encoder: dict, out_path: str):
    evr = encoder["explained_variance_ratio"]
    cum_var = np.cumsum(evr)
    
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(cum_var) + 1), cum_var, marker='o', linestyle='-')
    plt.axhline(y=0.99, color='r', linestyle='--', label='99% Variance')
    plt.title('Scree Plot (Cumulative Explained Variance)')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def recon_error(encoder: dict, M: np.ndarray, out_path: str) -> dict:
    # Align validation shapes
    aligned_M, _ = gpa_align(M)
    flat_M = aligned_M.reshape(aligned_M.shape[0], -1)
    
    pca_mean = encoder["pca_mean"]
    components = encoder["components"]
    centered = flat_M - pca_mean
    scores = centered @ components.T # (N, K)
    
    ks = [1, 5, 10, 20, 30, 40, 50]
    ks = [k for k in ks if k <= components.shape[0]]
    errors = []
    
    plt.figure(figsize=(8, 5))
    for k in ks:
        # Reconstruct with top k
        recon = (scores[:, :k] @ components[:k, :]) + pca_mean
        # Mean Euclidean distance per point (RMSE)
        mse = np.mean((flat_M - recon)**2)
        rmse = np.sqrt(mse)
        errors.append(float(rmse))
        
    plt.plot(ks, errors, marker='o')
    plt.title('Reconstruction RMSE vs Components')
    plt.xlabel('Number of Components (K)')
    plt.ylabel('RMSE (Euclidean distance)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    
    return dict(zip(ks, errors))

def traversal_plot(encoder: dict, comp_idx: int, out_path: str):
    pca_mean = encoder["pca_mean"]
    comp = encoder["components"][comp_idx]
    sigma = encoder["whiten_sigma"][comp_idx]
    mu = encoder["whiten_mu"][comp_idx]
    
    zs = [-3, -1.5, 0, 1.5, 3]
    colors = ['red', 'orange', 'gray', 'blue', 'purple']
    
    plt.figure(figsize=(6, 6))
    for z, color in zip(zs, colors):
        score = z * sigma + mu
        shape_flat = pca_mean + score * comp
        shape = shape_flat.reshape(-1, 2)
        
        # Invert Y (-shape[:, 1]) because image Y goes down, but plot Y goes up
        plt.scatter(shape[:, 0], -shape[:, 1], c=color, label=f'{z}σ', alpha=0.6, s=15)
        
    plt.title(f'Component {comp_idx+1} Traversal (±3σ)')
    plt.axis('equal')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
