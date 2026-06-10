import numpy as np
from sklearn.decomposition import PCA

def fit_encoder(aligned_M: np.ndarray, gpa_mean: np.ndarray, k: int = 50) -> dict:
    """
    Fit PCA on GPA-aligned shapes and compute whitening statistics.
    
    Args:
        aligned_M: (N, 68, 2) array of GPA-aligned facial keypoints
        gpa_mean: (68, 2) mean reference shape from GPA
        k: number of principal components to keep
        
    Returns:
        Dictionary of arrays comprising the frozen encoder.
    """
    N = aligned_M.shape[0]
    flat_M = aligned_M.reshape(N, -1)  # (N, 136)
    
    pca = PCA(n_components=k)
    scores = pca.fit_transform(flat_M)  # (N, k)
    
    # Whitening stats calculated directly from the projected training scores
    # This guarantees that the inference whitening exactly matches the training distribution.
    whiten_mu = scores.mean(axis=0)
    whiten_sigma = scores.std(axis=0)
    
    # Prevent divide-by-zero on zero variance components
    whiten_sigma[whiten_sigma < 1e-8] = 1.0
    
    return {
        "components": pca.components_.astype(np.float32),
        "explained_variance_ratio": pca.explained_variance_ratio_.astype(np.float32),
        "pca_mean": pca.mean_.astype(np.float32),
        "whiten_mu": whiten_mu.astype(np.float32),
        "whiten_sigma": whiten_sigma.astype(np.float32),
        "gpa_mean": gpa_mean.astype(np.float32)
    }

def save_encoder(encoder: dict, path: str):
    """Save the encoder dictionary to a compressed .npz file."""
    np.savez_compressed(path, **encoder)

def load_encoder(path: str) -> dict:
    """Load the encoder dictionary from a .npz file."""
    data = np.load(path)
    return {k: data[k] for k in data.files}
