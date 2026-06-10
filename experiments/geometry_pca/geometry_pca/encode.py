import numpy as np
from geometry_pca.gpa import align_single

def encode_pose(source: np.ndarray, encoder: dict) -> np.ndarray:
    """
    Project and whiten a single (68, 2) unaligned pose into the z_g slider vector.
    
    Args:
        source: (68, 2) raw keypoint coordinates
        encoder: Dictionary returned by fit_encoder() or load_encoder()
        
    Returns:
        z_g: (k,) whitened geometric vector
    """
    # 1. Align to the frozen GPA mean shape (removes translation, scale, rotation)
    aligned = align_single(source, encoder["gpa_mean"])
    
    # 2. Flatten
    flat = aligned.reshape(-1)
    
    # 3. Center using the PCA mean
    centered = flat - encoder["pca_mean"]
    
    # 4. Project onto the PCA components
    score = centered @ encoder["components"].T
    
    # 5. Whiten to zero mean, unit variance
    z_g = (score - encoder["whiten_mu"]) / encoder["whiten_sigma"]
    
    return z_g.astype(np.float32)
