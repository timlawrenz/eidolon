import numpy as np

def center_and_scale(shape: np.ndarray) -> np.ndarray:
    """
    Center shape at origin (subtract centroid) and scale to unit Frobenius norm.
    """
    centroid = shape.mean(axis=0)
    centered = shape - centroid
    norm = np.linalg.norm(centered)
    if norm > 1e-8:
        return centered / norm
    return centered

def get_rotation_matrix(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Find optimal rotation matrix R to align source to target using SVD.
    Assumes source and target are already centered.
    """
    # Covariance matrix H
    H = source.T @ target
    
    # SVD
    U, S, Vt = np.linalg.svd(H)
    
    # R = U * Vt
    R = U @ Vt
    
    # Prevent reflection (ensure determinant is 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt
        
    return R

def align_single(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Align a single shape to a reference shape (translation, scale, rotation).
    Returns the aligned shape.
    """
    src_cs = center_and_scale(source)
    # The reference is assumed to be centered/scaled already in the GPA loop,
    # but we don't strictly modify it here.
    R = get_rotation_matrix(src_cs, reference)
    return src_cs @ R

def gpa_align(M: np.ndarray, tol: float = 1e-5, max_iter: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """
    Generalized Procrustes Analysis (GPA) on a dataset of shapes M (B, N, 2).
    Iteratively aligns all shapes to the mean shape until convergence.
    
    Returns:
        aligned_M: (B, N, 2) array of aligned shapes
        mean_shape: (N, 2) the final mean reference shape
    """
    B, N, D = M.shape
    aligned = np.zeros_like(M, dtype=np.float32)
    
    # 1. Initialize: center and scale all shapes
    for i in range(B):
        aligned[i] = center_and_scale(M[i])
        
    # Pick the first shape as the initial reference
    mean_shape = aligned[0].copy()
    
    for iteration in range(max_iter):
        # Align everything to the current mean shape
        for i in range(B):
            aligned[i] = align_single(aligned[i], mean_shape)
            
        # Recompute the new mean shape
        new_mean = aligned.mean(axis=0)
        new_mean = center_and_scale(new_mean)
        
        # Check for convergence
        diff = np.linalg.norm(new_mean - mean_shape)
        mean_shape = new_mean
        
        if diff < tol:
            break
            
    return aligned, mean_shape
