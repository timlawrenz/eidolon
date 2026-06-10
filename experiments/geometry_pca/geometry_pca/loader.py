import os
import numpy as np
from geometry_pca.constants import STRATUM_ROOT, FACE_SLICE, N_FACE_PTS

def iter_sample_ids(limit: int | None = None):
    """Yield zero-padded sample IDs up to a limit."""
    i = 0
    while True:
        if limit is not None and i >= limit:
            break
        yield f"{i:05d}"
        i += 1

def load_face_keypoints(sample_id: str, return_conf: bool = False) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Load a single sample's pose.npy, slice facial points, and return (68, 2) float32.
    If return_conf is True, also return the (68,) confidence array.
    """
    path = os.path.join(STRATUM_ROOT, sample_id, "pose.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing pose.npy for sample {sample_id} at {path}")
    
    # Load the (133, 3) float16 array
    pose = np.load(path)
    
    # Slice the 68 face keypoints -> (68, 3)
    face_pts = pose[FACE_SLICE]
    
    # Split into coords and confidence
    coords = face_pts[:, :2].astype(np.float32)
    
    if return_conf:
        conf = face_pts[:, 2].astype(np.float32)
        return coords, conf
        
    return coords

def load_matrix(limit: int = 10000, drop_low_conf: bool = True, conf_threshold: float = 0.5) -> np.ndarray:
    """
    Load up to `limit` faces into a matrix M of shape (N, 68, 2).
    If drop_low_conf is True, samples with mean confidence below the threshold are dropped.
    """
    valid_faces = []
    
    for sample_id in iter_sample_ids(limit):
        try:
            coords, conf = load_face_keypoints(sample_id, return_conf=True)
            
            if drop_low_conf:
                # If the face is highly occluded or missing, confidence will be low
                if np.mean(conf) < conf_threshold:
                    continue
                    
            valid_faces.append(coords)
            
        except FileNotFoundError:
            # If a sample directory is missing, just skip it
            continue
            
    if not valid_faces:
        raise ValueError("No valid faces found.")
        
    return np.stack(valid_faces, axis=0)
