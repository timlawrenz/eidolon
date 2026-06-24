import numpy as np
import pytest

from tools.hegre_dataset.review.geometry import normalize_face_geometry
from tools.hegre_dataset.review.procrustes import scale_and_center_landmarks

def create_mock_face():
    # Create 68 points
    face = np.zeros((68, 2), dtype=np.float32)
    
    # Let's set the eyes:
    # left_eye is indices 36:42
    # right_eye is indices 42:48
    # nose_tip is index 30
    
    # Left eye center roughly at (-10, 10)
    for i in range(36, 42):
        face[i] = [-10.0, 10.0]
    
    # Right eye center roughly at (10, 10)
    for i in range(42, 48):
        face[i] = [10.0, 10.0]
        
    # Nose tip at (0, 0)
    face[30] = [0.0, 0.0]
    
    # Add bounds so size_c is non-zero
    face[0] = [-20.0, -20.0]
    face[16] = [20.0, 20.0]
    
    return face

def test_normalize_face_geometry_unrotated():
    face = create_mock_face()
    
    # Rotate the face by 45 degrees around origin
    theta = np.radians(45)
    c, s = np.cos(theta), np.sin(theta)
    rot_mat = np.array([[c, -s], [s, c]])
    
    tilted_face = face @ rot_mat.T
    
    # Shift nose tip arbitrarily
    tilted_face += np.array([20.0, 30.0])
    
    normalized = normalize_face_geometry(tilted_face)
    
    # Check that eyes are horizontal
    left_eye = np.mean(normalized[36:42], axis=0)
    right_eye = np.mean(normalized[42:48], axis=0)
    
    assert np.isclose(left_eye[1], right_eye[1], atol=1e-5), f"Eyes are not horizontal! Left: {left_eye}, Right: {right_eye}"
    
    # Check that nose tip is invariant relative to the input
    # The tilted face nose tip is exactly at (20, 30)
    assert np.isclose(normalized[30][0], 20.0, atol=1e-5), "Nose tip X changed!"
    assert np.isclose(normalized[30][1], 30.0, atol=1e-5), "Nose tip Y changed!"

def test_scale_and_center_landmarks():
    face = create_mock_face()
    
    # Apply some arbitrary transformation
    face = face * 2.0 + np.array([50.0, 60.0])
    
    # Scale and center into a 300x300 box
    out_size = (300, 300)
    scaled = scale_and_center_landmarks(face, out_size)
    
    # Check nose tip is perfectly centered
    assert np.isclose(scaled[30][0], 150.0, atol=1e-5), "Nose tip X not centered!"
    assert np.isclose(scaled[30][1], 150.0, atol=1e-5), "Nose tip Y not centered!"
