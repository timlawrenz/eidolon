import numpy as np
import pytest

# We will implement this function next
try:
    from tools.hegre_dataset.review.flame_projector import compute_uv_coordinates
except ImportError:
    compute_uv_coordinates = None

@pytest.mark.skipif(compute_uv_coordinates is None, reason="Not implemented yet")
def test_uv_mapping_anchors_nose_to_center():
    vertices = np.array([
        [-10.0, 0.0, 0.0],
        [0.0, 0.0, 10.0],
        [10.0, 0.0, 0.0]
    ], dtype=np.float32)
    
    landmarks = np.zeros((68, 3))
    landmarks[30] = vertices[1] # nose
    landmarks[36:42] = [[-5, 5, 5]] * 6 # left eye
    landmarks[42:48] = [[5, 5, 5]] * 6 # right eye
    
    out_size = (300, 300)
    uvs = compute_uv_coordinates(vertices, landmarks, out_size)
    
    assert np.isclose(uvs[1, 0], 0.5, atol=1e-5), "Nose U coordinate not centered"
    assert np.isclose(uvs[1, 1], 0.5, atol=1e-5), "Nose V coordinate not centered"
    assert uvs[0, 0] < 0.5, "Left cheek UV mapping flipped"
    assert uvs[2, 0] > 0.5, "Right cheek UV mapping flipped"

@pytest.mark.skipif(compute_uv_coordinates is None, reason="Not implemented yet")
def test_uv_mapping_flips_y_axis():
    vertices = np.array([
        [0.0, 0.0, 10.0],
        [0.0, 10.0, 0.0]
    ], dtype=np.float32)
    
    landmarks = np.zeros((68, 3))
    landmarks[30] = vertices[0] # nose
    landmarks[36:42] = [[-5, 5, 5]] * 6 # left eye
    landmarks[42:48] = [[5, 5, 5]] * 6 # right eye
    
    uvs = compute_uv_coordinates(vertices, landmarks, out_size=(300, 300))
    # Forehead (vertices[1]) has +Y. In OpenGL, V=1 is TOP. So forehead V > nose V.
    assert uvs[1, 1] > uvs[0, 1], "Y-axis should NOT be flipped for OpenGL UV mapping!"

