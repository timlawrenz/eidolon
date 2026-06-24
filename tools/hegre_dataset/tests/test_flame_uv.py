import numpy as np
import pytest

# We will implement this function next
try:
    from tools.hegre_dataset.review.flame_projector import compute_uv_coordinates
except ImportError:
    compute_uv_coordinates = None

@pytest.mark.skipif(compute_uv_coordinates is None, reason="Not implemented yet")
def test_uv_mapping_anchors_nose_to_center():
    # Mock a FLAME skull with 3 vertices
    # v0: Left cheek (-10, 0, 0)
    # v1: Nose tip (0, 0, 10)
    # v2: Right cheek (10, 0, 0)
    vertices = np.array([
        [-10.0, 0.0, 0.0],
        [0.0, 0.0, 10.0],
        [10.0, 0.0, 0.0]
    ], dtype=np.float32)
    
    nose_idx = 1
    out_size = (300, 300)
    
    uvs = compute_uv_coordinates(vertices, nose_idx, out_size)
    
    # UV coordinates are normalized [0, 1] for .obj files
    # The nose (v1) MUST map perfectly to (0.5, 0.5) because we anchored the pixel average there.
    assert np.isclose(uvs[1, 0], 0.5, atol=1e-5), "Nose U coordinate not centered"
    assert np.isclose(uvs[1, 1], 0.5, atol=1e-5), "Nose V coordinate not centered"
    
    # Left cheek (negative X in 3D right-handed space) should be on the left side of the UV (u < 0.5)
    assert uvs[0, 0] < 0.5, "Left cheek UV mapping flipped"
    
    # Right cheek should be on the right side (u > 0.5)
    assert uvs[2, 0] > 0.5, "Right cheek UV mapping flipped"

@pytest.mark.skipif(compute_uv_coordinates is None, reason="Not implemented yet")
def test_uv_mapping_flips_y_axis():
    # v0: Nose tip (0, 0, 10)
    # v1: Forehead (0, 10, 0) -> +Y is UP in 3D
    vertices = np.array([
        [0.0, 0.0, 10.0],
        [0.0, 10.0, 0.0]
    ], dtype=np.float32)
    
    uvs = compute_uv_coordinates(vertices, nose_idx=0, out_size=(300, 300))
    
    # Image UV origin (0,0) is TOP-LEFT. 
    # Therefore, the forehead (+Y in 3D) should have a SMALLER V coordinate than the nose.
    assert uvs[1, 1] < uvs[0, 1], "Y-axis was not flipped for UV mapping!"

