import numpy as np
import pytest
from unittest.mock import MagicMock, patch

try:
    from tools.hegre_dataset.review.flame_projector import render_spin_gif
except ImportError:
    render_spin_gif = None

# @pytest.mark.skipif(render_spin_gif is None, reason="Not implemented yet")
def test_render_spin_gif_produces_frames(tmp_path):
    # Mock a trimesh object
    mock_mesh = MagicMock()
    
    output_path = tmp_path / "spin.gif"
    
    # We will patch pyrender/trimesh/imageio inside the function
    # Just verifying it calls imageio.mimsave with a list of arrays
    with patch("tools.hegre_dataset.review.flame_projector.imageio.mimsave") as mock_save:
        with patch("tools.hegre_dataset.review.flame_projector.pyrender.OffscreenRenderer"):
            with patch("tools.hegre_dataset.review.flame_projector.pyrender.Mesh.from_trimesh"):
                with patch("tools.hegre_dataset.review.flame_projector.pyrender.Scene"):
                    
                    # We will mock the render() call to just return a black 300x300 image
                    mock_renderer = MagicMock()
                    mock_renderer.render.return_value = (np.zeros((300, 300, 3), dtype=np.uint8), None)
                    
                    with patch("tools.hegre_dataset.review.flame_projector.pyrender.OffscreenRenderer", return_value=mock_renderer):
                        
                        render_spin_gif(mock_mesh, output_path, num_frames=3, resolution=(300, 300))
                        
                        # Verify mimsave was called with our path and a list of 3 frames
                        assert mock_save.called
                        args, kwargs = mock_save.call_args
                        
                        assert str(args[0]) == str(output_path)
                        frames = args[1]
                        assert len(frames) == 3
                        assert frames[0].shape == (300, 300, 3)

