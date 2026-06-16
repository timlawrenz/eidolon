from PIL import Image
import json
from unittest.mock import patch, MagicMock

from tools.hegre_dataset.face_extraction import get_square_box, extract_faces_for_image
from tools.hegre_dataset.cli import main

def test_get_square_box():
    # Simple test for bounds clamping without padding
    box = [100, 100, 200, 200]
    sq_box = get_square_box(box, img_width=500, img_height=500, expand_ratio=1.0)
    assert sq_box == [100, 100, 200, 200]
    
    # Test shifting
    box = [0, 100, 100, 300]
    sq_box = get_square_box(box, img_width=500, img_height=500, expand_ratio=1.0)
    assert sq_box[0] >= 0

@patch("tools.hegre_dataset.face_extraction.MTCNN")
def test_extract_faces_success(mock_mtcnn_class, tmp_path):
    mock_mtcnn = MagicMock()
    mock_mtcnn_class.return_value = mock_mtcnn
    
    mock_mtcnn.detect.return_value = (
        [[10, 10, 50, 50], [60, 60, 100, 100]],
        [0.99, 0.95]
    )

    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    
    img_path = dataset_dir / "set1/img1.jpg"
    img_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (200, 200))
    img.save(img_path)

    extract_faces_for_image(
        str(img_path),
        dataset_dir,
        "id1",
        "set1",
        "img1.jpg",
        mock_mtcnn
    )
    
    out_dir = dataset_dir / "faces" / "id1" / "set1"
    assert out_dir.exists()
    assert (out_dir / "img1_face1.jpg").exists()
    assert (out_dir / "img1_face2.jpg").exists()

@patch("tools.hegre_dataset.face_extraction.extract_all")
def test_cli_extract_faces(mock_extract_all, tmp_path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    with open(dataset_dir / "manifest.json", "w") as f:
        json.dump({}, f)

    ret = main(["extract-faces", "--dataset", str(dataset_dir)])
    assert ret == 0
