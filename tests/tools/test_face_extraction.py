from PIL import Image
import json
from unittest.mock import patch, MagicMock

from tools.hegre_dataset.face_extraction import get_square_box, extract_faces
from tools.hegre_dataset.cli import main

def test_get_square_box():
    # Basic square centering
    box = [100, 100, 200, 300] # w=100, h=200
    # Center is 150, 200. Max dim is 200. Square box side is 200.
    # So left=150-100=50, top=200-100=100, right=250, bottom=300
    sq_box = get_square_box(box, img_width=500, img_height=500)
    assert sq_box == (50, 100, 250, 300)
    
    # Clamping and shifting:
    # If the square goes off the left edge, shift it right
    box = [0, 100, 100, 300] # w=100, h=200. Center is 50, 200
    # sq_box before shift: (-50, 100, 150, 300)
    # shift right by 50: (0, 100, 200, 300)
    sq_box = get_square_box(box, img_width=500, img_height=500)
    assert sq_box == (0, 100, 200, 300)
    
    # Clamping and shifting to top
    box = [100, 0, 300, 100] # w=200, h=100. Center 200, 50. Max dim 200.
    # before shift: (100, -50, 300, 150)
    # shift down by 50: (100, 0, 300, 200)
    sq_box = get_square_box(box, img_width=500, img_height=500)
    assert sq_box == (100, 0, 300, 200)

    # Clamping both bounds (image too small to fit square)
    box = [10, 10, 90, 190] # w=80, h=180.
    # Center 50, 100. Max dim 180.
    # Try to make 180x180. Image is 150x150.
    # Since it can't fit 180x180 in 150x150, just clamp to image dimensions? Wait, it says shifting, clamping, no padding.
    # If max dim > img width or height, clamp to max possible or image edges.
    sq_box = get_square_box(box, img_width=150, img_height=150)
    # the box gets constrained to the image size.
    assert sq_box[0] >= 0 and sq_box[1] >= 0
    assert sq_box[2] <= 150 and sq_box[3] <= 150

@patch("tools.hegre_dataset.face_extraction.MTCNN")
def test_extract_faces_success(mock_mtcnn_class, tmp_path):
    mock_mtcnn = MagicMock()
    mock_mtcnn_class.return_value = mock_mtcnn
    
    # Mock return 2 faces
    mock_mtcnn.detect.return_value = (
        [[10, 10, 50, 50], [60, 60, 100, 100]],
        [0.99, 0.95]
    )

    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    manifest = {
        "id1": ["set1/img1.jpg"]
    }
    with open(dataset_dir / "manifest.json", "w") as f:
        json.dump(manifest, f)
        
    img_path = dataset_dir / "set1/img1.jpg"
    img_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (200, 200))
    img.save(img_path)

    extract_faces(dataset_dir)
    
    # Check outputs: images should be saved as face1.jpg, face2.jpg
    out_dir = dataset_dir / "id1" / "set1"
    assert out_dir.exists()
    assert (out_dir / "img1_face1.jpg").exists()
    assert (out_dir / "img1_face2.jpg").exists()
    
    # Check manifest is updated? Or maybe we just check idempotency
    
    # Run again, should be idempotent (not overwrite or error out if exist)
    # Change mock to see if it calls again
    mock_mtcnn.detect.reset_mock()
    extract_faces(dataset_dir)
    # By default, idempotency means it might skip or just overwrite cleanly. If it skips, detect won't be called.
    # Let's say we expect it to skip if output exists.
    # Or just don't crash.
    assert mock_mtcnn.detect.call_count == 0  # Assuming it skips if crops exist

@patch("tools.hegre_dataset.face_extraction.MTCNN")
def test_cli_extract_faces(mock_mtcnn_class, tmp_path):
    mock_mtcnn = MagicMock()
    mock_mtcnn_class.return_value = mock_mtcnn
    mock_mtcnn.detect.return_value = (None, None)
    
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    with open(dataset_dir / "manifest.json", "w") as f:
        json.dump({}, f)

    ret = main(["extract-faces", "--dataset", str(dataset_dir)])
    assert ret == 0
