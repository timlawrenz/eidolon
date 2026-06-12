import json
import pytest
from tools.hegre_dataset.identity import idkey, discover_identities, build_manifest, save_manifest
from tools.hegre_dataset.cli import main

def test_idkey():
    assert idkey("darina-l") == "darina-l"
    assert idkey("keity-climbing") == "keity"
    assert idkey("muriel") == "muriel"

@pytest.fixture
def mock_ground_truth(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    
    # darina-l: 3 sets
    (source_dir / "01_darina-l").mkdir()
    (source_dir / "02_darina-l").mkdir()
    (source_dir / "03_darina-l").mkdir()
    
    # keity: 2 sets (should be filtered out if min_sets=3)
    (source_dir / "04_keity-climbing").mkdir()
    (source_dir / "05_keity-pool").mkdir()

    # muriel: 3 sets
    (source_dir / "06_muriel").mkdir()
    (source_dir / "07_muriel").mkdir()
    (source_dir / "08_muriel").mkdir()
    
    # Create some dummy files
    (source_dir / "01_darina-l" / "img1.jpg").touch()
    (source_dir / "01_darina-l" / "img2.png").touch()
    (source_dir / "01_darina-l" / "_ignored.jpg").touch()
    (source_dir / "01_darina-l" / "doc.txt").touch()

    return source_dir

def test_discover_identities(mock_ground_truth):
    identities = discover_identities(mock_ground_truth, min_sets=3)
    
    assert "darina-l" in identities
    assert "muriel" in identities
    assert "keity" not in identities
    
    assert len(identities["darina-l"]) == 3
    assert len(identities["muriel"]) == 3

def test_build_manifest(mock_ground_truth):
    identities = {"darina-l": ["01_darina-l"]}
    manifest = build_manifest(mock_ground_truth, identities)
    
    assert "darina-l" in manifest
    assert len(manifest["darina-l"]) == 2
    
    entry_img1 = next(item for item in manifest["darina-l"] if item["filename"] == "img1.jpg")
    assert entry_img1["set_slug"] == "darina-l"
    assert "image_path" in entry_img1

def test_save_manifest(tmp_path):
    manifest = {"test": [{"foo": "bar"}]}
    path = save_manifest(manifest, tmp_path)
    assert path.exists()
    assert path.name == "manifest.json"
    
    with open(path) as f:
        data = json.load(f)
    assert data == manifest

def test_cli_discover(mock_ground_truth, tmp_path):
    out_dir = tmp_path / "dataset"
    args = [
        "discover",
        "--source", str(mock_ground_truth),
        "--dataset", str(out_dir),
        "--min-sets", "3"
    ]
    
    ret = main(args)
    assert ret == 0
    
    manifest_path = out_dir / "manifest.json"
    assert manifest_path.exists()
    
    with open(manifest_path) as f:
        manifest = json.load(f)
        
    assert "darina-l" in manifest
    assert "muriel" in manifest
    assert "keity" not in manifest
