import sqlite3
from tools.hegre_dataset.enrichment import generate_approved_list, run_stratum_enrichment

def test_generate_approved_list(tmp_path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    
    db_path = dataset_dir / "review.db"
    
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("CREATE TABLE images (image_path TEXT, status TEXT)")
    # Using relative paths typical of the review DB
    c.execute("INSERT INTO images VALUES ('test1.jpg', 'approved')")
    c.execute("INSERT INTO images VALUES ('test2.jpg', 'rejected')")
    c.execute("INSERT INTO images VALUES ('test3.jpg', 'approved')")
    c.execute("INSERT INTO images VALUES ('test4.jpg', 'pending')")
    conn.commit()
    conn.close()
    
    faces_dir = dataset_dir / "faces"
    faces_dir.mkdir()
    (faces_dir / "test1.jpg").touch()
    (faces_dir / "test2.jpg").touch()
    (faces_dir / "test3.jpg").touch()
    (faces_dir / "test4.jpg").touch()
    
    list_path = generate_approved_list(dataset_dir)
    
    assert list_path.exists()
    
    content = list_path.read_text().splitlines()
    assert len(content) == 2
    assert str((faces_dir / "test1.jpg").absolute()) in content
    assert str((faces_dir / "test3.jpg").absolute()) in content

def test_run_stratum_enrichment(tmp_path, mocker):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    list_path = dataset_dir / "stratum_approved_list.txt"
    list_path.write_text("dummy")
    
    mock_run = mocker.patch("subprocess.run")
    
    run_stratum_enrichment(list_path, dataset_dir)
    
    mock_run.assert_called_once()
    args, kwargs = mock_run.call_args
    cmd = args[0]
    
    assert cmd[0] == "stratum"
    assert cmd[1] == "process"
    assert "--passes" in cmd
    assert cmd[cmd.index("--passes") + 1] == "pose,seg,depth,normal"
    assert "--image-list" in cmd
    assert cmd[cmd.index("--image-list") + 1] == str(list_path)
    assert "--output" in cmd
    assert cmd[cmd.index("--output") + 1] == str(dataset_dir / "stratum")
