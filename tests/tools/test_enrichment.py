import sqlite3
from tools.hegre_dataset.enrichment import generate_image_list, run_stratum_enrichment

def test_generate_image_list(tmp_path):
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
    c.execute("INSERT INTO images VALUES ('test5.jpg', 'unreviewed')")
    c.execute("INSERT INTO images VALUES ('test6.jpg', 'unreviewed')")
    conn.commit()
    conn.close()
    
    faces_dir = dataset_dir / "faces"
    faces_dir.mkdir()
    (faces_dir / "test1.jpg").touch()
    (faces_dir / "test2.jpg").touch()
    (faces_dir / "test3.jpg").touch()
    (faces_dir / "test4.jpg").touch()
    (faces_dir / "test5.jpg").touch()
    (faces_dir / "test6.jpg").touch()
    
    paths = generate_image_list(db_path, faces_dir)
    
    # Should return 4 paths: 2 approved + 2 unreviewed
    assert len(paths) == 4
    assert str((faces_dir / "test1.jpg").absolute()) in [str(p) for p in paths]
    assert str((faces_dir / "test3.jpg").absolute()) in [str(p) for p in paths]
    assert str((faces_dir / "test5.jpg").absolute()) in [str(p) for p in paths]
    assert str((faces_dir / "test6.jpg").absolute()) in [str(p) for p in paths]
    # Rejected and pending should not be included
    assert str((faces_dir / "test2.jpg").absolute()) not in [str(p) for p in paths]
    assert str((faces_dir / "test4.jpg").absolute()) not in [str(p) for p in paths]

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
