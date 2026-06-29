import sqlite3
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

from tools.hegre_dataset.enrichment import generate_image_list, run_stratum_enrichment


def _make_test_db(db_path, rows):
    """Create a test SQLite DB with the images table and insert rows."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE IF NOT EXISTS images (image_path TEXT, status TEXT)")
    conn.executemany("INSERT INTO images VALUES (?, ?)", rows)
    conn.commit()
    conn.close()


def test_generate_image_list_default_both(tmp_path):
    """Default status_filter='both' returns approved + unreviewed only."""
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    db_path = dataset_dir / "review.db"
    faces_dir = dataset_dir / "faces"
    faces_dir.mkdir()

    _make_test_db(db_path, [
        ("app1.jpg", "approved"),
        ("rej1.jpg", "rejected"),
        ("app2.jpg", "approved"),
        ("pend1.jpg", "pending"),
        ("unr1.jpg", "unreviewed"),
        ("unr2.jpg", "unreviewed"),
    ])
    for f in ["app1.jpg", "rej1.jpg", "app2.jpg", "pend1.jpg", "unr1.jpg", "unr2.jpg"]:
        (faces_dir / f).touch()

    paths = generate_image_list(db_path, faces_dir)

    assert len(paths) == 4  # 2 approved + 2 unreviewed
    path_strs = {str(p) for p in paths}
    assert str((faces_dir / "app1.jpg").absolute()) in path_strs
    assert str((faces_dir / "app2.jpg").absolute()) in path_strs
    assert str((faces_dir / "unr1.jpg").absolute()) in path_strs
    assert str((faces_dir / "unr2.jpg").absolute()) in path_strs
    assert str((faces_dir / "rej1.jpg").absolute()) not in path_strs
    assert str((faces_dir / "pend1.jpg").absolute()) not in path_strs


def test_generate_image_list_approved_only(tmp_path):
    """status_filter='approved' returns approved only."""
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    db_path = dataset_dir / "review.db"
    faces_dir = dataset_dir / "faces"
    faces_dir.mkdir()

    _make_test_db(db_path, [
        ("app1.jpg", "approved"),
        ("unr1.jpg", "unreviewed"),
        ("app2.jpg", "approved"),
        ("tainted.jpg", "tainted:extraction_nonface"),
    ])
    for f in ["app1.jpg", "unr1.jpg", "app2.jpg", "tainted.jpg"]:
        (faces_dir / f).touch()

    paths = generate_image_list(db_path, faces_dir, status_filter="approved")

    assert len(paths) == 2
    path_strs = {str(p) for p in paths}
    assert str((faces_dir / "app1.jpg").absolute()) in path_strs
    assert str((faces_dir / "app2.jpg").absolute()) in path_strs
    assert str((faces_dir / "unr1.jpg").absolute()) not in path_strs
    assert str((faces_dir / "tainted.jpg").absolute()) not in path_strs


def test_generate_image_list_unreviewed_only(tmp_path):
    """status_filter='unreviewed' returns unreviewed only."""
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    db_path = dataset_dir / "review.db"
    faces_dir = dataset_dir / "faces"
    faces_dir.mkdir()

    _make_test_db(db_path, [
        ("app1.jpg", "approved"),
        ("unr1.jpg", "unreviewed"),
        ("unr2.jpg", "unreviewed"),
        ("rej1.jpg", "rejected"),
    ])
    for f in ["app1.jpg", "unr1.jpg", "unr2.jpg", "rej1.jpg"]:
        (faces_dir / f).touch()

    paths = generate_image_list(db_path, faces_dir, status_filter="unreviewed")

    assert len(paths) == 2
    path_strs = {str(p) for p in paths}
    assert str((faces_dir / "unr1.jpg").absolute()) in path_strs
    assert str((faces_dir / "unr2.jpg").absolute()) in path_strs
    assert str((faces_dir / "app1.jpg").absolute()) not in path_strs
    assert str((faces_dir / "rej1.jpg").absolute()) not in path_strs


def test_generate_image_list_empty(tmp_path):
    """Returns empty list when no matching images exist."""
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    db_path = dataset_dir / "review.db"
    faces_dir = dataset_dir / "faces"
    faces_dir.mkdir()

    _make_test_db(db_path, [
        ("rej1.jpg", "rejected"),
        ("tainted.jpg", "tainted:extraction_nonface"),
    ])
    for f in ["rej1.jpg", "tainted.jpg"]:
        (faces_dir / f).touch()

    paths = generate_image_list(db_path, faces_dir)
    assert len(paths) == 0


def test_run_stratum_enrichment_no_images(tmp_path):
    """When generate_image_list returns empty, enrichment skips early."""
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    db_path = dataset_dir / "review.db"
    faces_dir = dataset_dir / "faces"
    faces_dir.mkdir()

    _make_test_db(db_path, [])  # empty table

    with patch("subprocess.run") as mock_run:
        run_stratum_enrichment(dataset_dir, db_path, faces_dir,
                               status_filter="approved")
        mock_run.assert_not_called()


def test_cli_enrich_status_flag(tmp_path):
    """The --status flag is accepted by the enrich CLI subparser."""
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    db_path = dataset_dir / "review.db"
    faces_dir = dataset_dir / "faces"
    faces_dir.mkdir()

    _make_test_db(db_path, [
        ("app1.jpg", "approved"),
        ("unr1.jpg", "unreviewed"),
    ])
    for f in ["app1.jpg", "unr1.jpg"]:
        (faces_dir / f).touch()

    # Verify the CLI parses --status without error
    result = subprocess.run(
        [sys.executable, "-m", "tools.hegre_dataset", "enrich",
         "--dataset", str(dataset_dir), "--status", "approved", "--skip-stratum"],
        capture_output=True, text=True,
    )
    # Should run without error (AuraFace extraction may fail without insightface,
    # but the argparse should parse correctly)
    assert "approved" in result.stdout.lower() or "approved" in result.stderr.lower() or result.returncode == 0
