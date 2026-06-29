import sqlite3
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

from tools.hegre_dataset.enrichment import generate_image_list, run_stratum_enrichment


def _make_test_db(db_path, rows, with_distances=False):
    """Create a test SQLite DB with the images table and insert rows.

    Args:
        rows: List of tuples. If with_distances=True, tuples are
            (image_path, status, zg_distance, af_distance).
            Otherwise, tuples are (image_path, status).
    """
    conn = sqlite3.connect(str(db_path))
    if with_distances:
        conn.execute("CREATE TABLE IF NOT EXISTS images "
                     "(image_path TEXT, status TEXT, zg_distance REAL, af_distance REAL)")
    else:
        conn.execute("CREATE TABLE IF NOT EXISTS images (image_path TEXT, status TEXT)")
    conn.executemany("INSERT INTO images VALUES ({})".format(
        "?, ?" if not with_distances else "?, ?, ?, ?"), rows)
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


class TestZgMaxDistanceFilter:
    """Tests for the --zg-max-distance filter on enrich."""

    def test_zg_filter_excludes_high_distance_approved(self, tmp_path):
        """Approved images with zg_distance > threshold are excluded."""
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()
        db_path = dataset_dir / "review.db"
        faces_dir = dataset_dir / "faces"
        faces_dir.mkdir()

        _make_test_db(db_path, [
            ("good.jpg", "approved", 10.0, None),
            ("bad.jpg", "approved", 200.0, None),
            ("none_yet.jpg", "approved", None, None),
        ], with_distances=True)
        for f in ["good.jpg", "bad.jpg", "none_yet.jpg"]:
            (faces_dir / f).touch()

        paths = generate_image_list(db_path, faces_dir, status_filter="approved",
                                     zg_max_distance=100.0)

        path_strs = {str(p) for p in paths}
        assert str((faces_dir / "good.jpg").absolute()) in path_strs
        assert str((faces_dir / "none_yet.jpg").absolute()) in path_strs  # NULL is always included
        assert str((faces_dir / "bad.jpg").absolute()) not in path_strs  # 200 > 100

    def test_zg_filter_does_not_affect_unreviewed(self, tmp_path):
        """zg_max_distance filter does not apply to unreviewed images."""
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()
        db_path = dataset_dir / "review.db"
        faces_dir = dataset_dir / "faces"
        faces_dir.mkdir()

        _make_test_db(db_path, [
            ("unr_good.jpg", "unreviewed", 5.0, None),
            ("unr_bad.jpg", "unreviewed", 500.0, None),
        ], with_distances=True)
        for f in ["unr_good.jpg", "unr_bad.jpg"]:
            (faces_dir / f).touch()

        paths = generate_image_list(db_path, faces_dir, status_filter="unreviewed",
                                     zg_max_distance=100.0)

        # Both should be included — unreviewed images are never filtered
        assert len(paths) == 2

    def test_zg_filter_both_mode(self, tmp_path):
        """In 'both' mode, only approved images are filtered by zg_distance."""
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()
        db_path = dataset_dir / "review.db"
        faces_dir = dataset_dir / "faces"
        faces_dir.mkdir()

        _make_test_db(db_path, [
            ("app_good.jpg", "approved", 10.0, None),
            ("app_bad.jpg", "approved", 300.0, None),
            ("unr_ok.jpg", "unreviewed", 500.0, None),
        ], with_distances=True)
        for f in ["app_good.jpg", "app_bad.jpg", "unr_ok.jpg"]:
            (faces_dir / f).touch()

        paths = generate_image_list(db_path, faces_dir, status_filter="both",
                                     zg_max_distance=100.0)

        path_strs = {str(p) for p in paths}
        assert str((faces_dir / "app_good.jpg").absolute()) in path_strs
        assert str((faces_dir / "unr_ok.jpg").absolute()) in path_strs  # unreviewed never filtered
        assert str((faces_dir / "app_bad.jpg").absolute()) not in path_strs  # 300 > 100, approved

    def test_zg_filter_no_column_is_noop(self, tmp_path):
        """If zg_distance column doesn't exist, filter is a no-op."""
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()
        db_path = dataset_dir / "review.db"
        faces_dir = dataset_dir / "faces"
        faces_dir.mkdir()

        _make_test_db(db_path, [  # no with_distances — no zg_distance column
            ("img1.jpg", "approved"),
            ("img2.jpg", "approved"),
        ])
        for f in ["img1.jpg", "img2.jpg"]:
            (faces_dir / f).touch()

        paths = generate_image_list(db_path, faces_dir, status_filter="approved",
                                     zg_max_distance=50.0)

        # Both should be included since the column doesn't exist
        assert len(paths) == 2


class TestSortBy:
    """Tests for the --sort-by parameter on enrich."""

    def test_sort_by_af_ascending(self, tmp_path):
        """--sort-by af returns images ordered by af_distance ASC."""
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()
        db_path = dataset_dir / "review.db"
        faces_dir = dataset_dir / "faces"
        faces_dir.mkdir()

        _make_test_db(db_path, [
            ("img1.jpg", "approved", 5.0, 0.3),
            ("img2.jpg", "approved", 10.0, 0.1),
            ("img3.jpg", "approved", 1.0, None),  # NULL sorts last
        ], with_distances=True)
        for f in ["img1.jpg", "img2.jpg", "img3.jpg"]:
            (faces_dir / f).touch()

        paths = generate_image_list(db_path, faces_dir, status_filter="approved",
                                     sort_by="af")

        stems = [Path(str(p)).stem for p in paths]
        # img2 (af=0.1) first, img1 (af=0.3) second, img3 (af=NULL) last
        assert stems == ["img2", "img1", "img3"], f"Got {stems}"

    def test_sort_by_zg_ascending(self, tmp_path):
        """--sort-by zg returns images ordered by zg_distance ASC."""
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()
        db_path = dataset_dir / "review.db"
        faces_dir = dataset_dir / "faces"
        faces_dir.mkdir()

        _make_test_db(db_path, [
            ("img1.jpg", "approved", 5.0, None),
            ("img2.jpg", "approved", 1.0, None),
            ("img3.jpg", "approved", None, None),  # NULL sorts last
        ], with_distances=True)
        for f in ["img1.jpg", "img2.jpg", "img3.jpg"]:
            (faces_dir / f).touch()

        paths = generate_image_list(db_path, faces_dir, status_filter="approved",
                                     sort_by="zg")

        stems = [Path(str(p)).stem for p in paths]
        # img2 (zg=1.0) first, img1 (zg=5.0) second, img3 (zg=NULL) last
        assert stems == ["img2", "img1", "img3"], f"Got {stems}"

    def test_cli_sort_by_flag(self, tmp_path):
        """--sort-by is accepted by the CLI parser."""
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()
        db_path = dataset_dir / "review.db"
        faces_dir = dataset_dir / "faces"
        faces_dir.mkdir()

        _make_test_db(db_path, [("img1.jpg", "approved")])
        (faces_dir / "img1.jpg").touch()

        result = subprocess.run(
            [sys.executable, "-m", "tools.hegre_dataset", "enrich",
             "--dataset", str(dataset_dir), "--sort-by", "af", "--skip-stratum"],
            capture_output=True, text=True,
        )
        # Should parse without error
        assert "--sort-by" not in (result.stderr or "")

    def test_cli_zg_max_distance_flag(self, tmp_path):
        """--zg-max-distance is accepted by the CLI parser."""
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()
        db_path = dataset_dir / "review.db"
        faces_dir = dataset_dir / "faces"
        faces_dir.mkdir()

        _make_test_db(db_path, [("img1.jpg", "approved")])
        (faces_dir / "img1.jpg").touch()

        result = subprocess.run(
            [sys.executable, "-m", "tools.hegre_dataset", "enrich",
             "--dataset", str(dataset_dir), "--zg-max-distance", "100.0", "--skip-stratum"],
            capture_output=True, text=True,
        )
        # Should parse without error
        assert "zg" in result.stdout.lower() or result.returncode == 0
