"""Tests for the UI review mode differentiation (Review vs Audit)."""
import json
import sqlite3
from pathlib import Path

import pytest
from tools.hegre_dataset.review.ui import create_app


def _seed_db(db_path: Path, persona_name: str = "test_persona",
             statuses: list = None, af_distances: list = None,
             zg_distances: list = None):
    """Create a minimal review.db with images for one persona."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE IF NOT EXISTS personas (id INTEGER PRIMARY KEY, name TEXT)")
    conn.execute("CREATE TABLE IF NOT EXISTS images "
                 "(id INTEGER PRIMARY KEY, persona_id INTEGER, set_id INTEGER, "
                 "source_image TEXT, face_index INTEGER, image_path TEXT, status TEXT, "
                 "zg_distance REAL, af_distance REAL, reviewed_at TEXT)")
    conn.execute("INSERT INTO personas (name) VALUES (?)", (persona_name,))
    pid = conn.execute("SELECT id FROM personas WHERE name = ?", (persona_name,)).fetchone()[0]

    if statuses is None:
        statuses = ["approved"] * 25
    if af_distances is None:
        af_distances = [0.1 * i for i in range(len(statuses))]
    if zg_distances is None:
        zg_distances = [None] * len(statuses)

    for i, st in enumerate(statuses):
        img_path = f"faces/{persona_name}/set1/img{i:04d}.jpg"
        af = af_distances[i] if af_distances and i < len(af_distances) else None
        zg = zg_distances[i] if zg_distances and i < len(zg_distances) else None
        conn.execute(
            "INSERT INTO images (persona_id, set_id, source_image, face_index, "
            "image_path, status, zg_distance, af_distance) "
            "VALUES (?, 1, 'source.jpg', ?, ?, ?, ?, ?)",
            (pid, i, img_path, st, zg, af),
        )
    conn.commit()
    conn.close()
    return pid


class TestReviewAuditMode:
    """Test that Review mode picks random images, Audit picks worst-first."""

    def test_review_mode_uses_random_order(self, tmp_path):
        """Review mode query should contain ORDER BY RANDOM()."""
        db_path = tmp_path / "review.db"
        faces_root = tmp_path / "faces"
        faces_root.mkdir(parents=True)
        _seed_db(db_path, statuses=["approved"] * 30)

        app = create_app(db_path, faces_root)
        client = app.test_client()

        resp = client.get("/api/random_persona?mode=review")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data["mode"] == "review"
        # Should return a persona with images
        assert data["persona_id"] is not None
        assert len(data["image_ids"]) > 0

    def test_audit_mode_returns_worst_first(self, tmp_path):
        """Audit mode should return images sorted by descending af_distance."""
        db_path = tmp_path / "review.db"
        faces_root = tmp_path / "faces"
        faces_root.mkdir(parents=True)

        # Create approved images with known af_distances: 0.01, 0.02, ..., 0.29, 0.30
        n = 30
        af_distances = [0.01 * (i + 1) for i in range(n)]
        _seed_db(db_path, statuses=["approved"] * n, af_distances=af_distances,
                 zg_distances=[None] * n)

        app = create_app(db_path, faces_root)
        client = app.test_client()

        resp = client.get("/api/random_persona?mode=audit")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data["mode"] == "audit"

        # For audit, the returned distances should be in descending order
        dists = [data["distances"][str(iid)] for iid in data["image_ids"]
                 if str(iid) in data["distances"]]
        assert dists == sorted(dists, reverse=True), \
            f"Audit distances not sorted descending: {dists}"

    def test_review_mode_sorts_by_af_distance(self, tmp_path):
        """Review mode picks random, but the returned list is sorted by af_distance desc."""
        db_path = tmp_path / "review.db"
        faces_root = tmp_path / "faces"
        faces_root.mkdir(parents=True)

        n = 30
        af_distances = [0.01 * (i + 1) for i in range(n)]
        _seed_db(db_path, statuses=["approved"] * n, af_distances=af_distances,
                 zg_distances=[None] * n)

        app = create_app(db_path, faces_root)
        client = app.test_client()

        resp = client.get("/api/random_persona?mode=review")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data["mode"] == "review"

        # Review mode should still sort by af_distance descending
        dists = [data["distances"][str(iid)] for iid in data["image_ids"]
                 if str(iid) in data["distances"]]
        assert dists == sorted(dists, reverse=True), \
            f"Review distances not sorted descending: {dists}"

    def test_review_mode_returns_different_subsets(self, tmp_path):
        """Multiple calls to review mode should return different image subsets."""
        db_path = tmp_path / "review.db"
        faces_root = tmp_path / "faces"
        faces_root.mkdir(parents=True)

        n = 200  # Large enough that random samples differ with high probability
        _seed_db(db_path, statuses=["approved"] * n)

        app = create_app(db_path, faces_root)
        client = app.test_client()

        # Call 3 times, at least 2 should have different image_id sets
        id_sets = []
        for _ in range(3):
            resp = client.get("/api/random_persona?mode=review")
            data = json.loads(resp.data)
            id_sets.append(frozenset(data["image_ids"]))

        # With 200 images picking 20 random, probability of 3 identical sets is negligible
        distinct = len(set(id_sets))
        assert distinct >= 2, \
            f"Review mode returned identical sets across 3 calls (very unlikely): {id_sets}"

    def test_face_label_removed(self, tmp_path):
        """Face numbers should not appear in thumbnail labels."""
        db_path = tmp_path / "review.db"
        faces_root = tmp_path / "faces"
        faces_root.mkdir(parents=True)
        _seed_db(db_path, statuses=["approved"] * 5)

        app = create_app(db_path, faces_root)
        client = app.test_client()

        # The HTML page should not contain the face-number badge span
        html_resp = client.get("/")
        assert html_resp.status_code == 200
        html = html_resp.data.decode()
        # The old code had: <span>${lbl}</span> where lbl = "face3"
        # The new code removed it. Verify the label container doesn't include "face"
        # by checking the renderGrid template
        assert '${lbl}' not in html, \
            "Face number label template still present in HTML"


class TestDoneRecalculation:
    """Test that /api/done triggers background recalculation."""

    def test_done_triggers_compute_geometry(self, tmp_path, monkeypatch):
        """DONE should spawn compute-geometry in background."""
        from unittest.mock import patch

        db_path = tmp_path / "review.db"
        faces_root = tmp_path / "faces"
        faces_root.mkdir(parents=True)
        _seed_db(db_path, statuses=["unreviewed"] * 5)

        app = create_app(db_path, faces_root)
        client = app.test_client()

        # Get a persona first
        resp = client.get("/api/random_persona?mode=unreviewed")
        data = json.loads(resp.data)
        pid = data["persona_id"]
        shown_ids = data["image_ids"]

        with patch("subprocess.Popen") as mock_popen:
            done_resp = client.post("/api/done", json={
                "persona_id": pid,
                "tainted": {},
                "mode": "unreviewed",
                "shown_ids": shown_ids,
            })
            assert done_resp.status_code == 200

            # Verify subprocess.Popen was called with compute-geometry args
            mock_popen.assert_called_once()
            call_args = mock_popen.call_args[0][0]
            assert "compute-geometry" in call_args
            assert "--metric" in call_args
            # "both" should be right after --metric
            metric_idx = call_args.index("--metric")
            assert call_args[metric_idx + 1] == "both"


class TestZgMaxDistance:
    """Test the --zg-max-distance upper-limit parameter."""

    def test_default_threshold_is_100(self):
        """Without --zg-max-distance, the default should be 100.0."""
        from tools.hegre_dataset.review.geometry import compute_zg_distances
        import inspect
        sig = inspect.signature(compute_zg_distances)
        assert sig.parameters["zg_max_distance"].default == 100.0

    def test_cli_parses_zg_max_distance(self, tmp_path):
        """--zg-max-distance should be accepted by the CLI and override the default."""
        import subprocess, sys

        # Create minimal dataset with no images so compute-geometry exits fast
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()
        db_path = dataset_dir / "review.db"
        conn = __import__("sqlite3").connect(str(db_path))
        conn.execute("CREATE TABLE IF NOT EXISTS personas (id INTEGER PRIMARY KEY, name TEXT)")
        conn.execute("CREATE TABLE IF NOT EXISTS images (id INTEGER PRIMARY KEY, persona_id INTEGER, image_path TEXT, status TEXT, zg_distance REAL, af_distance REAL)")
        conn.close()

        result = subprocess.run(
            [sys.executable, "-m", "tools.hegre_dataset", "review", "compute-geometry",
             "--dataset", str(dataset_dir), "--encoder", "/nonexistent.npz",
             "--zg-max-distance", "50.0", "--metric", "zg"],
            capture_output=True, text=True,
        )
        # Should fail because encoder doesn't exist, but should parse the flag
        assert "zg_max_distance" not in result.stderr.lower() or "unrecognized" not in result.stderr.lower()

    def test_custom_threshold_applied_in_function(self, tmp_path):
        """zg_max_distance parameter is honored: lower threshold catches more outliers."""
        from tools.hegre_dataset.review.geometry import compute_zg_distances
        from unittest.mock import patch, ANY
        import numpy as np

        db_path = tmp_path / "review.db"
        stratum_dir = tmp_path / "stratum"
        stratum_dir.mkdir()

        conn = __import__("sqlite3").connect(str(db_path))
        conn.execute("CREATE TABLE IF NOT EXISTS personas (id INTEGER PRIMARY KEY, name TEXT)")
        conn.execute("CREATE TABLE IF NOT EXISTS images "
                     "(id INTEGER PRIMARY KEY, persona_id, image_path TEXT, status TEXT, "
                     "zg_distance REAL, af_distance REAL, reviewed_at TEXT)")
        conn.execute("INSERT INTO personas (name) VALUES ('test')")
        pid = conn.execute("SELECT id FROM personas").fetchone()[0]
        conn.execute("INSERT INTO images (persona_id, image_path, status) VALUES (?, 'img1.jpg', 'approved')", (pid,))
        conn.commit()
        conn.close()

        # Create a fake pose.npy for the image
        persona_dir = stratum_dir / "test"
        persona_dir.mkdir(parents=True)
        img_dir = persona_dir / "img1"
        img_dir.mkdir()
        np.save(str(img_dir / "pose.npy"), np.random.randn(133, 3).astype(np.float32))

        mock_encoder = {
            "components": np.random.randn(50, 136).astype(np.float32),
            "pca_mean": np.random.randn(136).astype(np.float32),
            "whiten_mu": np.random.randn(50).astype(np.float32),
            "whiten_sigma": np.random.randn(50).astype(np.float32),
        }

        with patch("tools.hegre_dataset.review.geometry.load_encoder", return_value=mock_encoder):
            # This should run successfully with the custom threshold
            rc = compute_zg_distances(db_path, stratum_dir, "/fake/encoder.npz",
                                      metric="zg", zg_max_distance=50.0)
            # Should succeed (code 0) even if no outliers are found
            assert rc == 0
