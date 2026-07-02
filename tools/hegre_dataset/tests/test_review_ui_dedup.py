"""Tests for subprocess deduplication in the review UI."""

import pytest
import subprocess
import sys
from unittest.mock import MagicMock, patch
from pathlib import Path

try:
    from tools.hegre_dataset.review.ui import _maybe_spawn_geometry_compute
except ImportError:
    _maybe_spawn_geometry_compute = None


@pytest.fixture(autouse=True)
def _reset_job_tracker():
    """Reset the module-level active job dict between tests."""
    import tools.hegre_dataset.review.ui as ui_mod
    ui_mod._active_geometry_jobs = {}


@pytest.mark.skipif(
    _maybe_spawn_geometry_compute is None,
    reason="_maybe_spawn_geometry_compute not implemented yet"
)
class TestGeometrySubprocessDedup:
    """Tests that rapid DONE clicks for the same persona don't spawn duplicates."""

    def test_first_call_spawns_subprocess(self, tmp_path):
        """A first DONE for a persona should spawn a subprocess."""
        faces_root = tmp_path
        encoder_path = tmp_path / "encoder.npz"
        encoder_path.write_bytes(b"")

        mock_proc = MagicMock(spec=subprocess.Popen)
        mock_proc.poll.return_value = None  # still running

        with patch("tools.hegre_dataset.review.ui.subprocess.Popen", return_value=mock_proc) as mock_popen:
            result = _maybe_spawn_geometry_compute(42, faces_root, encoder_path)

            mock_popen.assert_called_once()
            assert result is mock_proc

    def test_second_call_for_same_persona_is_skipped(self, tmp_path):
        """Second DONE for the same persona skips spawning while first is running."""
        faces_root = tmp_path
        encoder_path = tmp_path / "encoder.npz"
        encoder_path.write_bytes(b"")

        mock_proc = MagicMock(spec=subprocess.Popen)
        mock_proc.poll.return_value = None  # still running

        with patch("tools.hegre_dataset.review.ui.subprocess.Popen", return_value=mock_proc) as mock_popen:
            first = _maybe_spawn_geometry_compute(42, faces_root, encoder_path)
            second = _maybe_spawn_geometry_compute(42, faces_root, encoder_path)

            # Only one Popen call total
            assert mock_popen.call_count == 1
            assert first is mock_proc
            assert second is None  # skipped

    def test_different_personas_spawn_separately(self, tmp_path):
        """DONE for different personas each spawn their own subprocess."""
        faces_root = tmp_path
        encoder_path = tmp_path / "encoder.npz"
        encoder_path.write_bytes(b"")

        proc_a = MagicMock(spec=subprocess.Popen)
        proc_a.poll.return_value = None
        proc_b = MagicMock(spec=subprocess.Popen)
        proc_b.poll.return_value = None

        call_count = {"n": 0}

        def fake_popen(*args, **kwargs):
            call_count["n"] += 1
            return proc_a if call_count["n"] == 1 else proc_b

        with patch("tools.hegre_dataset.review.ui.subprocess.Popen", side_effect=fake_popen):
            r1 = _maybe_spawn_geometry_compute(42, faces_root, encoder_path)
            r2 = _maybe_spawn_geometry_compute(99, faces_root, encoder_path)

            assert r1 is proc_a
            assert r2 is proc_b
            assert call_count["n"] == 2

    def test_completed_job_allows_respawn(self, tmp_path):
        """After a job completes, a new one can be spawned for the same persona."""
        faces_root = tmp_path
        encoder_path = tmp_path / "encoder.npz"
        encoder_path.write_bytes(b"")

        dead_proc = MagicMock(spec=subprocess.Popen)
        dead_proc.poll.return_value = 0  # completed (exit code 0)

        new_proc = MagicMock(spec=subprocess.Popen)
        new_proc.poll.return_value = None  # still running

        call_count = {"n": 0}

        def fake_popen(*args, **kwargs):
            call_count["n"] += 1
            return dead_proc if call_count["n"] == 1 else new_proc

        with patch("tools.hegre_dataset.review.ui.subprocess.Popen", side_effect=fake_popen):
            # First spawn
            r1 = _maybe_spawn_geometry_compute(42, faces_root, encoder_path)
            assert r1 is dead_proc

            # Simulate the proc completing
            dead_proc.poll.return_value = 0

            # Second call — should detect completion and spawn a new one
            r2 = _maybe_spawn_geometry_compute(42, faces_root, encoder_path)
            assert r2 is new_proc
            assert call_count["n"] == 2
