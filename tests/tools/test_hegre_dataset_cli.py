import subprocess
import sys

def test_hegre_dataset_cli_stubs():
    """Test that the CLI stubs exist and return not yet implemented."""
    for cmd in ["discover", "extract-faces", "review"]:
        result = subprocess.run(
            [sys.executable, "-m", "tools.hegre_dataset", cmd],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        assert "not yet implemented" in result.stdout
