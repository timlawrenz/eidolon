"""Prior training data loader — paired (T5, z_g) and (T5, AuraFace-LDA) datasets."""
import numpy as np
from pathlib import Path


Z_G_MAX_NORM = 25.0   # filter degenerate DWPose projections


class PriorDataset:
    """Numpy-backed dataset for Flow Matching Prior training.
    
    Args:
        t5_paths: list of paths to t5_hidden.npy files
        target_paths: list of paths to target .npy files (z_g or AuraFace-LDA)
        pool_t5: if True, mean-pool T5 sequence (512,1024) → (1024,)
    """
    
    def __init__(self, t5_paths, target_paths, pool_t5=True):
        self.t5_paths = t5_paths
        self.target_paths = target_paths
        self.pool_t5 = pool_t5
        assert len(t5_paths) == len(target_paths)
    
    def __len__(self):
        return len(self.t5_paths)
    
    def __getitem__(self, idx):
        t5 = np.load(self.t5_paths[idx]).astype(np.float64)
        if self.pool_t5 and t5.ndim == 2:
            t5 = t5.mean(axis=0)  # (512,1024) → (1024,)
        target = np.load(self.target_paths[idx]).astype(np.float64)
        return t5, target


def build_ffhq_zg_dataset(ffhq_root="/mnt/nas-ai-models/training-data/ffhq", max_samples=None, shuffle=True):
    """Build paired (T5, z_g) dataset from FFHQ stratum and zg directories.
    
    Excludes degenerate z_g vectors (L2 norm > Z_G_MAX_NORM).
    
    Args:
        ffhq_root: root of the FFHQ data tree
        max_samples: cap on number of pairs (for testing; None = load all)
        shuffle: if True, shuffle file order (set False for deterministic tests)
    """
    stratum_dir = Path(ffhq_root) / "stratum"
    zg_dir = Path(ffhq_root) / "zg"
    
    # Collect valid pairs — glob is fast for discovery, norm check is the cost
    zg_files = sorted(zg_dir.glob("*/zg.npy"))
    if shuffle:
        import random
        random.shuffle(zg_files)
    
    t5_paths, zg_paths = [], []
    for zg_f in zg_files:
        if max_samples and len(t5_paths) >= max_samples:
            break
        fid = zg_f.parent.name
        t5_f = stratum_dir / fid / "t5_hidden.npy"
        if t5_f.exists():
            z = np.load(zg_f)
            if np.linalg.norm(z) < Z_G_MAX_NORM:
                t5_paths.append(str(t5_f))
                zg_paths.append(str(zg_f))
    
    return PriorDataset(t5_paths, zg_paths)


def _skip_slow(reason):
    """Decorator to skip slow tests that scan the full FFHQ dataset over NAS."""
    import pytest
    return pytest.mark.skip(reason=reason)
