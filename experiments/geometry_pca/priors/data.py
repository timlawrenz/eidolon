"""Prior training data loader — paired (T5, z_g) and (T5, AuraFace-LDA) datasets."""
import numpy as np
from pathlib import Path


Z_G_MAX_NORM = 25.0   # filter degenerate DWPose projections


class PriorDataset:
    """Numpy-backed dataset for Flow Matching Prior training.
    
    Args:
        t5_paths: list of paths to t5_hidden.npy files
        target_paths: list of paths to target .npy files (z_g or AuraFace-LDA),
                      OR list of pre-loaded numpy arrays (if from_arrays=True)
        pool_t5: if True, mean-pool T5 sequence (512,1024) → (1024,)
        from_arrays: if True, target_paths is a list of numpy arrays
    """
    
    def __init__(self, t5_paths, target_paths, pool_t5=True, from_arrays=False):
        self.t5_paths = t5_paths
        self.target_paths = target_paths
        self.pool_t5 = pool_t5
        self.from_arrays = from_arrays
        assert len(t5_paths) == len(target_paths)
    
    def __len__(self):
        return len(self.t5_paths)
    
    def __getitem__(self, idx):
        t5 = np.load(self.t5_paths[idx]).astype(np.float64)
        if self.pool_t5 and t5.ndim == 2:
            t5 = t5.mean(axis=0)  # (512,1024) → (1024,)
        if self.from_arrays:
            target = self.target_paths[idx]  # already an array
        else:
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


def build_ffhq_lda_dataset(ffhq_root="/mnt/nas-ai-models/training-data/ffhq", max_samples=None):
    """Build paired (T5, AuraFace-LDA) dataset from FFHQ.
    
    Applies clean_auraface() + project_to_lda() to each AuraFace vector.
    
    Args:
        ffhq_root: root of the FFHQ data tree
        max_samples: cap on number of pairs (for testing; None = load all)
    """
    from geometry_pca.auraface_preprocessing import clean_auraface, project_to_lda
    
    aura_dir = Path(ffhq_root) / "auraface"
    stratum_dir = Path(ffhq_root) / "stratum"
    
    # Use os.listdir on auraface for speed (no glob star over NAS)
    import os
    aura_files = [f for f in os.listdir(str(aura_dir)) if f.endswith('.npy')]
    aura_files.sort()
    
    t5_paths, lda_targets = [], []
    for af in aura_files:
        if max_samples and len(t5_paths) >= max_samples:
            break
        fid = af.replace('.npy', '')
        t5_f = stratum_dir / fid / "t5_hidden.npy"
        aura_f = aura_dir / af
        if t5_f.exists() and aura_f.exists():
            aura_vec = np.load(aura_f).astype(np.float64)
            cleaned = clean_auraface(aura_vec)
            lda = project_to_lda(cleaned)
            t5_paths.append(str(t5_f))
            lda_targets.append(lda.ravel())  # (64,)
    
    return PriorDataset(t5_paths, lda_targets, from_arrays=True)
