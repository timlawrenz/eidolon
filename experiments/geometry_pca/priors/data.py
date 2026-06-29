"""Prior training data loader — paired (T5, z_g) and (T5, AuraFace-LDA) datasets."""
import numpy as np
from pathlib import Path
import os


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
        self.raw_targets = None
        assert len(t5_paths) == len(target_paths)
    
    def __len__(self):
        return len(self.t5_paths)
    
    def __getitem__(self, idx):
        if self.from_arrays:
            t5 = self.t5_paths[idx]
            target = self.target_paths[idx]
        else:
            t5 = np.load(self.t5_paths[idx]).astype(np.float64)
            target = np.load(self.target_paths[idx]).astype(np.float64)
        if self.pool_t5 and t5.ndim == 2:
            t5 = t5.mean(axis=0)  # (512,1024) → (1024,)
        return t5, target


def build_ffhq_zg_dataset(ffhq_root="/mnt/nas-ai-models/training-data/ffhq", max_samples=None, 
                           skip_norm_check=False, preload=True):
    """Build paired (T5, z_g) dataset from FFHQ stratum and zg directories.
    
    Args:
        ffhq_root: root of the FFHQ data tree
        max_samples: cap on number of pairs (for testing; None = load all)
        skip_norm_check: skip L2 norm check (safe if extraction already filtered degenerate z_g)
        preload: if True, load all data into memory at build time (fast training, heavy RAM)
    """
    stratum_dir = Path(ffhq_root) / "stratum"
    zg_dir = Path(ffhq_root) / "zg"
    
    # Fast directory listing (avoids glob star over NAS)
    try:
        zg_dirs = sorted(os.listdir(str(zg_dir)))
    except FileNotFoundError:
        zg_dirs = []
    
    if preload:
        t5_arrays, zg_arrays = [], []
        for fid in zg_dirs:
            if max_samples and len(t5_arrays) >= max_samples:
                break
            zg_f = zg_dir / fid / "zg.npy"
            t5_f = stratum_dir / fid / "t5_hidden.npy"
            if not zg_f.exists() or not t5_f.exists():
                continue
            z = np.load(zg_f).astype(np.float64)
            if not skip_norm_check and np.linalg.norm(z) >= Z_G_MAX_NORM:
                continue
            t5 = np.load(t5_f).astype(np.float64)
            if t5.ndim == 2:
                t5 = t5.mean(axis=0)
            t5_arrays.append(t5)
            zg_arrays.append(z)
        
        print(f"  Preloaded {len(t5_arrays)} pairs into memory")
        return PriorDataset(t5_arrays, zg_arrays, pool_t5=False, from_arrays=True)
    
    # Slow path (for tests that want path-based loading)
    t5_paths, zg_paths = [], []
    for fid in zg_dirs:
        if max_samples and len(t5_paths) >= max_samples:
            break
        zg_f = zg_dir / fid / "zg.npy"
        t5_f = stratum_dir / fid / "t5_hidden.npy"
        if not zg_f.exists() or not t5_f.exists():
            continue
        if not skip_norm_check:
            z = np.load(zg_f)
            if np.linalg.norm(z) >= Z_G_MAX_NORM:
                continue
        t5_paths.append(str(t5_f))
        zg_paths.append(str(zg_f))
    
    return PriorDataset(t5_paths, zg_paths)


def _skip_slow(reason):
    """Decorator to skip slow tests that scan the full FFHQ dataset over NAS."""
    import pytest
    return pytest.mark.skip(reason=reason)


def build_ffhq_lda_dataset(ffhq_root="/mnt/nas-ai-models/training-data/ffhq", max_samples=None, preload=True):
    """Build paired (T5, AuraFace-LDA) dataset from FFHQ.
    
    Applies clean_auraface() + project_to_lda() to each AuraFace vector.
    
    Args:
        ffhq_root: root of the FFHQ data tree
        max_samples: cap on number of pairs (for testing; None = load all)
        preload: if True, load all data into memory at build time (fast training)
    """
    from geometry_pca.auraface_preprocessing import clean_auraface, project_to_lda
    
    aura_dir = Path(ffhq_root) / "auraface"
    stratum_dir = Path(ffhq_root) / "stratum"
    
    aura_files = [f for f in os.listdir(str(aura_dir)) if f.endswith('.npy')]
    aura_files.sort()
    
    if preload:
        t5_arrays, lda_arrays, raw_arrays = [], [], []
        for af in aura_files:
            if max_samples and len(t5_arrays) >= max_samples:
                break
            fid = af.replace('.npy', '')
            t5_f = stratum_dir / fid / "t5_hidden.npy"
            aura_f = aura_dir / af
            if t5_f.exists() and aura_f.exists():
                aura_vec = np.load(aura_f).astype(np.float64)
                cleaned = clean_auraface(aura_vec)
                lda = project_to_lda(cleaned).ravel()
                t5 = np.load(t5_f).astype(np.float64)
                if t5.ndim == 2:
                    t5 = t5.mean(axis=0)
                t5_arrays.append(t5)
                lda_arrays.append(lda)
                raw_arrays.append(aura_vec.ravel())
        print(f"  Preloaded {len(t5_arrays)} LDA pairs into memory")
        ds = PriorDataset(t5_arrays, lda_arrays, pool_t5=False, from_arrays=True)
        ds.raw_targets = raw_arrays
        return ds
    
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
