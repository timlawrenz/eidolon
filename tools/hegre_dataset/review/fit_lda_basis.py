"""
fit_lda_basis.py — Fit auraface_preprocess.npz and auraface_lda.npz.

Pools FFHQ + Hegre AuraFace vectors, fits PC1 direction (domain axis),
preserves the existing yaw direction (stable across fits), then fits
an LDA basis on cleaned Hegre identity vectors.

Yaw direction is preserved from the previous fit rather than recomputed
because recomputing requires loading 166k+ pose.npy files from NAS,
which is prohibitively slow. The yaw direction (head-pose cleanup) is
stable across dataset changes; PC1 (domain axis) and the LDA basis are
the components that benefit from refitting on cleaned data.

Algorithm:
  1. Load FFHQ AuraFace vectors (flat numbered .npy files)
  2. Load Hegre AuraFace vectors from approved images
  3. Pool FFHQ+Hegre → PCA → PC1 direction
  4. Load old yaw direction from existing preprocess.npz
  5. Save auraface_preprocess.npz (new pooled_mean + pc1, old yaw)
  6. Clean Hegre vectors (PC1 + yaw removal)
  7. LDA on cleaned Hegre vectors (80/20 persona split)
  8. Save auraface_lda.npz
"""
import os, sys, time, numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from tqdm import tqdm


def _load_ffhq_auraface(ffhq_root: Path, max_ffhq: int = None) -> np.ndarray:
    """Load FFHQ AuraFace vectors from flat numbered .npy files."""
    af_dir = ffhq_root / "auraface"
    if not af_dir.is_dir():
        print(f"  FFHQ auraface dir not found: {af_dir}")
        return np.empty((0, 512), dtype=np.float64)
    
    files = sorted(af_dir.glob("*.npy"), key=lambda p: int(p.stem))
    if max_ffhq:
        files = files[:max_ffhq]
    
    vectors = []
    for fp in tqdm(files, desc="  [FFHQ]"):
        try:
            v = np.load(fp).astype(np.float64)
            if v.shape == (512,):
                vectors.append(v)
        except Exception:
            continue
    return np.stack(vectors) if vectors else np.empty((0, 512), dtype=np.float64)


def _load_hegre_auraface(dataset_root: Path, max_per_persona: int = None) -> tuple[np.ndarray, np.ndarray]:
    """Load Hegre AuraFace vectors from approved images.
    
    Returns:
        vectors: (N, 512) float64
        pids:    (N,) int64 — persona IDs for LDA fitting
    """
    from tools.hegre_dataset.dataset import HegreDataset
    os.environ.setdefault('EIDOLON_SKIP_REVIEWDB_GUARD', '1')
    
    ds = HegreDataset(dataset_root)
    
    rows = ds.db.execute("""
        SELECT i.persona_id, i.image_path
        FROM images i
        WHERE i.status = 'approved'
        ORDER BY i.persona_id, i.image_path
    """).fetchall()
    
    vectors, pids = [], []
    seen_per_persona = defaultdict(int)
    
    for persona_id, img_path in tqdm(rows, desc="  [Hegre]"):
        if max_per_persona and seen_per_persona[persona_id] >= max_per_persona:
            continue
        
        af_path = dataset_root / "auraface" / img_path.replace(".jpg", ".npy")
        if not af_path.exists():
            continue
        
        try:
            v = np.load(af_path).astype(np.float64)
        except Exception:
            continue
        
        if v.shape != (512,):
            continue
        
        vectors.append(v)
        pids.append(persona_id)
        seen_per_persona[persona_id] += 1
    
    return np.stack(vectors), np.array(pids, dtype=np.int64)


def fit_all(
    ffhq_root: Path,
    dataset_root: Path,
    output_dir: Path,
    n_components: int = 64,
    heldout_frac: float = 0.20,
    seed: int = 42,
    max_ffhq: int = None,
    max_hegre_per_persona: int = None,
    overwrite: bool = False,
):
    """Main entry point — called by CLI."""
    output_dir.mkdir(parents=True, exist_ok=True)
    preprocess_path = output_dir / "auraface_preprocess.npz"
    lda_path = output_dir / "auraface_lda.npz"
    
    if not overwrite and preprocess_path.exists() and lda_path.exists():
        print("Both .npz files exist. Use --overwrite to replace.")
        return 0
    
    rng = np.random.RandomState(seed)
    t0 = time.time()
    
    # ============================================================
    # Phase 1: Load FFHQ
    # ============================================================
    print("=" * 60)
    print("Phase 1: Loading FFHQ AuraFace")
    print("=" * 60)
    ffhq_vecs = _load_ffhq_auraface(ffhq_root, max_ffhq)
    print(f"  FFHQ: {len(ffhq_vecs)} vectors, shape {ffhq_vecs.shape}")
    
    # ============================================================
    # Phase 2: Load Hegre
    # ============================================================
    print("=" * 60)
    print("Phase 2: Loading Hegre AuraFace")
    print("=" * 60)
    hegre_vecs, hegre_pids = _load_hegre_auraface(dataset_root, max_hegre_per_persona)
    n_personas = len(set(hegre_pids))
    print(f"  Hegre: {len(hegre_vecs)} vectors, {n_personas} personas")
    
    # ============================================================
    # Phase 3: Fit auraface_preprocess.npz
    # ============================================================
    print("=" * 60)
    print("Phase 3: Fitting auraface_preprocess.npz")
    print("=" * 60)
    
    pooled = np.concatenate([ffhq_vecs, hegre_vecs], axis=0)
    pooled_mean = pooled.mean(axis=0)
    pooled_centered = pooled - pooled_mean
    print(f"  Pooled: {len(pooled)} vectors (FFHQ {len(ffhq_vecs)} + Hegre {len(hegre_vecs)})")
    
    # PCA for PC1 direction
    print("  Fitting PCA ...")
    pca = PCA(n_components=1, random_state=seed)
    pca.fit(pooled_centered)
    pc1_direction = pca.components_[0].astype(np.float64)
    pc1_var = float(pca.explained_variance_ratio_[0])
    print(f"  PC1 explains {pc1_var:.4%} variance")
    
    # Yaw direction: load from existing (stable across fits, avoids NAS pose.npy scan)
    yaw_direction = None
    if preprocess_path.exists():
        old = np.load(preprocess_path)
        if "yaw_direction" in old:
            yaw_direction = old["yaw_direction"].astype(np.float64)
            print(f"  Loaded yaw_direction from existing preprocess.npz")
    
    if yaw_direction is None:
        # Fallback: create a zero yaw direction (identity-invariant direction doesn't exist)
        print("  WARNING: No existing yaw_direction found — using zero vector")
        yaw_direction = np.zeros(512, dtype=np.float64)
    
    # Verify PC1 · yaw ≈ 0 (orthogonality should be preserved)
    dot = abs(np.dot(pc1_direction, yaw_direction))
    print(f"  yaw_direction · pc1 = {dot:.2e}" + ("  ✓ orthogonal" if dot < 1e-4 else "  ⚠ not orthogonal"))
    
    np.savez_compressed(
        preprocess_path,
        pooled_mean=pooled_mean.astype(np.float64),
        pc1_direction=pc1_direction,
        yaw_direction=yaw_direction,
    )
    print(f"  Saved auraface_preprocess.npz ({preprocess_path.stat().st_size} bytes)"
          f" with keys: ['pooled_mean', 'pc1_direction', 'yaw_direction']")
    
    # ============================================================
    # Phase 4: Fit auraface_lda.npz
    # ============================================================
    print("=" * 60)
    print("Phase 4: Fitting auraface_lda.npz")
    print("=" * 60)
    print("  Cleaning Hegre vectors (PC1 + yaw removal) ...")
    
    # Clean: center, remove PC1, remove yaw
    hegre_cleaned = hegre_vecs - pooled_mean
    hegre_cleaned -= np.outer(hegre_cleaned @ pc1_direction, pc1_direction)
    hegre_cleaned -= np.outer(hegre_cleaned @ yaw_direction, yaw_direction)
    
    # L2 renormalize
    norms = np.linalg.norm(hegre_cleaned, axis=1, keepdims=True)
    hegre_cleaned = hegre_cleaned / (norms + 1e-12)
    
    n_unique = len(set(hegre_pids))
    print(f"  {n_unique} unique personas, {len(hegre_cleaned)} total images")
    
    # Split personas: 80% train, 20% held-out
    unique_pids = sorted(set(hegre_pids))
    rng.shuffle(unique_pids)
    split_idx = int(len(unique_pids) * (1 - heldout_frac))
    train_pids = set(unique_pids[:split_idx])
    
    train_mask = np.array([pid in train_pids for pid in hegre_pids])
    X_train = hegre_cleaned[train_mask]
    y_train = hegre_pids[train_mask]
    
    n_train_personas = len(set(y_train))
    n_train_images = len(X_train)
    print(f"  Fitting LDA: {n_train_images} images, {n_train_personas} personas")
    
    # Cap n_components by n_classes - 1
    actual_k = min(n_components, n_train_personas - 1)
    if actual_k < n_components:
        print(f"  Note: n_components capped at {actual_k} (n_classes-1={n_train_personas-1})")
    
    lda = LDA(solver='eigen', n_components=actual_k)
    lda.fit(X_train, y_train)
    
    # IMPORTANT: scalings_ is FULL (512, 512) for solver='eigen'; slice to top-k
    W = lda.scalings_[:, :actual_k].astype(np.float64)
    evals = lda.explained_variance_ratio_[:actual_k].astype(np.float64)
    
    print(f"  LDA eigenvalues range: {evals[0]:.4f}–{evals[-1]:.4f}")
    
    np.savez_compressed(
        lda_path,
        lda_basis=W,
        lda_eigenvalues=evals,
        pooled_mean=pooled_mean.astype(np.float64),
        n_components=np.array(actual_k),
    )
    print(f"  Saved auraface_lda.npz ({lda_path.stat().st_size} bytes)"
          f" with keys: ['lda_basis', 'lda_eigenvalues', 'pooled_mean', 'n_components']")
    
    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s")
    return 0