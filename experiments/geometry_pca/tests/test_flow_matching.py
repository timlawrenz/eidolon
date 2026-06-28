"""Tests for Rectified Flow Matching Prior — core flow, model, and data loading."""
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "experiments" / "geometry_pca"))

from priors.flow_matching import RectifiedFlowMatching
from priors.models import AdaLNResNet
from priors.data import PriorDataset, build_ffhq_zg_dataset, Z_G_MAX_NORM


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def dummy_model():
    """A simple linear model for testing the flow matching wrapper."""
    class LinearModel:
        def __init__(self):
            self.d_in = 2
            self.d_out = 2
        def __call__(self, x, t, cond):
            return np.zeros_like(x)  # always predicts zero velocity (stays put)
    return LinearModel()


@pytest.fixture
def dummy_cond(rng):
    return rng.normal(0, 1, (64, 16)).astype(np.float64)


# =============================================================================
# Task B1: Flow Matching core
# =============================================================================

class TestRectifiedFlowMatching:
    
    def test_import(self):
        """GREEN: module imports successfully."""
        assert RectifiedFlowMatching is not None
    
    def test_loss_shape(self, dummy_model, dummy_cond):
        """loss() returns a scalar float."""
        rfm = RectifiedFlowMatching(dummy_model, d_output=2)
        x_1 = np.random.randn(64, 2).astype(np.float64)
        loss = rfm.loss(x_1, dummy_cond)
        assert np.isscalar(loss) or (isinstance(loss, (float, np.floating)) and not hasattr(loss, 'shape'))
    
    def test_loss_decreases_with_capacity(self, rng):
        """A model trained on an UNCONDITIONED flow should reduce loss.
        
        Note: a linear model with only [x | sin(t*pi)] → d has limited capacity.
        The test verifies loss decreases at all, not strict convergence rates.
        """
        
        class LearnableModel:
            def __init__(self, d):
                self.d = d
                self.W = rng.normal(0, 0.01, (d + 1, d)).astype(np.float64)
            def __call__(self, x, t, cond):
                inp = np.concatenate([x, np.sin(t * np.pi)], axis=1)
                return inp @ self.W
        
        d = 2; model = LearnableModel(d); rfm = RectifiedFlowMatching(model, d_output=d)
        n = 200
        centroids = np.array([[1,1], [-1,1], [-1,-1], [1,-1]], dtype=np.float64)
        labels = rng.integers(4, size=n)
        x_1 = centroids[labels] + rng.normal(0, 0.1, (n, d)).astype(np.float64)
        cond = np.zeros((n, 1))
        
        loss_before = rfm.loss(x_1, cond)
        
        lr = 0.01
        for _ in range(800):
            B = 64; idx = rng.integers(n, size=B)
            x_batch = x_1[idx]
            x_0 = rng.standard_normal((B, d)).astype(np.float64)
            t = rng.random((B, 1)).astype(np.float64)
            x_t = (1 - t) * x_0 + t * x_batch
            v_target = x_batch - x_0
            inp = np.concatenate([x_t, np.sin(t * np.pi)], axis=1)
            v_pred = inp @ model.W
            model.W -= lr * 2 * inp.T @ (v_pred - v_target) / B
        
        loss_after = rfm.loss(x_1, cond)
        assert loss_after < loss_before, f"loss_before={loss_before:.3f}, loss_after={loss_after:.3f}"
    
    def test_sample_shape(self, dummy_model, dummy_cond):
        """sample() returns vectors of correct shape."""
        rfm = RectifiedFlowMatching(dummy_model, d_output=3, n_steps=10)
        out = rfm.sample(dummy_cond, n_samples=4)
        assert out.shape == (4, 3)
    
    def test_sample_finite(self, dummy_model, dummy_cond):
        """sample() output is finite."""
        rfm = RectifiedFlowMatching(dummy_model, d_output=2, n_steps=10)
        out = rfm.sample(dummy_cond, n_samples=8)
        assert np.isfinite(out).all()


# =============================================================================
# Task B3: AdaLN-ResNet model
# =============================================================================

class TestAdaLNResNet:
    
    def test_import(self):
        """Module imports."""
        assert AdaLNResNet is not None
    
    def test_forward_shape(self, rng):
        """Forward pass returns correct shape."""
        model = AdaLNResNet(d_in=50, d_out=50, d_hidden=128, n_blocks=4, d_cond=1024)
        B = 8
        x = rng.normal(0, 1, (B, 50)).astype(np.float64)
        t = rng.random((B, 1)).astype(np.float64)
        cond = rng.normal(0, 1, (B, 1024)).astype(np.float64)
        out = model(x, t, cond)
        assert out.shape == (B, 50)
    
    def test_forward_not_identity(self, rng):
        """Forward pass is NOT the identity function (model does something)."""
        model = AdaLNResNet(d_in=10, d_out=10, d_hidden=64, n_blocks=4, d_cond=32)
        B = 4
        x = rng.normal(0, 1, (B, 10)).astype(np.float64)
        t = rng.random((B, 1)).astype(np.float64)
        cond = rng.normal(0, 1, (B, 32)).astype(np.float64)
        out = model(x, t, cond)
        # With random initialization, output should NOT equal input
        assert not np.allclose(out, x, atol=1e-4)
    
    def test_conditional_difference(self, rng):
        """Different conditioning produces different outputs."""
        model = AdaLNResNet(d_in=10, d_out=10, d_hidden=64, n_blocks=4, d_cond=32)
        B = 4
        x = rng.normal(0, 1, (B, 10)).astype(np.float64)
        t = rng.random((B, 1)).astype(np.float64)
        cond_a = rng.normal(0, 1, (B, 32)).astype(np.float64)
        cond_b = rng.normal(0, 1, (B, 32)).astype(np.float64)
        out_a = model(x, t, cond_a)
        out_b = model(x, t, cond_b)
        assert not np.allclose(out_a, out_b, atol=1e-4)


# =============================================================================
# Task B5: Data loading
# =============================================================================

class TestPriorDataset:
    
    def test_import(self):
        """Module imports."""
        assert PriorDataset is not None
    
    def test_ffhq_dataset_builds(self):
        """build_ffhq_zg_dataset() returns a non-empty dataset from the first files."""
        import os
        zg_root = "/mnt/nas-ai-models/training-data/ffhq/zg"
        dirs = sorted(os.listdir(zg_root))[:200]
        t5_paths, zg_paths = [], []
        for fid in dirs:
            zg_f = f"{zg_root}/{fid}/zg.npy"
            t5_f = f"/mnt/nas-ai-models/training-data/ffhq/stratum/{fid}/t5_hidden.npy"
            if Path(zg_f).exists() and Path(t5_f).exists():
                z = np.load(zg_f)
                if np.linalg.norm(z) < Z_G_MAX_NORM:
                    t5_paths.append(t5_f)
                    zg_paths.append(zg_f)
        ds = PriorDataset(t5_paths, zg_paths)
        assert len(ds) > 0, "FFHQ dataset should have samples"
        t5, zg = ds[0]
        assert t5.shape == (1024,), f"T5 shape {t5.shape}"
        assert zg.shape == (50,), f"z_g shape {zg.shape}"
    
    def test_ffhq_dataset_no_degenerate_zg(self):
        """All z_g vectors have L2 norm below Z_G_MAX_NORM."""
        import os
        zg_root = "/mnt/nas-ai-models/training-data/ffhq/zg"
        dirs = sorted(os.listdir(zg_root))[:500]
        for fid in dirs:
            zg_f = f"{zg_root}/{fid}/zg.npy"
            if Path(zg_f).exists():
                z = np.load(zg_f)
                assert np.linalg.norm(z) < Z_G_MAX_NORM, f"{zg_f} L2 norm {np.linalg.norm(z):.1f} exceeds {Z_G_MAX_NORM}"
    
    def test_ffhq_dataset_finite(self):
        """All loaded tensors are finite."""
        import os
        zg_root = "/mnt/nas-ai-models/training-data/ffhq/zg"
        dirs = sorted(os.listdir(zg_root))[:100]
        for fid in dirs:
            zg_f = f"{zg_root}/{fid}/zg.npy"
            t5_f = f"/mnt/nas-ai-models/training-data/ffhq/stratum/{fid}/t5_hidden.npy"
            if not Path(zg_f).exists() or not Path(t5_f).exists():
                continue
            t5 = np.load(t5_f).astype(np.float64)
            z = np.load(zg_f).astype(np.float64)
            assert np.isfinite(t5).all(), f"t5 {zg_f} not finite"
            assert np.isfinite(z).all(), f"zg {zg_f} not finite"
    
    @pytest.mark.skip(reason="full 70k NAS scan — run manually when verifying dataset integrity")
    def test_ffhq_dataset_has_many_samples(self):
        """FFHQ should yield ~70k paired samples (full scan for count only)."""
        ds = build_ffhq_zg_dataset()
        assert len(ds) > 60000, f"expected >60k FFHQ pairs, got {len(ds)}"


# =============================================================================
# Phase C: AuraFace-LDA data loading
# =============================================================================

class TestAuraFaceLDADataset:
    
    def test_import_builder(self):
        """build_ffhq_lda_dataset is importable."""
        from priors.data import build_ffhq_lda_dataset
        assert build_ffhq_lda_dataset is not None
    
    def test_lda_dataset_builds(self):
        """build_ffhq_lda_dataset returns a non-empty dataset (first 200 files)."""
        from priors.data import build_ffhq_lda_dataset
        ds = build_ffhq_lda_dataset(max_samples=200)
        assert len(ds) > 0, "LDA dataset should have samples"
    
    def test_lda_dataset_pair_shapes(self):
        """Each item returns (T5_1024, LDA_64)."""
        from priors.data import build_ffhq_lda_dataset
        ds = build_ffhq_lda_dataset(max_samples=200)
        t5, lda = ds[0]
        assert t5.shape == (1024,), f"T5 shape {t5.shape}"
        assert lda.shape == (64,), f"LDA shape {lda.shape}"
    
    def test_lda_dataset_finite(self):
        """All loaded LDA vectors are finite."""
        from priors.data import build_ffhq_lda_dataset
        ds = build_ffhq_lda_dataset(max_samples=100)
        for i in range(min(100, len(ds))):
            t5, lda = ds[i]
            assert np.isfinite(t5).all(), f"t5[{i}] not finite"
            assert np.isfinite(lda).all(), f"lda[{i}] not finite"
    
    def test_lda_dataset_nonzero(self):
        """LDA vectors are not all-zero (they carry identity signal)."""
        from priors.data import build_ffhq_lda_dataset
        ds = build_ffhq_lda_dataset(max_samples=100)
        all_norms = []
        for i in range(min(100, len(ds))):
            _, lda = ds[i]
            all_norms.append(np.linalg.norm(lda))
        mean_norm = np.mean(all_norms)
        assert mean_norm > 0.1, f"LDA vectors too close to zero (mean norm {mean_norm:.3f})"
