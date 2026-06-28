"""Tests for the PyTorch training stack — AdaLN-ResNet model + Flow Matching wrapper.

These tests verify the deliverable can actually train — the gap the NumPy scaffold
left open. Uses synthetic 2D data to keep tests fast and deterministic.
"""
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "experiments" / "geometry_pca"))


# =============================================================================
# Helpers
# =============================================================================

def _has_torch():
    try:
        import torch
        return True
    except ImportError:
        return False

requires_torch = pytest.mark.skipif(not _has_torch(), reason="PyTorch not installed")


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def synthetic_data(rng):
    """Synthetic 2D cluster data for quick training tests."""
    d = 2
    n = 500
    centroids = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=np.float64)
    labels = rng.integers(4, size=n)
    x_1 = centroids[labels] + rng.normal(0, 0.1, (n, d)).astype(np.float64)
    # Dummy T5 conditioning (all zeros — unconditional)
    cond = np.zeros((n, 1024), dtype=np.float64)
    train_n = 400
    return (x_1[:train_n], cond[:train_n]), (x_1[train_n:], cond[train_n:])


# =============================================================================
# Tests
# =============================================================================

class TestPyTorchTrainingStack:
    """Verify the FULL training stack: model → flow → train → held-out improvement."""

    @requires_torch
    def test_imports(self):
        """GREEN: PyTorch model and flow modules import without error."""
        import priors.models_torch as models
        import priors.flow_matching_torch as fm
        assert models.AdaLNResNet is not None
        assert fm.RectifiedFlowMatching is not None

    @requires_torch
    def test_model_is_trainable(self, synthetic_data):
        """RED (will fail until rewritten): AdaLN-ResNet params receive gradients
        and the model can reduce held-out loss."""
        import torch
        from priors.models_torch import AdaLNResNet
        from priors.flow_matching_torch import RectifiedFlowMatching
        
        (x_train, cond_train), (x_held, cond_held) = synthetic_data
        
        d_out = 2
        model = AdaLNResNet(d_in=d_out, d_out=d_out, d_hidden=64, n_blocks=4, d_cond=1024)
        rfm = RectifiedFlowMatching(model, d_output=d_out, device="cpu")
        
        # Convert data to torch
        x1 = torch.from_numpy(x_train).float()
        c = torch.from_numpy(cond_train).float()
        x1_held = torch.from_numpy(x_held).float()
        c_held = torch.from_numpy(cond_held).float()
        
        # Measure held-out loss BEFORE training
        loss_before = rfm.compute_held_out_loss(x1_held, c_held)
        
        # Train for 80 steps (enough to show progress on 2D synthetic data)
        opt = torch.optim.AdamW(model.parameters(), lr=5e-4)
        steps = 80
        for step in range(steps):
            loss = rfm.loss(x1, c)
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        # Measure held-out loss AFTER
        loss_after = rfm.compute_held_out_loss(x1_held, c_held)
        
        assert loss_after.item() < loss_before.item(), \
            f"loss_before={loss_before.item():.3f}, loss_after={loss_after.item():.3f}"

    @requires_torch
    def test_sampling_after_training(self, synthetic_data):
        """After training, the model should sample plausible vectors (not NaN, not zero)."""
        import torch
        from priors.models_torch import AdaLNResNet
        from priors.flow_matching_torch import RectifiedFlowMatching
        
        (x_train, cond_train), _ = synthetic_data
        d_out = 2
        model = AdaLNResNet(d_in=d_out, d_out=d_out, d_hidden=64, n_blocks=4, d_cond=1024)
        rfm = RectifiedFlowMatching(model, d_output=d_out, device="cpu")
        
        x1 = torch.from_numpy(x_train).float()
        c = torch.from_numpy(cond_train).float()
        
        opt = torch.optim.AdamW(model.parameters(), lr=5e-4)
        for _ in range(80):
            loss = rfm.loss(x1, c)
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        # Sample
        model.eval()
        samples = rfm.sample(c[:16], n_samples=16)
        assert samples.shape == (16, d_out)
        assert torch.isfinite(samples).all(), "samples contain NaN/inf"
        # Should not be all-zero (the model actually moves noise)
        assert not torch.allclose(samples, torch.zeros_like(samples))

    @requires_torch
    def test_zg_model_fits_small_batch(self):
        """The 50-d z_g model can train on a small real batch without crashing."""
        import torch
        from priors.models_torch import AdaLNResNet
        from priors.flow_matching_torch import RectifiedFlowMatching
        
        d_out = 50
        B = 32
        # Synthetic z_g-like data: ~N(0, 1) per dim, whitened
        x1 = torch.randn(B, d_out)
        cond = torch.randn(B, 1024)  # T5-like
        
        model = AdaLNResNet(d_in=d_out, d_out=d_out, d_hidden=256, n_blocks=6, d_cond=1024)
        rfm = RectifiedFlowMatching(model, d_output=d_out, device="cpu")
        
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        losses = []
        for _ in range(50):
            loss = rfm.loss(x1, cond)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        
        assert losses[-1] < losses[0], f"loss flat: {losses[0]:.3f} → {losses[-1]:.3f}"
