"""Rectified Flow Matching — PyTorch implementation.

Trains a velocity-predicting model to push Gaussian noise to data distribution
along straight-line ODE paths. Uses autograd for end-to-end backprop.
"""
import torch
import torch.nn as nn


class RectifiedFlowMatching:
    """Rectified Flow Matching trainer/inference wrapper.
    
    Loss: L = E_t,x_0 [ || v_θ(x_t, t, cond) - (x_1 - x_0) ||^2 ]
    where x_t = (1-t)*x_0 + t*x_1, x_0 ~ N(0,I), t ~ U(0,1).
    """
    
    def __init__(self, model, d_output, n_steps=10, sigma_min=1e-4, device="cpu"):
        self.model = model
        self.d_output = d_output
        self.n_steps = n_steps
        self.sigma_min = sigma_min
        self.device = device
    
    def loss(self, x_1, cond):
        """Compute RFM loss for one batch.
        
        Args:
            x_1: (B, d_output) ground-truth vectors
            cond: (B, d_cond) conditioning (mean-pooled T5)
        
        Returns:
            scalar loss tensor (differentiable)
        """
        B = x_1.shape[0]
        x_0 = torch.randn(B, self.d_output, device=x_1.device, dtype=x_1.dtype)
        t = torch.rand(B, 1, device=x_1.device, dtype=x_1.dtype)
        x_t = (1 - t) * x_0 + t * x_1
        v_target = x_1 - x_0
        v_pred = self.model(x_t, t, cond)
        return torch.mean((v_pred - v_target) ** 2)
    
    def compute_held_out_loss(self, x_1, cond):
        """Compute loss without gradients (for evaluation)."""
        with torch.no_grad():
            return self.loss(x_1, cond)
    
    @torch.no_grad()
    def sample(self, cond, n_samples=1):
        """Generate samples via Euler ODE solver.
        
        Args:
            cond: (N, d_cond) conditioning, or None for unconditional
            n_samples: number of samples to generate (if cond is None, used directly)
        
        Returns:
            (n_samples, d_output) generated vectors
        """
        if cond is not None:
            n = cond.shape[0]
        else:
            n = n_samples if isinstance(n_samples, int) else 1
        B = n if n > 0 else 1
        
        x = torch.randn(B, self.d_output, device=self.device)
        dt = 1.0 / self.n_steps
        
        for step in range(self.n_steps):
            t_val = step * dt
            t_arr = torch.full((B, 1), t_val, device=self.device)
            v = self.model(x, t_arr, cond if cond is not None else torch.zeros(B, 0, device=self.device))
            x = x + v * dt
        
        return x
