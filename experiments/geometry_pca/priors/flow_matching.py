"""Rectified Flow Matching for 1D continuous vectors.

Straight-line ODE: x_t = (1-t)*x_0 + t*x_1
Velocity target: v = x_1 - x_0
Model predicts velocity v_θ(x_t, t, cond); loss = MSE(v_θ, v).
"""
import numpy as np

class RectifiedFlowMatching:
    """Rectified Flow Matching trainer/inference wrapper for a velocity-predicting model."""
    
    def __init__(self, model, d_output, n_steps=10, sigma_min=1e-4):
        self.model = model
        self.d_output = d_output
        self.n_steps = n_steps
        self.sigma_min = sigma_min
    
    def loss(self, x_1, cond):
        """Compute RFM loss for one batch.
        
        Args:
            x_1: (B, d_output) ground-truth vectors
            cond: conditioning (e.g. T5 embeddings), passed to model
        
        Returns:
            scalar MSE loss between predicted and true velocity
        """
        B = x_1.shape[0]
        x_0 = np.random.randn(B, self.d_output).astype(x_1.dtype)
        t = np.random.rand(B, 1).astype(x_1.dtype)
        x_t = (1 - t) * x_0 + t * x_1
        v_target = x_1 - x_0
        v_pred = self.model(x_t, t, cond)
        return np.mean((v_pred - v_target) ** 2)
    
    def sample(self, cond, n_samples=1):
        """Generate samples via Euler ODE solver.
        
        Args:
            cond: conditioning for the model
            n_samples: number of samples to generate
        
        Returns:
            (n_samples, d_output) generated vectors
        """
        B = n_samples if isinstance(n_samples, int) else cond.shape[0] if hasattr(cond, 'shape') else 1
        B = B if B > 0 else 1
        x = np.random.randn(B, self.d_output).astype(np.float64)
        dt = 1.0 / self.n_steps
        
        for step in range(self.n_steps):
            t_val = step * dt
            t_arr = np.full((B, 1), t_val, dtype=np.float64)
            v = self.model(x, t_arr, cond)
            x = x + v * dt
        
        return x
