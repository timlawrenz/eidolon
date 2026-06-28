"""AdaLN-ResNet MLP for 1D Rectified Flow Matching.

Pure NumPy implementation — no PyTorch dependency needed for 1D MLPs.
"""
import numpy as np


def _sinusoidal_embedding(t, dim, max_period=10000.0):
    """Sinusoidal positional embeddings for continuous timesteps.
    
    Args:
        t: (B, 1) timestep in [0, 1]
        dim: output dimension (must be even)
    
    Returns:
        (B, dim) sinusoidal embedding
    """
    half = dim // 2
    freqs = np.exp(-np.arange(half) * np.log(max_period) / (half - 1))
    args = t * freqs.reshape(1, -1)
    return np.concatenate([np.sin(args), np.cos(args)], axis=1)


def _layer_norm(x, eps=1e-6):
    """Layer normalization along the last axis."""
    mu = x.mean(axis=-1, keepdims=True)
    sigma = x.std(axis=-1, keepdims=True)
    return (x - mu) / (sigma + eps)


class AdaLNResNet:
    """Residual MLP with adaptive layer normalization conditioned on text + timestep.
    
    Args:
        d_in: input dimension (noise vector size, = d_output)
        d_out: output dimension (velocity vector size, = d_output)
        d_hidden: hidden dimension per block
        n_blocks: number of residual blocks
        d_cond: conditioning dimension (text embedding size)
        t_embed_dim: dimension of sinusoidal time embedding
    """
    
    def __init__(self, d_in, d_out, d_hidden=1024, n_blocks=12, d_cond=1024, t_embed_dim=64):
        self.d_in = d_in
        self.d_out = d_out
        self.d_hidden = d_hidden
        self.n_blocks = n_blocks
        self.d_cond = d_cond
        self.t_embed_dim = t_embed_dim
        
        rng = np.random.default_rng(42)
        
        # Input projection
        self.W_in = rng.normal(0, 1/np.sqrt(d_in), (d_in, d_hidden)).astype(np.float64)
        self.b_in = np.zeros(d_hidden, dtype=np.float64)
        
        # Conditioning MLP: [t_embed + cond] → per-block params
        cond_in = t_embed_dim + d_cond
        cond_hidden = d_hidden // 2
        self.W_cond1 = rng.normal(0, 1/np.sqrt(cond_in), (cond_in, cond_hidden)).astype(np.float64)
        self.b_cond1 = np.zeros(cond_hidden, dtype=np.float64)
        
        # Per-block params: scale, shift, bias for AdaLN + linear layers
        # Each block: LN_scale(d_hidden), LN_shift(d_hidden), 2x Linear weights
        self.blocks = []
        for i in range(n_blocks):
            block_in = d_hidden
            block = {
                # LN params (predicted by cond MLP)
                "cond_out": rng.normal(0, 0.01/np.sqrt(cond_hidden), (cond_hidden, block_in * 2)).astype(np.float64),
                "cond_bias": np.zeros(block_in * 2, dtype=np.float64),
                # Linear 1: block_in → d_hidden
                "W1": rng.normal(0, 1/np.sqrt(block_in), (block_in, d_hidden)).astype(np.float64),
                "b1": np.zeros(d_hidden, dtype=np.float64),
                # Linear 2: d_hidden → block_in (for residual)
                "W2": rng.normal(0, 1/np.sqrt(d_hidden), (d_hidden, block_in)).astype(np.float64),
                "b2": np.zeros(block_in, dtype=np.float64),
            }
            self.blocks.append(block)
        
        # Output projection
        self.W_out = rng.normal(0, 1/np.sqrt(d_hidden), (d_hidden, d_out)).astype(np.float64)
        self.b_out = np.zeros(d_out, dtype=np.float64)
    
    def _cond_forward(self, t_embed, cond):
        """Forward pass through the conditioning MLP → per-block scale/shift."""
        c = np.concatenate([t_embed, cond], axis=1)  # (B, t_embed+d_cond)
        c = np.maximum(0, c @ self.W_cond1 + self.b_cond1)  # ReLU
        return c  # (B, cond_hidden)
    
    def __call__(self, x, t, cond):
        """Forward pass.
        
        Args:
            x: (B, d_in) noisy vector
            t: (B, 1) timestep in [0, 1]
            cond: (B, d_cond) conditioning (mean-pooled T5)
        
        Returns:
            (B, d_out) predicted velocity
        """
        t_embed = _sinusoidal_embedding(t, self.t_embed_dim)  # (B, t_embed)
        cond_feat = self._cond_forward(t_embed, cond)          # (B, cond_hidden)
        
        h = np.maximum(0, x @ self.W_in + self.b_in)  # input projection, ReLU
        
        for block in self.blocks:
            # Predict AdaLN params for this block
            ln_params = cond_feat @ block["cond_out"] + block["cond_bias"]  # (B, block_in*2)
            scale = ln_params[:, :self.d_hidden]
            shift = ln_params[:, self.d_hidden:]
            
            # Adaptive LayerNorm
            h_norm = _layer_norm(h)
            h_mod = h_norm * (1 + scale) + shift
            
            # Residual block
            h_new = np.maximum(0, h_mod @ block["W1"] + block["b1"])  # SiLU-ish
            h_new = h_new @ block["W2"] + block["b2"]
            h = h + h_new  # residual
        
        return h @ self.W_out + self.b_out
