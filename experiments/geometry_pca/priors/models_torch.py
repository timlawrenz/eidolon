"""AdaLN-ResNet MLP for 1D Rectified Flow Matching — PyTorch implementation.

Trainable via autograd + AdamW. Same architecture as the NumPy scaffold but
with nn.Module, proper parameter registration, and differentiable forward.
"""
import torch
import torch.nn as nn
import math


class SinusoidalEmbedding(nn.Module):
    """Sinusoidal positional embeddings for continuous timesteps."""
    def __init__(self, dim, max_period=10000.0):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.register_buffer('freqs', torch.exp(
            -torch.arange(dim // 2).float() * math.log(max_period) / (dim // 2 - 1)
        ))

    def forward(self, t):
        # t: (B, 1) in [0, 1]
        args = t @ self.freqs.view(1, -1)           # (B, 1) @ (1, dim//2) → (B, dim//2)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=1)  # (B, dim)


class AdaGN(nn.Module):
    """Adaptive Group Normalization — scale/shift predicted from conditioning.
    
    Uses GroupNorm (groups=1 = LayerNorm) so the normalization is self-contained
    per sample — no cross-sample stats leakage.
    """
    def __init__(self, dim, cond_dim, eps=1e-6):
        super().__init__()
        self.norm = nn.GroupNorm(1, dim, eps=eps)
        self.scale_shift = nn.Linear(cond_dim, dim * 2)
        # Initialize scale to 1, shift to 0
        nn.init.zeros_(self.scale_shift.weight)
        nn.init.zeros_(self.scale_shift.bias)

    def forward(self, x, cond):
        # x: (B, dim), cond: (B, cond_dim)
        scale_shift = self.scale_shift(cond)  # (B, dim*2)
        scale, shift = scale_shift.chunk(2, dim=1)
        x = self.norm(x.unsqueeze(-1)).squeeze(-1)  # GroupNorm needs (B, C, 1)
        return x * (1 + scale) + shift


class ResidualBlock(nn.Module):
    """Residual block: AdaGN → Linear → SiLU → Linear → +residual."""
    def __init__(self, dim, cond_dim, expansion=1):
        super().__init__()
        inner_dim = dim * expansion
        self.norm = AdaGN(dim, cond_dim)
        self.linear1 = nn.Linear(dim, inner_dim)
        self.linear2 = nn.Linear(inner_dim, dim)

    def forward(self, x, cond):
        h = self.norm(x, cond)
        h = self.linear1(h)
        h = torch.nn.functional.silu(h)
        h = self.linear2(h)
        return x + h


class AdaLNResNet(nn.Module):
    """Residual MLP with adaptive normalization conditioned on text + timestep.
    
    Args:
        d_in: input dimension (noise vector = d_output)
        d_out: output dimension (velocity vector = d_output)
        d_hidden: hidden dimension per block
        n_blocks: number of residual blocks
        d_cond: conditioning dimension (T5 text embedding)
        t_embed_dim: time embedding dimension
    """
    def __init__(self, d_in, d_out, d_hidden=1024, n_blocks=12, d_cond=1024, t_embed_dim=64):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        
        # Time embedding
        self.time_embed = SinusoidalEmbedding(t_embed_dim)
        
        # Conditioning MLP: [t_embed | cond] → cond_hidden
        cond_in = t_embed_dim + d_cond
        cond_hidden = d_hidden
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_in, cond_hidden),
            nn.SiLU(),
            nn.Linear(cond_hidden, cond_hidden),
        )
        
        # Input projection: d_in → d_hidden
        self.proj_in = nn.Linear(d_in, d_hidden)
        
        # Residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(d_hidden, cond_hidden) for _ in range(n_blocks)
        ])
        
        # Output projection (random init — honest baseline, not zero-biased)
        self.proj_out = nn.Linear(d_hidden, d_out)

    def forward(self, x, t, cond):
        """Forward pass.
        
        Args:
            x: (B, d_in) noisy vector at time t
            t: (B, 1) timestep in [0, 1]
            cond: (B, d_cond) T5 text conditioning
        
        Returns:
            (B, d_out) predicted velocity
        """
        t_emb = self.time_embed(t)  # (B, t_embed_dim)
        c = torch.cat([t_emb, cond], dim=1)  # (B, t_embed + d_cond)
        cond_feat = self.cond_mlp(c)  # (B, cond_hidden)
        
        h = self.proj_in(x)
        for block in self.blocks:
            h = block(h, cond_feat)
        
        return self.proj_out(h)
