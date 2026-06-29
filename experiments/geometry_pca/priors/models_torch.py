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


class SeqCrossAttnPool(nn.Module):
    """Pools a (B, S, d_cond) token sequence into a (B, cond_hidden) conditioning
    vector via multihead cross-attention from learned query tokens.

    Unlike mean-pooling, this lets the model attend to specific identity-bearing
    tokens (e.g. 'light brown skin', 'dark eyes') rather than averaging them away.
    """
    def __init__(self, d_cond, cond_hidden, n_heads=8, n_queries=4):
        super().__init__()
        self.n_queries = n_queries
        self.query = nn.Parameter(torch.randn(1, n_queries, cond_hidden) * 0.02)
        self.kv_proj = nn.Linear(d_cond, cond_hidden)
        self.attn = nn.MultiheadAttention(cond_hidden, n_heads, batch_first=True)
        self.out = nn.Linear(cond_hidden * n_queries, cond_hidden)

    def forward(self, cond_seq):
        # cond_seq: (B, S, d_cond)
        B = cond_seq.shape[0]
        kv = self.kv_proj(cond_seq)                 # (B, S, cond_hidden)
        q = self.query.expand(B, -1, -1)            # (B, n_queries, cond_hidden)
        attended, _ = self.attn(q, kv, kv)          # (B, n_queries, cond_hidden)
        return self.out(attended.reshape(B, -1))    # (B, cond_hidden)


class AdaLNResNetCrossAttn(nn.Module):
    """Flow-Matching velocity model conditioned on the FULL T5 token sequence
    via cross-attention pooling (Arm B). Same residual-AdaGN backbone as
    AdaLNResNet, but conditioning comes from attending to all tokens instead of
    a mean-pooled vector.
    """
    def __init__(self, d_in, d_out, d_hidden=1024, n_blocks=12, d_cond=1024,
                 t_embed_dim=64, n_heads=8, n_queries=4):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        cond_hidden = d_hidden

        self.time_embed = SinusoidalEmbedding(t_embed_dim)
        self.seq_pool = SeqCrossAttnPool(d_cond, cond_hidden, n_heads, n_queries)
        # Fuse pooled-sequence conditioning with the timestep embedding
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_hidden + t_embed_dim, cond_hidden),
            nn.SiLU(),
            nn.Linear(cond_hidden, cond_hidden),
        )
        self.proj_in = nn.Linear(d_in, d_hidden)
        self.blocks = nn.ModuleList([
            ResidualBlock(d_hidden, cond_hidden) for _ in range(n_blocks)
        ])
        self.proj_out = nn.Linear(d_hidden, d_out)

    def forward(self, x, t, cond_seq):
        """Args:
            x: (B, d_in) noisy vector
            t: (B, 1) timestep
            cond_seq: (B, S, d_cond) full T5 token sequence
        Returns: (B, d_out) predicted velocity
        """
        t_emb = self.time_embed(t)                  # (B, t_embed_dim)
        seq_feat = self.seq_pool(cond_seq)          # (B, cond_hidden)
        cond_feat = self.cond_mlp(torch.cat([seq_feat, t_emb], dim=1))
        h = self.proj_in(x)
        for block in self.blocks:
            h = block(h, cond_feat)
        return self.proj_out(h)


class IdentityRegressor(nn.Module):
    """Deterministic regressor (Arm C): full-sequence cross-attention → target,
    NO flow matching, NO noise. Tests whether FM stochasticity is hurting a
    near-deterministic text→identity mapping.
    """
    def __init__(self, d_out, d_hidden=1024, n_blocks=6, d_cond=1024,
                 n_heads=8, n_queries=4):
        super().__init__()
        self.d_out = d_out
        cond_hidden = d_hidden
        self.seq_pool = SeqCrossAttnPool(d_cond, cond_hidden, n_heads, n_queries)
        self.trunk = nn.ModuleList([
            ResidualBlock(d_hidden, cond_hidden) for _ in range(n_blocks)
        ])
        # A constant learned token serves as the "input" the trunk refines
        self.x0 = nn.Parameter(torch.zeros(1, d_hidden))
        self.proj_out = nn.Linear(d_hidden, d_out)

    def forward(self, cond_seq):
        """Args: cond_seq (B, S, d_cond). Returns: (B, d_out) predicted target."""
        B = cond_seq.shape[0]
        cond_feat = self.seq_pool(cond_seq)         # (B, cond_hidden)
        h = self.x0.expand(B, -1)
        for block in self.trunk:
            h = block(h, cond_feat)
        return self.proj_out(h)
