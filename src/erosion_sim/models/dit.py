"""DiT (Diffusion Transformer): a transformer over image patches, with the same interface as ``UNet``.

The image is cut into ``patch`` × ``patch`` patches, each embedded as one token. Every block attends over
all tokens, so a cell can depend on distant terrain in one layer, which a U-Net reaches only through
its downsampling path.

- Positions: 2D RoPE (rotary position embeddings). Each attention head rotates its queries and keys by
  angles proportional to the token's row (first half of the head) and column (second half). Attention
  scores then depend only on the offset between two tokens, not their absolute place, so a window
  behaves the same anywhere, as a world level needs.
- Conditioning: AdaLN (adaptive layer norm). The conditioning embedding sets a shift, scale, and gate for
  each block's attention and MLP (multilayer perceptron). The gates start at zero ("adaLN-Zero", Peebles
  & Xie 2023), so each block starts as the identity.
- The output layer starts at zero, so an untrained model predicts no change, as the U-Net does.
"""

import torch
from torch import nn
from torch.nn import functional as F

from .layers import SinusoidalFeatures, zero_edges


def rope_angles(rows: int, cols: int, head_dim: int, base: float = 100.0, device=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Cosines and sines for 2D RoPE on a ``rows`` × ``cols`` token grid, each (rows * cols, head_dim / 2).

    The first quarter of the frequencies follow the row, the second the column. ``base`` sets the longest
    wavelength, in tokens.
    """
    if head_dim % 4:
        raise ValueError(f"2D RoPE needs a head size divisible by 4, got {head_dim}")
    quarter = head_dim // 4
    freqs = base ** (-torch.arange(quarter, dtype=torch.float32, device=device) / quarter)
    r = torch.arange(rows, dtype=torch.float32, device=device).repeat_interleave(cols)
    c = torch.arange(cols, dtype=torch.float32, device=device).repeat(rows)
    angles = torch.cat([r[:, None] * freqs, c[:, None] * freqs], dim=1)
    return angles.cos(), angles.sin()


def apply_rope(t: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate consecutive pairs of channels of ``t`` (batch, heads, tokens, head_dim) by the angles."""
    even, odd = t[..., 0::2], t[..., 1::2]
    cos, sin = cos.to(t.dtype), sin.to(t.dtype)
    return torch.stack([even * cos - odd * sin, even * sin + odd * cos], dim=-1).flatten(-2)


class Block(nn.Module):
    """Pre-norm transformer block: attention with 2D RoPE, then an MLP, each modulated by AdaLN."""

    def __init__(self, width: int, heads: int, mlp_ratio: int = 4):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(width, mlp_ratio * width), nn.GELU(approximate="tanh"),
                                 nn.Linear(mlp_ratio * width, width))
        self.modulation = nn.Linear(width, 6 * width)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, x: torch.Tensor, emb: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        b, n, w = x.shape
        shift1, scale1, gate1, shift2, scale2, gate2 = self.modulation(emb)[:, None].chunk(6, dim=-1)
        h = self.norm1(x) * (1 + scale1) + shift1
        q, k, v = self.qkv(h).view(b, n, 3, self.heads, w // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(apply_rope(q, cos, sin), apply_rope(k, cos, sin), v)
        x = x + gate1 * self.proj(a.transpose(1, 2).reshape(b, n, w))
        h = self.norm2(x) * (1 + scale2) + shift2
        return x + gate2 * self.mlp(h)


class DiT(nn.Module):
    """Predict a one-channel map from ``x`` and conditioning, like ``UNet``.

    - ``x``: (batch, in_channels, H, W). H and W must be divisible by ``patch``.
    - ``cond``: (batch, cond_dim), each value about in [0, 1].
    - Output: (batch, 1, H, W). With ``fixed_edges``, edge cells are set to zero.
    """

    def __init__(self, cond_dim: int, patch: int = 8, depth: int = 8, width: int = 256, heads: int = 4,
                 frequencies: int = 16, in_channels: int = 1, fixed_edges: bool = True):
        super().__init__()
        if width % heads:
            raise ValueError(f"width {width} must be divisible by heads {heads}")
        self.patch, self.heads, self.fixed_edges = patch, heads, fixed_edges
        self.features = SinusoidalFeatures(frequencies)
        self.embed = nn.Sequential(nn.Linear(cond_dim * 2 * frequencies, width), nn.SiLU(), nn.Linear(width, width),
                                   nn.SiLU())
        self.patchify = nn.Conv2d(in_channels, width, patch, stride=patch)
        self.blocks = nn.ModuleList(Block(width, heads) for _ in range(depth))
        self.out_norm = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.out_modulation = nn.Linear(width, 2 * width)
        self.out = nn.Linear(width, patch * patch)
        for layer in (self.out_modulation, self.out):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        b, _, height, width = x.shape
        p = self.patch
        if height % p or width % p:
            raise ValueError(f"input {height}×{width} is not divisible by patch {p}")
        rows, cols = height // p, width // p
        emb = self.embed(self.features(cond))
        tokens = self.patchify(x).flatten(2).transpose(1, 2)  # (batch, rows * cols, width)
        cos, sin = rope_angles(rows, cols, tokens.shape[-1] // self.heads, device=x.device)
        for block in self.blocks:
            tokens = block(tokens, emb, cos, sin)
        shift, scale = self.out_modulation(emb)[:, None].chunk(2, dim=-1)
        out = self.out(self.out_norm(tokens) * (1 + scale) + shift)  # (batch, tokens, p * p)
        out = out.view(b, rows, cols, p, p).permute(0, 1, 3, 2, 4).reshape(b, 1, height, width)
        return zero_edges(out) if self.fixed_edges else out
