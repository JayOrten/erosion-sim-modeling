"""FNO (Fourier Neural Operator, Li et al. 2021): a backbone that mixes the whole field in Fourier space,
with the same interface as ``UNet``. The optional arm of experiment B3, since the closest prior work on
landscape evolution used it.

Each layer adds a spectral convolution (a learned complex weight on the lowest ``modes`` × ``modes``
Fourier modes, all others dropped) to a 1 × 1 convolution, then FiLM conditioning (the conditioning
embedding scales and shifts the features, as in the U-Net), then GELU. No coordinate channels, so the
model treats every position the same.
"""

import torch
from torch import nn
from torch.nn import functional as F

from .layers import SinusoidalFeatures, zero_edges


class SpectralConv(nn.Module):
    """Multiply the lowest ``modes`` × ``modes`` Fourier modes (both signs of the row frequency) by learned
    complex weights; drop the rest. Fields smaller than ``2 * modes`` keep the modes they have. The weights
    are stored as real and imaginary parts, since fused optimizers take only real parameters."""

    def __init__(self, channels: int, modes: int):
        super().__init__()
        self.modes = modes
        scale = 1 / (channels * channels)
        self.weights = nn.Parameter(scale * torch.randn(2, channels, channels, modes, modes, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        spectrum = torch.fft.rfft2(x.float())
        m_rows, m_cols = min(self.modes, h // 2), min(self.modes, w // 2 + 1)
        weights = torch.view_as_complex(self.weights.float())
        out = torch.zeros_like(spectrum)
        for k, rows in enumerate((slice(0, m_rows), slice(h - m_rows, h))):
            out[:, :, rows, :m_cols] = torch.einsum("bixy,ioxy->boxy", spectrum[:, :, rows, :m_cols],
                                                    weights[k, :, :, :m_rows, :m_cols])
        return torch.fft.irfft2(out, s=(h, w)).to(x.dtype)


class FNO(nn.Module):
    """Predict a one-channel map from ``x`` (batch, in_channels, H, W) and ``cond`` (batch, cond_dim).
    With ``fixed_edges``, edge cells are set to zero. The output layer starts at zero."""

    def __init__(self, cond_dim: int, modes: int = 32, width: int = 64, depth: int = 4, frequencies: int = 16,
                 in_channels: int = 1, fixed_edges: bool = True):
        super().__init__()
        self.fixed_edges = fixed_edges
        self.features = SinusoidalFeatures(frequencies)
        self.embed = nn.Sequential(nn.Linear(cond_dim * 2 * frequencies, width), nn.SiLU(), nn.Linear(width, width),
                                   nn.SiLU())
        self.lift = nn.Conv2d(in_channels, width, 1)
        self.spectral = nn.ModuleList(SpectralConv(width, modes) for _ in range(depth))
        self.pointwise = nn.ModuleList(nn.Conv2d(width, width, 1) for _ in range(depth))
        self.film = nn.ModuleList(nn.Linear(width, 2 * width) for _ in range(depth))
        self.project = nn.Conv2d(width, 2 * width, 1)
        self.out = nn.Conv2d(2 * width, 1, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        emb = self.embed(self.features(cond))
        h = self.lift(x)
        for spectral, pointwise, film in zip(self.spectral, self.pointwise, self.film):
            scale, shift = film(emb)[:, :, None, None].chunk(2, dim=1)
            h = F.gelu((spectral(h) + pointwise(h)) * (1 + scale) + shift)
        out = self.out(F.gelu(self.project(h)))
        return zero_edges(out) if self.fixed_edges else out
