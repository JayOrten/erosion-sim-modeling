"""Pieces shared by the backbones."""

import math

import torch
from torch import nn


def zero_edges(t: torch.Tensor) -> torch.Tensor:
    """``t`` (..., H, W) with its outer ring of cells set to zero.

    The simulator holds its edge cells at a fixed height, so their change is always zero. Every model
    output and the diffusion noise go through this one function, unless the model was built with
    ``fixed_edges=False`` (world models, whose window edges are not boundaries).
    """
    interior = torch.zeros_like(t)
    interior[..., 1:-1, 1:-1] = 1
    return t * interior


class SinusoidalFeatures(nn.Module):
    """Map each scalar in [0, 1] to sines and cosines at ``frequencies`` geometrically spaced frequencies.

    Small changes in a scalar then change many features, which a small MLP can use more easily than
    the raw value.
    """

    def __init__(self, frequencies: int = 16, max_frequency: float = 100.0):
        super().__init__()
        freqs = torch.exp(torch.linspace(0.0, math.log(max_frequency), frequencies))
        self.register_buffer("freqs", freqs * math.pi)

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        angles = cond[..., None] * self.freqs  # (batch, scalars, frequencies)
        return torch.cat([angles.sin(), angles.cos()], dim=-1).flatten(1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
