"""Diffusion model for the normalized change, using the EDM formulation.

EDM is the noise schedule, preconditioning, and loss weighting from Karras et al. 2022, "Elucidating
the Design Space of Diffusion Models". A noisy target is ``y + sigma * noise``. The denoiser estimates
the clean ``y`` from it, given the noise level ``sigma``, the start heightmap, and the conditioning.
"""

import math

import torch
from torch import nn

from .model import UNet

# Noise range of the sampler, as multiples of sigma_data. EDM's defaults are 0.002 and 80 for images
# with sigma_data = 0.5.
SIGMA_MIN = 0.004
SIGMA_MAX = 160.0


class Denoiser(nn.Module):
    """Estimate the clean normalized change from a noisy one, with EDM preconditioning.

    The U-Net sees two channels, the start heightmap ``x`` and the scaled noisy target, and one extra
    conditioning scalar for the noise level. Its output is mixed with the noisy input:

        D(noisy, sigma) = c_skip * noisy + c_out * F(c_in * noisy, c_noise)

    The constants keep the U-Net's input and target near unit variance at every noise level. At small
    ``sigma`` the result is mostly ``noisy``; at large ``sigma`` it is mostly the U-Net's output.

    - ``x``: (batch, 1, H, W) start heightmap, as in ``ErosionPairs``.
    - ``cond``: (batch, cond_dim) normalized conditioning scalars.
    - ``noisy``: (batch, 1, H, W) noisy normalized change.
    - ``sigma``: (batch,) noise level of each sample.
    - Output: (batch, 1, H, W). Edge nodes are zero, since the simulator holds edges fixed.
    """

    def __init__(self, cond_dim: int, sigma_data: float = 1.0, **unet_kwargs):
        super().__init__()
        self.sigma_data = float(sigma_data)
        self.net = UNet(cond_dim + 1, in_channels=2, **unet_kwargs)

    def noise_feature(self, sigma: torch.Tensor) -> torch.Tensor:
        """EDM's ``c_noise = ln(sigma) / 4``, mapped so the sampler's range becomes [0, 1]."""
        low, high = math.log(SIGMA_MIN), math.log(SIGMA_MAX)
        return (torch.log(sigma / self.sigma_data) - low) / (high - low)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, noisy: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        interior = torch.zeros_like(noisy)
        interior[..., 1:-1, 1:-1] = 1
        noisy = noisy * interior
        s = sigma.reshape(-1, 1, 1, 1).to(noisy.dtype)
        sd = self.sigma_data
        c_skip = sd**2 / (s**2 + sd**2)
        c_out = s * sd / (s**2 + sd**2).sqrt()
        c_in = 1 / (s**2 + sd**2).sqrt()
        net_cond = torch.cat([cond, self.noise_feature(sigma.to(cond.dtype))[:, None]], dim=1)
        out = self.net(torch.cat([x, c_in * noisy], dim=1), net_cond)
        return (c_skip * noisy + c_out * out) * interior
