"""Diffusion model for the normalized change, using the EDM formulation.

EDM is the noise schedule, preconditioning, and loss weighting from Karras et al. 2022, "Elucidating
the Design Space of Diffusion Models". A noisy target is ``y + sigma * noise``. The denoiser estimates
the clean ``y`` from it, given the noise level ``sigma``, the start heightmap, and the conditioning.
"""

import hashlib
import json
import math
from statistics import NormalDist

import torch
from torch import nn

from .model import UNet

# Noise levels below are multiples of sigma_data. EDM's defaults assume images with sigma_data = 0.5,
# so each is rescaled from that.
# Sampler range. EDM: 0.002 to 80.
SIGMA_MIN = 0.004
SIGMA_MAX = 160.0
# Training draws ln(sigma) from a normal distribution. EDM: mean -1.2, standard deviation 1.2.
P_MEAN = -1.2 + math.log(2)
P_STD = 1.2
# Validation uses fixed noise levels: the 1/16, 3/16, ..., 15/16 quantiles of the training distribution,
# so their average loss tracks the training loss.
VAL_SIGMAS = tuple(math.exp(NormalDist(P_MEAN, P_STD).inv_cdf((i + 0.5) / 8)) for i in range(8))
# Sampler time steps: EDM's spacing exponent, and a default step count.
RHO = 7.0
DEFAULT_STEPS = 32


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


def loss_weight(sigma: torch.Tensor, sigma_data: float) -> torch.Tensor:
    """EDM's weight ``(sigma^2 + sigma_data^2) / (sigma * sigma_data)^2``. It makes the weighted loss
    equal the U-Net's own error against its effective target, which has unit variance at every ``sigma``."""
    return (sigma**2 + sigma_data**2) / (sigma * sigma_data) ** 2


def denoising_loss(denoiser: Denoiser, x, cond, y, sigma: torch.Tensor, noise: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-sample weighted MSE of denoising ``y + sigma * noise``, and the denoised estimate.

    ``sigma`` is (batch,) and in the same units as ``y``.
    """
    denoised = denoiser(x, cond, y + sigma.reshape(-1, 1, 1, 1) * noise, sigma).float()
    mse = (denoised - y).pow(2).mean(dim=(1, 2, 3))
    return loss_weight(sigma, denoiser.sigma_data) * mse, denoised


def training_loss(denoiser: Denoiser, x, cond, y, generator: torch.Generator) -> torch.Tensor:
    """EDM training loss for one batch: a random noise level and random noise per sample."""
    sd = denoiser.sigma_data
    sigma = sd * torch.exp(P_MEAN + P_STD * torch.randn(len(y), device=y.device, generator=generator))
    noise = torch.randn(y.shape, device=y.device, generator=generator)
    return denoising_loss(denoiser, x, cond, y, sigma, noise)[0].mean()


def step_generator(seed: int, step: int, device) -> torch.Generator:
    """Random stream for one training step. It depends only on the seed and the step, so a resumed run
    draws the same noise as an uninterrupted one."""
    return torch.Generator(device=device).manual_seed(seed * 1_000_003 + step)


def noise_seed(*key) -> int:
    """A 64-bit seed from a public key such as ``(seed, pair, sample)``, by hashing its canonical JSON."""
    payload = json.dumps(["erosion-sim-diffusion-v1", *key], separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def initial_noise(keys: list[tuple], shape: tuple[int, int]) -> torch.Tensor:
    """Unit Gaussian noise of ``shape`` for each key, shape (len(keys), 1, H, W), with zero edges.

    Drawn on the CPU, so a key gives the same noise on any device.
    """
    noise = torch.stack([torch.randn(1, *shape, generator=torch.Generator().manual_seed(noise_seed(*k)))
                         for k in keys])
    noise[..., [0, -1], :] = 0
    noise[..., :, [0, -1]] = 0
    return noise


def sigma_steps(steps: int, sigma_data: float) -> torch.Tensor:
    """EDM's noise levels from ``SIGMA_MAX`` down to ``SIGMA_MIN`` (times ``sigma_data``), then 0.

    Spaced evenly in ``sigma ** (1 / RHO)``, which puts more steps at low noise.
    """
    hi, lo = (SIGMA_MAX * sigma_data) ** (1 / RHO), (SIGMA_MIN * sigma_data) ** (1 / RHO)
    sigmas = (hi + torch.linspace(0, 1, steps, dtype=torch.float64) * (lo - hi)) ** RHO
    return torch.cat([sigmas, torch.zeros(1, dtype=torch.float64)])


@torch.no_grad()
def sample(denoiser: Denoiser, x: torch.Tensor, cond: torch.Tensor, noise: torch.Tensor,
           steps: int = DEFAULT_STEPS) -> torch.Tensor:
    """Turn unit noise into a normalized change with EDM's deterministic Heun sampler.

    The sampler solves the probability-flow ODE (ordinary differential equation) from high noise to
    zero. Each step takes an Euler step, then corrects it with the slope at the new point (Heun's
    method), except the last. That is ``2 * steps - 1`` denoiser calls. The result depends only on
    ``noise``, so the same noise always gives the same sample.

    - ``x``, ``cond``: as in ``Denoiser``.
    - ``noise``: (batch, 1, H, W) unit noise, for example from ``initial_noise``.
    - Output: (batch, 1, H, W) normalized change.
    """
    sigmas = sigma_steps(steps, denoiser.sigma_data).tolist()
    noise = noise.to(device=x.device, dtype=torch.float32)
    y = noise * sigmas[0]
    ones = torch.ones(len(y), device=x.device)

    def slope(y, s):
        return (y - denoiser(x, cond, y, ones * s).float()) / s

    for s, s_next in zip(sigmas[:-1], sigmas[1:]):
        d = slope(y, s)
        y_next = y + (s_next - s) * d
        if s_next > 0:
            y_next = y + (s_next - s) * 0.5 * (d + slope(y_next, s_next))
        y = y_next
    return y
