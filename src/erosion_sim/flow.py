"""Flow matching (rectified flow): a straight path from the target to noise, and a network that learns its
velocity.

The target ``y`` is divided by ``sigma_data`` (its standard deviation) so it is about unit size, like the
noise: ``z = y / sigma_data``. A point on the path at time ``s`` in [0, 1] is

    z_s = (1 - s) * z + s * noise

so ``s = 0`` is the target and ``s = 1`` is pure noise. Its velocity, ``noise - z``, is the same at every
``s``. The network predicts it from ``z_s`` and ``s``; the loss is the mean squared error.

- Training times: ``s = sigmoid(n)`` with ``n`` standard normal (logit-normal, Esser et al. 2024), which
  puts more training near the middle of the path.
- Sampling: Euler steps from ``s = 1`` to ``s = 0``, one network call per step.

``FlowModel`` has the same inputs as ``diffusion.Denoiser``, so either can wrap a U-Net or a DiT.
"""

import math
from statistics import NormalDist

import torch
from torch import nn

from .model import build_backbone, zero_edges

# Validation times: the 8 octiles of the logit-normal training distribution, so every validation pass
# uses the same times.
VAL_TIMES = tuple(1 / (1 + math.exp(-NormalDist().inv_cdf((i + 0.5) / 8))) for i in range(8))


class FlowModel(nn.Module):
    """Predict the velocity ``noise - z`` at path point ``z_s`` and time ``s``.

    - ``x``: (batch, in_channels, H, W), as in ``Denoiser``.
    - ``cond``: (batch, cond_dim) normalized conditioning.
    - ``z_s``: (batch, 1, H, W) path point, in units of ``sigma_data``.
    - ``s``: (batch,) time in [0, 1], given to the network as one more conditioning value.
    - Output: (batch, 1, H, W) velocity, zero on the edges with ``fixed_edges``.
    """

    def __init__(self, cond_dim: int, sigma_data: float = 1.0, fixed_edges: bool = True, in_channels: int = 1,
                 backbone: str = "unet", **net_kwargs):
        super().__init__()
        self.sigma_data = float(sigma_data)
        self.fixed_edges = fixed_edges
        self.net = build_backbone(backbone, cond_dim + 1, in_channels + 1, fixed_edges, **net_kwargs)

    def edges(self, t: torch.Tensor) -> torch.Tensor:
        return zero_edges(t) if self.fixed_edges else t

    def forward(self, x: torch.Tensor, cond: torch.Tensor, z_s: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        z_s = self.edges(z_s)
        net_cond = torch.cat([cond, s.to(cond.dtype)[:, None]], dim=1)
        return self.edges(self.net(torch.cat([x, z_s], dim=1), net_cond))


def flow_loss(model: FlowModel, x, cond, y, s: torch.Tensor, noise: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-sample velocity MSE at times ``s`` (batch,), and the estimate of ``y`` it implies,
    ``(z_s - s * velocity) * sigma_data``."""
    sd = model.sigma_data
    z, t = y / sd, s.reshape(-1, 1, 1, 1)
    z_s = (1 - t) * z + t * model.edges(noise)
    velocity = model(x, cond, z_s, s).float()
    mse = (velocity - (model.edges(noise) - z)).pow(2).mean(dim=(1, 2, 3))
    return mse, (z_s - t * velocity) * sd


def training_loss(model: FlowModel, x, cond, y, generator: torch.Generator) -> torch.Tensor:
    """Flow matching loss for one batch: a logit-normal time and fresh noise per sample."""
    s = torch.sigmoid(torch.randn(len(y), device=y.device, generator=generator))
    noise = torch.randn(y.shape, device=y.device, generator=generator)
    return flow_loss(model, x, cond, y, s, noise)[0].mean()


@torch.no_grad()
def sample(model: FlowModel, x: torch.Tensor, cond: torch.Tensor, noise: torch.Tensor, steps: int) -> torch.Tensor:
    """Turn unit noise into a target: ``steps`` Euler steps from ``s = 1`` to 0, one network call each.
    The same noise always gives the same sample. Returns (batch, 1, H, W) in the target's units."""
    z = model.edges(noise.to(device=x.device, dtype=torch.float32))
    times = torch.linspace(1.0, 0.0, steps + 1).tolist()
    ones = torch.ones(len(z), device=x.device)
    for s, s_next in zip(times[:-1], times[1:]):
        z = z + (s_next - s) * model(x, cond, z, ones * s).float()
    return z * model.sigma_data
