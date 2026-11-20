"""Residual diffusion (experiment B8): a regression model predicts the mean change, and a generative model
samples what it misses, the residual ``y - regression(x)``.

The generative model trains on the residual as its target, with the residual's own RMS as ``sigma_data``
and the regression model's ``Scaling``. ``ResidualModel`` adds the two at prediction time: one regression
call plus the generative model's sampler.
"""

import numpy as np
import torch
from torch import nn


class ResidualModel(nn.Module):
    """``base(x, cond) + inner``'s sample. ``base`` is a regression model; ``inner`` is EDM or flow matching."""

    def __init__(self, base: nn.Module, inner: nn.Module):
        super().__init__()
        self.base, self.inner = base, inner

    @property
    def sigma_data(self) -> float:
        return self.inner.sigma_data

    @property
    def fixed_edges(self) -> bool:
        return self.inner.fixed_edges


@torch.no_grad()
def residual_rms(base: nn.Module, dataset, count: int = 256, seed: int = 0) -> float:
    """RMS of ``y - base(x, cond)`` over ``count`` random items: the residual model's ``sigma_data``."""
    device = next(base.parameters()).device
    rng = np.random.default_rng(seed)
    squares = []
    for i in rng.choice(len(dataset), min(count, len(dataset)), replace=False):
        item = dataset[int(i)]
        x, cond, y = (item[k][None].to(device) for k in ("x", "cond", "y"))
        squares.append(float((y - base(x, cond).float()).pow(2).mean()))
    return float(np.sqrt(np.mean(squares)))
