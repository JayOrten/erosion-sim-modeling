"""Untrained world levels for tests: real models with random output layers, so outputs are not zero."""

from dataclasses import replace

import torch

from erosion_sim.config import get_config
from erosion_sim.data.normalization import ChangeScaling, ParameterScaling, Scaling
from erosion_sim.training.checkpoints import build_model

SCALING = Scaling(ParameterScaling(low=(0.0,) * 4, high=(1.0,) * 4), ChangeScaling(coef=1.0, power=0.0))


def level(cell: int, window: int, context_cell: int | None = None, objective: str = "edm", seed: int = 0, **changes):
    """``(model, scaling, config)`` of an untrained level, as ``load_checkpoint`` gives them."""
    cfg = replace(get_config("smoke", objective), fixed_edges=False, cell=cell, window=window,
                  context_cell=context_cell, **changes)
    torch.manual_seed(seed)
    model = build_model(cfg).eval()
    net = model.net if hasattr(model, "net") else model
    torch.nn.init.normal_(net.out.weight, std=0.1)
    return model, SCALING, cfg


# Two and three levels: cells 2 and 1; cells 4, 2, and 1.
TWO = [level(2, 8, seed=0), level(1, 16, context_cell=2, seed=1)]
THREE = [level(4, 8, seed=0), level(2, 8, context_cell=4, seed=1), level(1, 16, context_cell=2, seed=2)]
