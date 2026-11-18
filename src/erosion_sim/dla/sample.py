"""Grow clusters with a trained DLA model: one call adds ``x`` particles to a start cluster."""

import numpy as np
import torch

from ..diffusion import DEFAULT_STEPS, initial_noise
from ..normalization import dla_conditioning
from ..sampling import generate, is_generative

THRESHOLD = 0.5  # a cell is occupied if the model's output is above this


@torch.no_grad()
def grow(model, start: np.ndarray, p, x, keys: list[tuple], p_range: tuple[float, float], particles: int,
         steps: int = DEFAULT_STEPS, batch_size: int = 64, bf16: bool = True) -> np.ndarray:
    """One sample per row: the cluster after adding ``x[i]`` particles to ``start[i]`` at sticking
    probability ``p[i]``. Returns uint8 occupancy, shape (N, H, W).

    - ``start``: (N, H, W) occupancy, 1 occupied.
    - ``p``, ``x``: length N, or one value for every row.
    - ``keys``: one tuple per row. A generative model (EDM or flow matching) starts from the noise for that
      key, so the same key gives the same sample. Regression ignores it.
    - ``p_range``, ``particles``: the dataset's, for ``dla_conditioning``.
    - ``steps``: sampler steps (``sampling.network_calls`` gives the network calls).
    """
    start = np.asarray(start)
    n = len(start)
    p = np.broadcast_to(np.asarray(p, dtype=np.float64), (n,))
    x = np.broadcast_to(np.asarray(x, dtype=np.float64), (n,))
    generative = is_generative(model)
    if generative and len(keys) != n:
        raise ValueError("a generative model needs one noise key per row")
    device = next(model.parameters()).device
    autocast = torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16 and device.type == "cuda")
    cond = dla_conditioning(p, x, p_range, particles)
    out = np.empty(start.shape, dtype=np.uint8)
    for i in range(0, n, batch_size):
        rows = slice(i, i + batch_size)
        inputs = torch.as_tensor(start[rows], dtype=torch.float32, device=device)[:, None]
        c = torch.as_tensor(cond[rows], dtype=torch.float32, device=device)
        with autocast:
            noise = initial_noise(list(keys[rows]), start.shape[-2:], fixed_edges=False) if generative else None
            y = generate(model, inputs, c, noise, steps)
        out[rows] = (y.float()[:, 0] > THRESHOLD).cpu().numpy()
    return out
