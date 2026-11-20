"""Time the models and the simulator (experiments B13 and C9)."""

import json
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from ..objectives.edm import DEFAULT_STEPS
from ..objectives.sampling import generate, is_generative
from ..paths import DATAGEN_DIR


@torch.no_grad()
def time_model(model, size: tuple[int, int] = (256, 256), batch_sizes=(1, 16), repeats: int = 5,
               steps: int = DEFAULT_STEPS, in_channels: int = 1, cond_dim: int = 4) -> dict:
    """Seconds per prediction, for each batch size, on the model's device. Returns {batch size: seconds}.

    A regression prediction is one network call. A generative prediction is one full sample with
    ``steps`` sampler steps (``sampling.network_calls``).
    """
    device = next(model.parameters()).device
    out = {}
    for batch in batch_sizes:
        x = torch.rand(batch, in_channels, *size, device=device)
        cond = torch.rand(batch, cond_dim, device=device)
        noise = torch.randn(batch, 1, *size, device=device) if is_generative(model) else None

        def call():
            return generate(model, x, cond, noise, steps)
        for _ in range(1 if is_generative(model) else 3):  # warm up
            call()
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            call()
        if device.type == "cuda":
            torch.cuda.synchronize()
        out[batch] = (time.perf_counter() - start) / repeats
    return out


def time_simulator(z0: np.ndarray, K: float, D: float, m: float, U: float, *, dt: float, steps: int,
                   repeats: int = 1, cell_size: float = 1.0) -> float:
    """Seconds for ``steps`` simulator steps on ``z0``, the best of ``repeats``, on one CPU core.

    Runs ``erosion_datagen.timing`` in the datagen environment, as a separate process.
    """
    with tempfile.TemporaryDirectory() as tmp:
        request = Path(tmp) / "request.npz"
        np.savez(request, z0=z0, K=K, D=D, m=m, U=U, dt=dt, steps=steps, repeats=repeats, cell_size=cell_size)
        command = ["uv", "run", "--project", str(DATAGEN_DIR), "python", "-m", "erosion_datagen.timing", str(request)]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
    return json.loads(result.stdout.strip().splitlines()[-1])["seconds"]
