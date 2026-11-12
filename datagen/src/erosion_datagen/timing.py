"""Time the simulator, to compare it with the models in the main package.

The main package cannot import this environment, so it runs this module as a separate process
(``erosion_sim.evaluate.speed.time_simulator``):

    uv run --project datagen python -m erosion_datagen.timing <request.npz>

The request file holds ``z0`` and the arguments of ``time_simulator``. The result is printed as JSON.
The process uses one thread, as in dataset generation (``simulate`` sets ``OMP_NUM_THREADS``).
"""

import json
import sys
import time

import numpy as np

from .simulate import simulate


def time_simulator(
    z0: np.ndarray,
    K: float,
    D: float,
    m: float,
    U: float,
    *,
    dt: float,
    steps: int,
    repeats: int = 1,
    cell_size: float = 1.0,
) -> float:
    """Seconds for ``steps`` simulator steps on ``z0``, the best of ``repeats``, after a one-step warm-up."""
    args = dict(K=K, D=D, m=m, U=U, dt=dt, cell_size=cell_size)
    simulate(z0, **args, steps=1, snapshot_every=1)  # warm up imports and caches
    seconds = []
    for _ in range(repeats):
        start = time.perf_counter()
        simulate(z0, **args, steps=steps, snapshot_every=steps)
        seconds.append(time.perf_counter() - start)
    return min(seconds)


if __name__ == "__main__":
    with np.load(sys.argv[1]) as f:
        request = {k: f[k].item() if f[k].ndim == 0 else f[k] for k in f.files}
    z0 = request.pop("z0")
    print(json.dumps({"seconds": time_simulator(z0, **request)}))
