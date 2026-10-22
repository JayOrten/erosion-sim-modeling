"""Time the simulator on one run of a generated dataset, on one core. Prints JSON.

Run from datagen/ with: uv run python scripts/time_run.py <profile> <split> <run> [steps] [repeats]
The main project's evaluate.py calls this to compare the simulator with the model.
"""

import json
import sys
import time

from erosion_datagen.config import PROFILES
from erosion_datagen.simulate import dataset_dir, load_dataset, simulate

if __name__ == "__main__":
    profile, split, run = sys.argv[1], sys.argv[2], int(sys.argv[3])
    steps = int(sys.argv[4]) if len(sys.argv) > 4 else 100
    repeats = int(sys.argv[5]) if len(sys.argv) > 5 else 3
    cfg = PROFILES[profile]
    snapshots, params, _ = load_dataset(dataset_dir(cfg, split))
    z0 = snapshots[run, 0]
    args = dict(K=params["K"][run], D=params["D"][run], m=params["m"][run], U=params["U"][run])
    simulate(z0, **args, dt=cfg.dt, steps=1, snapshot_every=1)  # warm up imports and caches
    seconds = []
    for _ in range(repeats):
        start = time.perf_counter()
        simulate(z0, **args, dt=cfg.dt, steps=steps, snapshot_every=steps)
        seconds.append(time.perf_counter() - start)
    print(json.dumps({"seconds_per_step": min(seconds) / steps, "dt": cfg.dt, "steps": steps, "repeats": repeats}))
