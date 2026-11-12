"""Time the models and the simulator."""

import json
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from ..data import Split
from ..normalization import PARAMETER_NAMES
from ..diffusion import DEFAULT_STEPS, Denoiser, sample
from ..paths import DATAGEN_DIR
from ..train import load_checkpoint


@torch.no_grad()
def time_model(model, size: tuple[int, int] = (128, 128), batch_sizes=(1, 64), repeats: int = 20,
               steps: int = DEFAULT_STEPS) -> dict:
    """Seconds per prediction, for each batch size, on the model's device. Returns {batch size: seconds}.

    A regression prediction is one network call. A diffusion prediction is one full sample: ``steps``
    sampler steps, ``2 * steps - 1`` network calls.
    """
    device = next(model.parameters()).device
    out = {}
    for batch in batch_sizes:
        x = torch.rand(batch, 1, *size, device=device)
        cond = torch.rand(batch, len(PARAMETER_NAMES), device=device)
        if isinstance(model, Denoiser):
            noise = torch.randn(batch, 1, *size, device=device)

            def call():
                return sample(model, x, cond, noise, steps)
        else:
            def call():
                return model(x, cond)
        for _ in range(3 if not isinstance(model, Denoiser) else 1):  # warm up
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


def calls_needed(lags, total: int) -> int | None:
    """Fewest model calls whose elapsed times (each one of ``lags``, in snapshot steps) add up to ``total``.

    None if no combination adds up exactly. For example, a model trained only on lag 10 needs 5 calls
    for 50 snapshot steps and cannot do 15.
    """
    best = [0] + [None] * total
    for k in range(1, total + 1):
        options = [best[k - lag] for lag in lags if lag <= k and best[k - lag] is not None]
        best[k] = min(options) + 1 if options else None
    return best[total]


def speed_comparison(checkpoint: Path, split: str = "val", run: int = 0, elapsed=(1_000, 10_000, 25_000, 50_000),
                     steps: int = DEFAULT_STEPS, show: bool = True) -> dict:
    """Compare the time to erode one terrain by each elapsed time: simulator steps against model calls.

    The model needs as many calls as it takes to add up to the elapsed time using the elapsed times it
    was trained on (one call if it was trained on that elapsed time). For a diffusion model, one call is
    one full sample with ``steps`` sampler steps. The simulator runs on one CPU core, as in dataset
    generation. The model is timed on the GPU (if any) at batch size 1 and 64, and on the CPU at batch
    size 1.
    """
    model, _, cfg, _ = load_checkpoint(checkpoint)
    diffusion = isinstance(model, Denoiser)
    data = Split.open(cfg.data_profile, split)
    snapshot_dt = float(data.times[1] - data.times[0])
    lags = cfg.lags if cfg.lags is not None else range(1, len(data.times))
    e = data.erosion(run)
    dt, sim_steps = data.config["dt"], 100
    sim_seconds = time_simulator(data.snapshots[run, 0], e["K"], e["D"], e["m"], e["U"], dt=dt, steps=sim_steps,
                                 repeats=3, cell_size=data.config["cell_size"])
    sim = {"seconds_per_step": sim_seconds / sim_steps, "dt": dt}
    repeats = 3 if diffusion else 20
    gpu = time_model(model, repeats=repeats, steps=steps) if next(model.parameters()).device.type == "cuda" else {}
    cpu_model, _, _, _ = load_checkpoint(checkpoint, device="cpu")
    cpu = time_model(cpu_model, batch_sizes=(1,), repeats=1 if diffusion else 5, steps=steps)
    rows = []
    for dT in elapsed:
        calls = calls_needed(lags, round(dT / snapshot_dt)) if dT % snapshot_dt == 0 else None
        row = {"dT": dT, "calls": calls, "simulator": sim["seconds_per_step"] * dT / sim["dt"],
               "model_cpu": None, "model_gpu": None, "model_gpu_batched": None}
        if calls is not None:
            row["model_cpu"] = calls * cpu[1]
            if gpu:
                row["model_gpu"] = calls * gpu[1]
                row["model_gpu_batched"] = calls * gpu[64] / 64
        rows.append(row)
    result = {"simulator_seconds_per_step": sim["seconds_per_step"], "dt": sim["dt"], "rows": rows}
    if show:
        print(f"simulator: {sim['seconds_per_step'] * 1000:.2f} ms per {sim['dt']:.0f}-yr step, one CPU core")
        if diffusion:
            print(f"diffusion model: one call is one sample, {steps} sampler steps ({2 * steps - 1} network calls)")
        print(f"model: {cpu[1] * 1000:.1f} ms per call on CPU", end="")
        if gpu:
            print(f", {gpu[1] * 1000:.1f} ms on GPU, {gpu[64] / 64 * 1000:.2f} ms per terrain in batches of 64")
        print(f"model trained on elapsed times (yr): {sorted(int(lag * snapshot_dt) for lag in lags)}")
        print(f"\n{'elapsed time':>14} {'simulator':>12} {'model calls':>12} {'model, CPU':>12} {'model, GPU':>12} {'GPU, batched':>13}")

        def fmt(seconds):
            return "-" if seconds is None else f"{seconds * 1000:9.1f} ms"

        for row in rows:
            calls = "not possible" if row["calls"] is None else str(row["calls"])
            print(f"{row['dT']:>11,} yr {fmt(row['simulator']):>12} {calls:>12} {fmt(row['model_cpu']):>12} "
                  f"{fmt(row['model_gpu']):>12} {fmt(row['model_gpu_batched']):>13}")
    return result
