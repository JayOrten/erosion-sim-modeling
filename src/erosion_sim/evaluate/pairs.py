"""The ``pairs`` dict that every metric takes, and predicting it with chunk models.

A ``pairs`` dict holds, per pair, the simulator's start and end heightmaps (``z_a``, ``z_b``), one end
heightmap per method, and the pair's run, times, and parameters (``FACTORS``). ``methods`` lists the
methods in order, starting with ``"no change"`` (the start terrain), which skill is defined against;
``kind`` maps each method to ``"baseline"``, ``"regression"``, ``"sample"`` or ``"mean"``. Errors are in
meters.

- ``make_pairs``: a ``pairs`` dict with only the simulator and "no change"; ``add_method`` adds a method.
- ``subset_pairs``: the pairs selected by a mask.
- ``predict_pairs``: chunk models on a random sample of pairs.
"""

from pathlib import Path

import numpy as np
from tqdm import tqdm

from ..data import ErosionPairs, Split
from ..diffusion import DEFAULT_STEPS, Denoiser
from ..predict import predict_batch
from ..train import load_checkpoint

FACTORS = ("K_ref", "D_over_K", "m", "U_over_K", "start_time", "dT")


def make_pairs(data: Split, run, start, end, z_a: np.ndarray, z_b: np.ndarray) -> dict:
    """A ``pairs`` dict for pairs (``run[i]``, snapshot ``start[i]``, snapshot ``end[i]``) of ``data``, with
    heightmaps ``z_a`` and ``z_b``. Its only method is "no change"."""
    run, start, end = (np.broadcast_to(np.asarray(v, dtype=np.int64), (len(z_a),)) for v in (run, start, end))
    params, times = data.params, data.times
    return {
        "methods": ["no change"], "kind": {"no change": "baseline"}, "spread": {},
        "z_a": z_a, "z_b": z_b, "no change": z_a.copy(),
        "run": run, "start": start, "end": end, "start_time": times[start], "dT": times[end] - times[start],
        "K_ref": params["K_ref"][run], "m": params["m"][run],
        "D_over_K": (params["D"] / params["K_ref"])[run], "U_over_K": (params["U"] / params["K_ref"])[run],
    }


def add_method(pairs: dict, name: str, z: np.ndarray, kind: str) -> None:
    """Add method ``name`` with end heightmaps ``z`` (one per pair)."""
    pairs[name] = z
    pairs["methods"].append(name)
    pairs["kind"][name] = kind


def subset_pairs(pairs: dict, mask: np.ndarray) -> dict:
    """The pairs where ``mask`` is true (a boolean mask or index array), with every method."""
    n = len(pairs["z_a"])
    return {k: (v[mask] if isinstance(v, np.ndarray) and len(v) == n else v) for k, v in pairs.items()}


def predict_pairs(checkpoints: Path | dict[str, Path], split: str = "val", count: int = 512, seed: int = 0,
                  batch_size: int = 64, samples: int = 8, steps: int = DEFAULT_STEPS, bf16: bool = True,
                  progress: bool = True, data_root: Path | None = None) -> dict:
    """Predict ``count`` random pairs of ``split`` with each model.

    ``bf16`` runs the network in 16-bit autocast on the GPU (ignored on CPU), as in training.
    ``progress`` shows a progress bar per model.

    ``checkpoints`` is one path, whose method is named ``"model"``, or a dict ``{name: path}``. All of
    them must share the data profile, run filter, and elapsed times, so every model sees the same pairs:
    runs matching the run filter, at the training elapsed times.

    - A regression checkpoint adds one method, ``name``.
    - A diffusion checkpoint draws ``samples`` samples per pair with ``steps`` sampler steps. It adds
      ``"<name> (1 sample)"`` (sample 0) and ``"<name> (mean of N)"``. The per-pixel standard deviation
      across its samples goes in ``pairs["spread"][name]``. Sample ``k`` of pair ``i`` (an index into the
      dataset) uses noise key ``(seed, i, k)``.

    Returns a ``pairs`` dict.
    """
    if not isinstance(checkpoints, dict):
        checkpoints = {"model": checkpoints}
    loaded = {name: load_checkpoint(path) for name, path in checkpoints.items()}
    if len({(c.data_profile, c.runs, c.lags) for _, _, c, _ in loaded.values()}) > 1:
        raise ValueError("checkpoints must share data_profile, runs and lags to be compared on the same pairs")
    _, scaling, cfg, _ = next(iter(loaded.values()))
    split_data = Split.open(cfg.data_profile, split, data_root)
    params = split_data.params
    data = ErosionPairs(split_data, scaling, runs=split_data.select(**dict(cfg.runs)), lags=cfg.lags)
    picks = np.sort(np.random.default_rng(seed).choice(len(data), min(count, len(data)), replace=False))
    run, start, end = data.locate(picks)
    z_a = np.asarray(data.snapshots[run, start], dtype=np.float64)
    z_b = np.asarray(data.snapshots[run, end], dtype=np.float64)
    out = make_pairs(split_data, run, start, end, z_a, z_b)
    dT = out["dT"]
    for name, (model, scaling, _, _) in loaded.items():
        if not isinstance(model, Denoiser):
            add_method(out, name, predict_batch(model, scaling, z_a, params, run, dT, batch_size=batch_size,
                                                bf16=bf16, progress=name if progress else None), "regression")
            continue
        # Samples go through the model together, a few pairs at a time, so memory stays small.
        single, mean, spread = [], [], []
        chunk = max(1, batch_size // samples)
        bar = tqdm(total=len(picks), desc=name, unit="pair", disable=not progress)
        for i in range(0, len(picks), chunk):
            rows = slice(i, i + chunk)
            n = len(picks[rows])
            keys = [(seed, int(p), k) for p in picks[rows] for k in range(samples)]
            z = predict_batch(model, scaling, np.repeat(z_a[rows], samples, 0), params,
                              np.repeat(run[rows], samples), np.repeat(dT[rows], samples), keys, steps,
                              batch_size, bf16).reshape(n, samples, *z_a.shape[1:])
            single.append(z[:, 0])
            mean.append(z.mean(1))
            spread.append(z.std(1))
            bar.update(n)
        bar.close()
        add_method(out, f"{name} (1 sample)", np.concatenate(single), "sample")
        add_method(out, f"{name} (mean of {samples})", np.concatenate(mean), "mean")
        out["spread"][name] = np.concatenate(spread)
    return out


def model_methods(pairs: dict) -> list[str]:
    """Every method except the no-change baseline."""
    return pairs["methods"][1:]


def per_pair_skill(pairs: dict, method: str) -> np.ndarray:
    """``1 - MSE / MSE(no change)`` of each pair."""
    mse = ((pairs[method] - pairs["z_b"]) ** 2).mean(axis=(1, 2))
    mse_zero = ((pairs["z_a"] - pairs["z_b"]) ** 2).mean(axis=(1, 2))
    return 1 - mse / mse_zero
