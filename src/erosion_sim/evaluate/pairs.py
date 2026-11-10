"""Predict a fixed sample of validation pairs with one or more models, for the metrics.

``predict_pairs`` runs the models on a sample of pairs; the functions in ``metrics`` take its result.
"No change" (the start terrain) is kept as a reference row; skill is defined against it. Errors are in
meters, not in the normalized training targets.
"""

from pathlib import Path

import numpy as np
from tqdm import tqdm

from ..data import ErosionPairs, load_params, select_runs
from ..diffusion import DEFAULT_STEPS, Denoiser
from ..paths import data_root as resolve_root
from ..predict import predict_batch
from ..train import load_checkpoint

FACTORS = ("K_ref", "D_over_K", "m", "U_over_K", "start_time", "dT")


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

    Returns a dict of arrays: ``z_a``, ``z_b`` (simulator), one end heightmap per method, and per pair
    ``run``, ``start_time``, ``dT`` and the parameters in ``FACTORS``. ``pairs["methods"]`` lists the
    methods in order, starting with ``"no change"``. ``pairs["kind"]`` maps each method to
    ``"baseline"``, ``"regression"``, ``"sample"`` or ``"mean"``.
    """
    if not isinstance(checkpoints, dict):
        checkpoints = {"model": checkpoints}
    loaded = {name: load_checkpoint(path) for name, path in checkpoints.items()}
    if len({(c.data_profile, c.runs, c.lags) for _, _, c, _ in loaded.values()}) > 1:
        raise ValueError("checkpoints must share data_profile, runs and lags to be compared on the same pairs")
    _, normalizer, cfg, _ = next(iter(loaded.values()))
    root = resolve_root(cfg.data_profile, data_root)
    params, _ = load_params(root / split)
    data = ErosionPairs(root / split, normalizer, runs=select_runs(params, **dict(cfg.runs)), lags=cfg.lags)
    picks = np.sort(np.random.default_rng(seed).choice(len(data), min(count, len(data)), replace=False))
    run, start, end = data.locate(picks)
    z_a = np.asarray(data.snapshots[run, start], dtype=np.float64)
    z_b = np.asarray(data.snapshots[run, end], dtype=np.float64)
    dT = data.times[end] - data.times[start]

    out = {"methods": ["no change"], "kind": {"no change": "baseline"}, "spread": {}}
    for name, (model, normalizer, _, _) in loaded.items():
        if not isinstance(model, Denoiser):
            out[name] = predict_batch(model, normalizer, z_a, params, run, dT, batch_size=batch_size, bf16=bf16,
                                      progress=name if progress else None)
            out["methods"].append(name)
            out["kind"][name] = "regression"
            continue
        # Samples go through the model together, a few pairs at a time, so memory stays small.
        single, mean, spread = [], [], []
        chunk = max(1, batch_size // samples)
        bar = tqdm(total=len(picks), desc=name, unit="pair", disable=not progress)
        for i in range(0, len(picks), chunk):
            rows = slice(i, i + chunk)
            n = len(picks[rows])
            keys = [(seed, int(p), k) for p in picks[rows] for k in range(samples)]
            z = predict_batch(model, normalizer, np.repeat(z_a[rows], samples, 0), params,
                              np.repeat(run[rows], samples), np.repeat(dT[rows], samples), keys, steps,
                              batch_size, bf16).reshape(n, samples, *z_a.shape[1:])
            single.append(z[:, 0])
            mean.append(z.mean(1))
            spread.append(z.std(1))
            bar.update(n)
        bar.close()
        one, many = f"{name} (1 sample)", f"{name} (mean of {samples})"
        out[one], out[many] = np.concatenate(single), np.concatenate(mean)
        out["spread"][name] = np.concatenate(spread)
        out["methods"] += [one, many]
        out["kind"].update({one: "sample", many: "mean"})

    out.update(z_a=z_a, z_b=z_b, run=run, start=start, end=end, start_time=data.times[start], dT=dT)
    out["no change"] = z_a.copy()
    out["K_ref"], out["m"] = params["K_ref"][run], params["m"][run]
    out["D_over_K"], out["U_over_K"] = (params["D"] / params["K_ref"])[run], (params["U"] / params["K_ref"])[run]
    return out


def model_methods(pairs: dict) -> list[str]:
    """Every method except the no-change baseline."""
    return pairs["methods"][1:]


def per_pair_skill(pairs: dict, method: str) -> np.ndarray:
    """``1 - MSE / MSE(no change)`` of each pair."""
    mse = ((pairs[method] - pairs["z_b"]) ** 2).mean(axis=(1, 2))
    mse_zero = ((pairs["z_a"] - pairs["z_b"]) ** 2).mean(axis=(1, 2))
    return 1 - mse / mse_zero
