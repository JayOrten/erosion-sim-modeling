"""Error as model calls are chained over time."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..data import Split
from ..predict import batched_rollout, default_lag, predict_batch
from ..train import load_checkpoint


def rollout_error(checkpoint: Path, split: str = "val", runs: int = 64, lag: int | None = None, seed: int = 0,
                  show: bool = True, data_root: Path | None = None) -> dict:
    """Error against elapsed time from t = 0, as model calls are chained.

    For ``runs`` random matching runs, starting from the simulator's terrain at t = 0, compares:

    - ``rollout``: the model's own previous output fed back in, ``lag`` snapshots per call.
    - ``one call``: one model call from the simulator's terrain one step earlier. The gap between this
      and ``rollout`` is the error added by chaining.
    - ``direct``: one model call from t = 0 straight to each time. Only if the model was trained on
      every one of those elapsed times.
    - ``no change``: the terrain at t = 0.

    A diffusion model draws one sample per call. Two calls of ``rollout_error`` with the same ``seed`` and
    the same run filter use the same runs, so their results can be compared directly.

    Returns RMSE (m) and skill for each, per elapsed time.
    """
    model, scaling, cfg, _ = load_checkpoint(checkpoint)
    lag = default_lag(cfg) if lag is None else lag
    data = Split.open(cfg.data_profile, split, data_root)
    params, snapshots, times = data.params, data.snapshots, data.times
    matching = data.select(**dict(cfg.runs))
    chosen = np.random.default_rng(seed).choice(matching, min(runs, len(matching)), replace=False)
    picks = np.arange(0, len(times), lag)
    dT = float(times[lag] - times[0])
    trained = set(cfg.lags) if cfg.lags is not None else set(range(1, len(times)))
    names = ["no change", "one call", "rollout"] + (["direct"] if set(picks[1:]) <= trained else [])
    chosen, steps = np.sort(chosen), len(picks) - 1
    sim = np.asarray(snapshots[chosen][:, picks], dtype=np.float64)  # (runs, steps + 1, H, W)

    def sq_error(z):  # summed over runs, per frame
        return ((z - sim[:, 1:]) ** 2).mean(axis=(2, 3)).sum(0)

    sq = {"no change": sq_error(sim[:, :1])}
    # One call per frame, from the simulator's previous snapshot. All frames of all runs go in together.
    keys = [(seed, "one call", int(run), k, 0) for run in chosen for k in range(1, steps + 1)]
    sq["one call"] = sq_error(predict_batch(model, scaling, sim[:, :-1].reshape(-1, *sim.shape[2:]), params,
                                            np.repeat(chosen, steps), dT, keys).reshape(sim[:, 1:].shape))
    sq["rollout"] = sq_error(batched_rollout(model, scaling, sim[:, 0], params, chosen, dT, steps, seed=seed)[:, 1:])
    if "direct" in names:
        sq["direct"] = sq_error(batched_rollout(model, scaling, sim[:, 0], params, chosen, dT, steps, "direct",
                                                seed=seed)[:, 1:])
    elapsed = times[picks[1:]] - times[0]
    result = {"elapsed": elapsed}
    for name, total in sq.items():
        result[name] = {"rmse": np.sqrt(total / len(chosen)), "skill": 1 - total / sq["no change"]}
    if show:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
        for i, name in enumerate(sq):
            ax1.plot(elapsed, result[name]["rmse"], "o-", color=f"C{i}", label=name)
            if name != "no change":
                ax2.plot(elapsed, result[name]["skill"], "o-", color=f"C{i}", label=name)
        ax1.set_ylabel("RMSE (m)")
        ax2.set_ylabel("skill against no change from t = 0")
        for ax in (ax1, ax2):
            ax.set_xlabel("elapsed time from t = 0 (yr)")
            ax.legend()
        fig.suptitle(f"{Path(checkpoint).stem}: {len(chosen)} {split} runs, {lag}-snapshot calls ({dT:,.0f} yr)")
        plt.show()
    return result
