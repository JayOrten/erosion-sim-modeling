"""Evaluate a trained model against the simulator: predictions, accuracy, and speed."""

import json
import subprocess
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from .data import ARTIFACTS_DIR, COND_NAMES, ErosionPairs, denormalize, load_params, select_runs
from .generate import default_lag, predict, rollout
from .hydrology import drainage_area, pits
from .render import hillshade
from .train import load_checkpoint

DATAGEN_DIR = Path(__file__).resolve().parents[2] / "datagen"


@torch.no_grad()
def show_predictions(path: Path, count: int = 3, split: str = "val", data_root: Path | None = None, seed: int = 0) -> None:
    """Plot the model's prediction next to the simulator for ``count`` random pairs of ``split``.

    Columns: start, simulator end, model end, simulator change, model change, error. Changes and error
    share one color scale per row.
    """
    model, normalizer, cfg, _ = load_checkpoint(path)
    device = next(model.parameters()).device
    root = Path(data_root) if data_root is not None else ARTIFACTS_DIR / "data" / cfg.data_profile
    params = dict(np.load(root / split / "params.npz"))
    data = ErosionPairs(root / split, normalizer, runs=select_runs(params, **dict(cfg.runs)), lags=cfg.lags)
    fig, axes = plt.subplots(count, 6, figsize=(19, 3.3 * count), layout="constrained", squeeze=False)
    for row, i in zip(axes, np.random.default_rng(seed).choice(len(data), count, replace=False)):
        item = data[int(i)]
        pred = model(item["x"][None].to(device), item["cond"][None].to(device)).float().cpu()
        z_a, z_b = item["z_a"].numpy(), item["z_b"].numpy()
        z_pred = denormalize(pred, item["z_a"][None], item["relief"], item["scale"])[0].numpy()
        low, high = min(z_a.min(), z_b.min()), max(z_a.max(), z_b.max())
        true_change, pred_change = z_b - z_a, z_pred - z_a
        limit = np.abs(true_change).max()
        dT = data.times[item["end"]] - data.times[item["start"]]
        rmse = np.sqrt(((z_pred - z_b) ** 2).mean())
        panels = [
            (hillshade(z_a, (low, high)), f"run {item['run']}: start, t={data.times[item['start']]:,.0f} yr", {}),
            (hillshade(z_b, (low, high)), f"simulator, ΔT={dT:,.0f} yr", {}),
            (hillshade(z_pred, (low, high)), "model", {}),
            (true_change, "simulator change (m)", dict(cmap="RdBu", vmin=-limit, vmax=limit)),
            (pred_change, "model change (m)", dict(cmap="RdBu", vmin=-limit, vmax=limit)),
            (z_pred - z_b, f"model - simulator, RMSE {rmse:.2f} m", dict(cmap="RdBu", vmin=-limit, vmax=limit)),
        ]
        for ax, (image, title, kwargs) in zip(row, panels):
            im = ax.imshow(image, origin="lower", **kwargs)
            ax.set_title(title, fontsize=9)
            ax.set_axis_off()
        plt.colorbar(im, ax=row[3:], shrink=0.8)
    plt.show()


@torch.no_grad()
def time_model(model, size: tuple[int, int] = (128, 128), batch_sizes=(1, 64), repeats: int = 20) -> dict:
    """Seconds per model call, for each batch size, on the model's device. Returns {batch size: seconds}."""
    device = next(model.parameters()).device
    out = {}
    for batch in batch_sizes:
        x = torch.rand(batch, 1, *size, device=device)
        cond = torch.rand(batch, len(COND_NAMES), device=device)
        for _ in range(3):  # warm up
            model(x, cond)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            model(x, cond)
        if device.type == "cuda":
            torch.cuda.synchronize()
        out[batch] = (time.perf_counter() - start) / repeats
    return out


def time_simulator(profile: str, split: str, run: int, steps: int = 100, repeats: int = 3) -> dict:
    """Time the simulator on one core for ``steps`` steps of run ``run``, in the datagen environment.

    Returns datagen's JSON: ``seconds_per_step`` (best of ``repeats``) and ``dt`` (yr per step).
    """
    command = ["uv", "run", "--project", str(DATAGEN_DIR), "python", str(DATAGEN_DIR / "scripts" / "time_run.py"),
               profile, split, str(run), str(steps), str(repeats)]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    return json.loads(result.stdout.strip().splitlines()[-1])


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
                     show: bool = True) -> dict:
    """Compare the time to erode one terrain by each elapsed time: simulator steps against model calls.

    The model needs as many calls as it takes to add up to the elapsed time using the elapsed times it
    was trained on (one call if it was trained on that elapsed time). The simulator runs on one CPU core,
    as in dataset generation. The model is timed on the GPU (if any) at batch size 1 and 64, and on the
    CPU at batch size 1.
    """
    model, _, cfg, _ = load_checkpoint(checkpoint)
    params = dict(np.load(ARTIFACTS_DIR / "data" / cfg.data_profile / split / "params.npz"))
    snapshot_dt = float(params["times"][1] - params["times"][0])
    lags = cfg.lags if cfg.lags is not None else range(1, len(params["times"]))
    sim = time_simulator(cfg.data_profile, split, run)
    gpu = time_model(model) if next(model.parameters()).device.type == "cuda" else {}
    cpu_model, _, _, _ = load_checkpoint(checkpoint, device="cpu")
    cpu = time_model(cpu_model, batch_sizes=(1,), repeats=5)
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


# Accuracy. ``predict_pairs`` runs the model once on a sample of pairs; the metric functions below take
# its result. "No change" (the start terrain) is kept as a reference row; skill is defined against it. Errors are in meters, not in the normalized training targets.

METHODS = ("no change", "model")
FACTORS = ("K_ref", "D_over_K", "m", "U_over_K", "start_time", "dT")


@torch.no_grad()
def predict_pairs(checkpoint: Path, split: str = "val", count: int = 512, seed: int = 0, batch_size: int = 64,
                  data_root: Path | None = None) -> dict:
    """Predict ``count`` random pairs of ``split`` with the model.

    Pairs come from runs matching the checkpoint's run filter, at its training elapsed times. Returns a
    dict of arrays: ``z_a``, ``z_b`` (simulator), one end heightmap per entry of ``METHODS``, and per
    pair ``run``, ``start_time``, ``dT`` and the parameters in ``FACTORS``.
    """
    model, normalizer, cfg, _ = load_checkpoint(checkpoint)
    device = next(model.parameters()).device
    root = Path(data_root) if data_root is not None else ARTIFACTS_DIR / "data" / cfg.data_profile
    params = dict(np.load(root / split / "params.npz"))
    data = ErosionPairs(root / split, normalizer, runs=select_runs(params, **dict(cfg.runs)), lags=cfg.lags)
    picks = np.sort(np.random.default_rng(seed).choice(len(data), min(count, len(data)), replace=False))
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(data, picks), batch_size=batch_size)
    out = {"z_a": [], "z_b": [], "model": [], "run": [], "start": [], "end": []}
    for batch in loader:
        pred = model(batch["x"].to(device), batch["cond"].to(device)).float().cpu()
        out["model"].append(denormalize(pred, batch["z_a"], batch["relief"], batch["scale"]).numpy())
        for key in ("z_a", "z_b", "run", "start", "end"):
            out[key].append(batch[key].numpy())
    out = {key: np.concatenate(value) for key, value in out.items()}
    times, run = data.times, out["run"]
    out["start_time"], out["dT"] = times[out["start"]], times[out["end"]] - times[out["start"]]
    out["no change"] = out["z_a"].copy()
    out["K_ref"], out["m"] = params["K_ref"][run], params["m"][run]
    out["D_over_K"], out["U_over_K"] = (params["D"] / params["K_ref"])[run], (params["U"] / params["K_ref"])[run]
    return out


def _per_pair_skill(pairs: dict, method: str) -> np.ndarray:
    mse = ((pairs[method] - pairs["z_b"]) ** 2).mean(axis=(1, 2))
    mse_zero = ((pairs["z_a"] - pairs["z_b"]) ** 2).mean(axis=(1, 2))
    return 1 - mse / mse_zero


def error_metrics(pairs: dict, show: bool = True) -> dict:
    """Height error of each method against the simulator, over all pairs.

    - ``rmse``: root mean squared error of the end heightmap (m). Equal to the error of the change.
    - ``skill``: ``1 - MSE / MSE(no change)`` over all pairs together; ``median_skill`` per pair.
    - ``erosion_ratio``: mean absolute change of the method over that of the simulator. 1 means it moves
      as much material as the simulator; below 1, too little.
    """
    true_change = np.abs(pairs["z_b"] - pairs["z_a"]).mean()
    mse_zero = ((pairs["z_a"] - pairs["z_b"]) ** 2).mean()
    result = {}
    for method in METHODS:
        mse = ((pairs[method] - pairs["z_b"]) ** 2).mean()
        result[method] = {
            "rmse": float(np.sqrt(mse)),
            "skill": float(1 - mse / mse_zero),
            "median_skill": float(np.median(_per_pair_skill(pairs, method))),
            "erosion_ratio": float(np.abs(pairs[method] - pairs["z_a"]).mean() / true_change),
        }
    if show:
        print(f"{len(pairs['z_a'])} pairs; simulator's mean absolute change {true_change:.3f} m\n")
        print(f"{'method':>16} {'RMSE (m)':>9} {'skill':>7} {'median skill':>13} {'erosion ratio':>14}")
        for method, r in result.items():
            print(f"{method:>16} {r['rmse']:9.3f} {r['skill']:7.3f} {r['median_skill']:13.3f} {r['erosion_ratio']:14.3f}")
    return result


def _dilate(mask: np.ndarray) -> np.ndarray:
    """Grow a boolean mask by one cell in all 8 directions."""
    padded = np.pad(mask, 1)
    rows, cols = mask.shape
    return np.any([padded[1 + di:1 + di + rows, 1 + dj:1 + dj + cols]
                   for di in (-1, 0, 1) for dj in (-1, 0, 1)], axis=0)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else 1.0


def hydrology_metrics(pairs: dict, quantile: float = 0.95, show: bool = True, example: int = 0) -> dict:
    """Compare the rivers of each method's terrain with the simulator's.

    Water is routed with ``hydrology.drainage_area`` (D8, no pit filling) on every end heightmap.
    Channels are cells whose drainage area is at least one fixed threshold: the ``quantile`` of the
    simulator's drainage area over all pairs (0.95 marks the top 5% of cells as channel).

    - ``iou``: intersection over union of the method's channels and the simulator's, per pair, cell by
      cell. A channel one cell off counts as a miss.
    - ``iou_1``: the same after growing both masks by one cell, so channels one cell off still match.
    - ``log_area_rmse``: RMSE of log10 drainage area over all cells.
    - ``pits``: interior cells with no lower neighbor, per terrain. The simulator has a few too.

    Per-pair values are averaged. With ``show``, also plots channels for pair ``example``.
    """
    areas = {name: np.stack([drainage_area(z) for z in pairs[name]]) for name in ("z_b", *METHODS)}
    threshold = float(np.quantile(areas["z_b"], quantile))
    channels = {name: a >= threshold for name, a in areas.items()}
    sim = channels["z_b"]
    sim_wide = np.stack([_dilate(c) for c in sim])
    result = {"threshold": threshold, "simulator_pits": float(np.mean([pits(z).sum() for z in pairs["z_b"]]))}
    for method in METHODS:
        mine = channels[method]
        mine_wide = np.stack([_dilate(c) for c in mine])
        result[method] = {
            "iou": float(np.mean([_iou(a, b) for a, b in zip(mine, sim)])),
            "iou_1": float(np.mean([_iou(a, b) for a, b in zip(mine_wide, sim_wide)])),
            "log_area_rmse": float(np.sqrt(((np.log10(areas[method]) - np.log10(areas["z_b"])) ** 2).mean())),
            "pits": float(np.mean([pits(z).sum() for z in pairs[method]])),
        }
    if show:
        print(f"channel threshold: drainage area >= {threshold:.0f} cells (simulator's {quantile:.0%} quantile)")
        print(f"simulator: {result['simulator_pits']:.1f} pits per terrain\n")
        print(f"{'method':>16} {'channel IoU':>12} {'IoU, 1 cell':>12} {'log10 area RMSE':>16} {'pits':>7}")
        for method in METHODS:
            r = result[method]
            print(f"{method:>16} {r['iou']:12.3f} {r['iou_1']:12.3f} {r['log_area_rmse']:16.3f} {r['pits']:7.1f}")
        fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), layout="constrained")
        for ax, method in zip(axes, METHODS):
            image = np.ones((*sim[example].shape, 3))
            image[sim[example] & ~channels[method][example]] = (0.85, 0.2, 0.2)
            image[channels[method][example] & ~sim[example]] = (0.2, 0.4, 0.9)
            image[sim[example] & channels[method][example]] = (0.1, 0.1, 0.1)
            ax.imshow(image, origin="lower")
            ax.set_title(f"pair {example}, {method}: black both, red simulator only, blue {method} only", fontsize=9)
            ax.set_axis_off()
        plt.show()
    return result


def radial_power_spectrum(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Radially averaged power spectrum of a square (H, W) field. Returns (wavelength in cells, power)."""
    size = field.shape[0]
    power = np.abs(np.fft.fftshift(np.fft.fft2(field))) ** 2
    ii, jj = np.indices(field.shape) - size // 2
    radius = np.rint(np.hypot(ii, jj)).astype(int)
    total = np.bincount(radius.ravel(), power.ravel())
    counts = np.bincount(radius.ravel())
    k = np.arange(1, size // 2)
    return size / k, total[k] / counts[k]


def spectrum_comparison(pairs: dict, show: bool = True) -> dict:
    """Power spectrum of the change (end minus start), averaged over pairs, for the simulator and the model.

    Shows at which scales the model has too little change (smoothed away) or too much (noise). The ratio
    to the simulator is 1 where the model matches it. The change is zero at the fixed edges, so no window
    is needed.
    """
    result = {}
    for name in ("z_b", "model"):
        spectra = [radial_power_spectrum(z - z_a)[1] for z, z_a in zip(pairs[name], pairs["z_a"])]
        result["simulator" if name == "z_b" else name] = np.mean(spectra, axis=0)
    result["wavelength"] = radial_power_spectrum(pairs["z_a"][0])[0]
    if show:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2), layout="constrained")
        for name in ("simulator", "model"):
            ax1.loglog(result["wavelength"], result[name], label=name)
        ax2.semilogx(result["wavelength"], result["model"] / result["simulator"], label="model")
        ax2.axhline(1, color="k", lw=0.8)
        for ax in (ax1, ax2):
            ax.set_xlabel("wavelength (cells)")
            ax.invert_xaxis()
            ax.legend()
        ax1.set_ylabel("power of the change")
        ax2.set_ylabel("power / simulator's power")
        ax2.set_ylim(0, 2)
        plt.show()
    return result


def error_by_condition(pairs: dict, bins: int = 6, show: bool = True) -> dict:
    """Per-pair skill against each factor in ``FACTORS``, as medians in equal-count bins.

    Shows where the model fails instead of averaging it away. Returns {factor: {"edges", method: medians}}.
    Factors with one value (for example U_over_K when every run has no uplift) are skipped.
    """
    skill = _per_pair_skill(pairs, "model")
    result = {}
    for factor in FACTORS:
        values = pairs[factor]
        if np.ptp(values) == 0:
            continue
        edges = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)))
        which = np.clip(np.searchsorted(edges, values, side="right") - 1, 0, len(edges) - 2)
        result[factor] = {"edges": edges, "model": np.array([np.median(skill[which == b]) if (which == b).any()
                                                             else np.nan for b in range(len(edges) - 1)])}
    if show:
        fig, axes = plt.subplots(1, len(result), figsize=(3.4 * len(result), 3.4), layout="constrained", squeeze=False)
        for ax, (factor, r) in zip(axes[0], result.items()):
            ax.scatter(pairs[factor], skill, s=3, alpha=0.3, color="C1")
            centers = (r["edges"][:-1] + r["edges"][1:]) / 2
            ax.plot(centers, r["model"], "o-", color="C1", label="median")
            if factor == "K_ref":
                ax.set_xscale("log")
            ax.set_xlabel(factor)
            ax.set_ylim(max(-0.5, skill.min() - 0.05), 1.02)
        axes[0, 0].set_ylabel("skill per pair")
        axes[0, 0].legend(fontsize=8)
        plt.show()
    return result


def error_by_elapsed(pairs: dict, bins: int = 10, quantile: float = 0.95, show: bool = True) -> dict:
    """Error of the model and of no change against the elapsed time of each pair, for a model trained on
    many elapsed times.

    Pairs are grouped into ``bins`` equal-count bins of elapsed time. Per bin: RMSE (m), skill over the
    bin's pairs together, and mean channel IoU (channels as in ``hydrology_metrics``). Also returns the
    median skill per pair on a grid of start time against elapsed time.
    """
    areas = {name: np.stack([drainage_area(z) for z in pairs[name]]) for name in ("z_b", *METHODS)}
    threshold = float(np.quantile(areas["z_b"], quantile))
    sim = areas["z_b"] >= threshold
    sq = {m: ((pairs[m] - pairs["z_b"]) ** 2).mean(axis=(1, 2)) for m in METHODS}
    iou = {m: np.array([_iou(a, b) for a, b in zip(areas[m] >= threshold, sim)]) for m in METHODS}
    dT = pairs["dT"]
    edges = np.unique(np.quantile(dT, np.linspace(0, 1, bins + 1)))
    which = np.clip(np.searchsorted(edges, dT, side="right") - 1, 0, len(edges) - 2)
    groups = [which == b for b in range(len(edges) - 1)]
    result = {"edges": edges, "centers": np.array([dT[g].mean() for g in groups])}
    for m in METHODS:
        result[m] = {
            "rmse": np.array([np.sqrt(sq[m][g].mean()) for g in groups]),
            "skill": np.array([1 - sq[m][g].sum() / sq["no change"][g].sum() for g in groups]),
            "iou": np.array([iou[m][g].mean() for g in groups]),
        }
    # Grid of start time against elapsed time. End times past the last snapshot leave cells empty.
    skill = _per_pair_skill(pairs, "model")
    start_edges = np.linspace(pairs["start_time"].min(), pairs["start_time"].max() + 1, 6)
    dT_edges = np.linspace(dT.min(), dT.max() + 1, 6)
    grid = np.full((5, 5), np.nan)
    for i in range(5):
        for j in range(5):
            cell = ((pairs["start_time"] >= start_edges[i]) & (pairs["start_time"] < start_edges[i + 1])
                    & (dT >= dT_edges[j]) & (dT < dT_edges[j + 1]))
            if cell.any():
                grid[i, j] = np.median(skill[cell])
    result["grid"] = {"start_edges": start_edges, "dT_edges": dT_edges, "median_skill": grid}
    if show:
        fig, axes = plt.subplots(1, 4, figsize=(19, 3.8), layout="constrained")
        for m, style in (("no change", "s--"), ("model", "o-")):
            axes[0].plot(result["centers"], result[m]["rmse"], style, label=m)
            axes[2].plot(result["centers"], result[m]["iou"], style, label=m)
        axes[1].plot(result["centers"], result["model"]["skill"], "o-", color="C1", label="model")
        for ax, label in zip(axes[:3], ("RMSE (m)", "skill", f"channel IoU (top {1 - quantile:.0%} of cells)")):
            ax.set(xlabel="elapsed time (yr)", ylabel=label)
            ax.legend()
        im = axes[3].imshow(grid, origin="lower", aspect="auto", cmap="viridis",
                            extent=(dT_edges[0], dT_edges[-1], start_edges[0], start_edges[-1]))
        axes[3].set(xlabel="elapsed time (yr)", ylabel="start time (yr)", title="median skill per pair")
        for i in range(5):
            for j in range(5):
                if np.isfinite(grid[i, j]):
                    axes[3].text((dT_edges[j] + dT_edges[j + 1]) / 2, (start_edges[i] + start_edges[i + 1]) / 2,
                                 f"{grid[i, j]:.3f}", ha="center", va="center", fontsize=8, color="w")
        plt.colorbar(im, ax=axes[3])
        fig.suptitle(f"{len(dT)} pairs, {len(edges) - 1} bins of elapsed time")
        plt.show()
    return result


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

    Returns RMSE (m) and skill for each, per elapsed time.
    """
    model, normalizer, cfg, _ = load_checkpoint(checkpoint)
    lag = default_lag(cfg) if lag is None else lag
    root = Path(data_root) if data_root is not None else ARTIFACTS_DIR / "data" / cfg.data_profile
    params, _ = load_params(root / split)
    snapshots = np.load(root / split / "snapshots.npy", mmap_mode="r")
    times = params["times"]
    matching = select_runs(params, **dict(cfg.runs))
    chosen = np.random.default_rng(seed).choice(matching, min(runs, len(matching)), replace=False)
    picks = np.arange(0, len(times), lag)
    dT = float(times[lag] - times[0])
    trained = set(cfg.lags) if cfg.lags is not None else set(range(1, len(times)))
    names = ["no change", "one call", "rollout"] + (["direct"] if set(picks[1:]) <= trained else [])
    sq = {name: np.zeros(len(picks) - 1) for name in names}
    for run in chosen:
        sim = np.asarray(snapshots[run, picks], dtype=np.float64)
        chained = rollout(model, normalizer, sim[0], params, run, dT, len(picks) - 1)
        direct = rollout(model, normalizer, sim[0], params, run, dT, len(picks) - 1, "direct") if "direct" in sq else None
        for k in range(1, len(picks)):
            guesses = {
                "no change": sim[0],
                "one call": predict(model, normalizer, sim[k - 1], params, run, dT),
                "rollout": chained[k],
            }
            if direct is not None:
                guesses["direct"] = direct[k]
            for name, z in guesses.items():
                sq[name][k - 1] += ((z - sim[k]) ** 2).mean()
    elapsed = times[picks[1:]] - times[0]
    result = {"elapsed": elapsed}
    for name, total in sq.items():
        result[name] = {"rmse": np.sqrt(total / len(chosen)), "skill": 1 - total / sq["no change"]}
    if show:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
        for name in sq:
            ax1.plot(elapsed, result[name]["rmse"], "o-", label=name)
            if name != "no change":
                ax2.plot(elapsed, result[name]["skill"], "o-", label=name)
        ax1.set_ylabel("RMSE (m)")
        ax2.set_ylabel("skill against no change from t = 0")
        for ax in (ax1, ax2):
            ax.set_xlabel("elapsed time from t = 0 (yr)")
            ax.legend()
        fig.suptitle(f"{len(chosen)} {split} runs, {lag}-snapshot calls ({dT:,.0f} yr)")
        plt.show()
    return result
