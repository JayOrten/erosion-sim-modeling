"""Evaluate a trained model against the simulator: predictions, accuracy, and speed."""

import json
import subprocess
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from .data import ARTIFACTS_DIR, COND_NAMES, ErosionPairs, denormalize, load_params, select_runs
from .diffusion import DEFAULT_STEPS, Denoiser, initial_noise, sample
from .generate import batched_rollout, default_lag, predict, predict_batch
from .hydrology import drainage_area, pits
from .render import hillshade
from .train import load_checkpoint

DATAGEN_DIR = Path(__file__).resolve().parents[2] / "datagen"


@torch.no_grad()
def show_predictions(path: Path, count: int = 3, split: str = "val", data_root: Path | None = None, seed: int = 0) -> None:
    """Plot the model's prediction next to the simulator for ``count`` random pairs of ``split``.

    Columns: start, simulator end, model end, simulator change, model change, error. Changes and error
    share one color scale per row. A diffusion model shows one sample per pair, keyed by ``(seed, pair)``.
    """
    model, normalizer, cfg, _ = load_checkpoint(path)
    device = next(model.parameters()).device
    root = Path(data_root) if data_root is not None else ARTIFACTS_DIR / "data" / cfg.data_profile
    params = dict(np.load(root / split / "params.npz"))
    data = ErosionPairs(root / split, normalizer, runs=select_runs(params, **dict(cfg.runs)), lags=cfg.lags)
    fig, axes = plt.subplots(count, 6, figsize=(19, 3.3 * count), layout="constrained", squeeze=False)
    for row, i in zip(axes, np.random.default_rng(seed).choice(len(data), count, replace=False)):
        item = data[int(i)]
        x, cond = item["x"][None].to(device), item["cond"][None].to(device)
        if isinstance(model, Denoiser):
            pred = sample(model, x, cond, initial_noise([(seed, int(i))], x.shape[-2:])).cpu()
        else:
            pred = model(x, cond).float().cpu()
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


def show_comparison(checkpoints: dict[str, Path], count: int = 3, split: str = "val", samples: int = 2,
                    seed: int = 0, data_root: Path | None = None) -> None:
    """Plot the change predicted by several models next to the simulator's, on the same random pairs.

    Columns: start, simulator change, then each regression model's change and ``samples`` samples of
    each diffusion model, then the spread across those samples. Changes share one color scale per row.
    Pairs are drawn as in ``show_predictions``, from the first checkpoint's data and run filter.
    """
    loaded = {name: load_checkpoint(path) for name, path in checkpoints.items()}
    cfg = next(iter(loaded.values()))[2]
    root = Path(data_root) if data_root is not None else ARTIFACTS_DIR / "data" / cfg.data_profile
    params = dict(np.load(root / split / "params.npz"))
    data = ErosionPairs(root / split, next(iter(loaded.values()))[1], runs=select_runs(params, **dict(cfg.runs)),
                        lags=cfg.lags)
    picks = np.random.default_rng(seed).choice(len(data), count, replace=False)
    rows = []
    for i in picks:
        item = data[int(i)]
        z_a, z_b = item["z_a"].numpy().astype(np.float64), item["z_b"].numpy()
        dT = float(data.times[item["end"]] - data.times[item["start"]])
        panels = [("start", z_a), (f"simulator change, ΔT={dT:,.0f} yr", z_b - z_a)]
        for name, (model, normalizer, _, _) in loaded.items():
            z = predict(model, normalizer, z_a, params, item["run"], dT, samples=samples, seed=seed, key=(int(i),))
            if isinstance(model, Denoiser):
                panels += [(f"{name} sample {k + 1}", z[k] - z_a) for k in range(samples)]
                panels.append((f"{name} spread (m)", z.std(0)))
            else:
                panels.append((name, z - z_a))
        rows.append((item, z_b - z_a, panels))
    width = len(rows[0][2])
    fig, axes = plt.subplots(count, width, figsize=(3.2 * width, 3.3 * count), layout="constrained", squeeze=False)
    for row, (item, change, panels) in zip(axes, rows):
        limit = np.abs(change).max()
        for ax, (title, image) in zip(row, panels):
            if title == "start":
                ax.imshow(hillshade(image, (image.min(), image.max())), origin="lower")
                title = f"run {item['run']}: start"
            elif title.endswith("spread (m)"):
                im_spread = ax.imshow(image, origin="lower", cmap="magma", vmin=0)
                plt.colorbar(im_spread, ax=ax, shrink=0.8)
            else:
                im = ax.imshow(image, origin="lower", cmap="RdBu", vmin=-limit, vmax=limit)
            ax.set_title(title, fontsize=9)
            ax.set_axis_off()
        plt.colorbar(im, ax=row[1], shrink=0.8, label="change (m)")
    plt.show()


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
        cond = torch.rand(batch, len(COND_NAMES), device=device)
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
    params = dict(np.load(ARTIFACTS_DIR / "data" / cfg.data_profile / split / "params.npz"))
    snapshot_dt = float(params["times"][1] - params["times"][0])
    lags = cfg.lags if cfg.lags is not None else range(1, len(params["times"]))
    sim = time_simulator(cfg.data_profile, split, run)
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


# Accuracy. ``predict_pairs`` runs one or more models on a sample of pairs; the metric functions below
# take its result. "No change" (the start terrain) is kept as a reference row; skill is defined against
# it. Errors are in meters, not in the normalized training targets.

FACTORS = ("K_ref", "D_over_K", "m", "U_over_K", "start_time", "dT")


@torch.no_grad()
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
    cfg = next(iter(loaded.values()))[2]
    root = Path(data_root) if data_root is not None else ARTIFACTS_DIR / "data" / cfg.data_profile
    params = dict(np.load(root / split / "params.npz"))
    runs = select_runs(params, **dict(cfg.runs))
    out = {"methods": ["no change"], "kind": {"no change": "baseline"}, "spread": {}}
    raw = {"z_a": [], "z_b": [], "run": [], "start": [], "end": []}
    picks = None
    for name, (model, normalizer, _, _) in loaded.items():
        # Each model gets the pairs through its own normalizer; the raw heightmaps are the same.
        data = ErosionPairs(root / split, normalizer, runs=runs, lags=cfg.lags)
        if picks is None:
            picks = np.sort(np.random.default_rng(seed).choice(len(data), min(count, len(data)), replace=False))
        device = next(model.parameters()).device
        diffusion = isinstance(model, Denoiser)
        per_batch = max(1, batch_size // samples) if diffusion else batch_size
        loader = torch.utils.data.DataLoader(torch.utils.data.Subset(data, picks), batch_size=per_batch)
        preds, means, spreads, done = [], [], [], 0
        autocast = torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16 and device.type == "cuda")
        bar = tqdm(total=len(picks), desc=name, unit="pair", disable=not progress)
        for batch in loader:
            x, cond = batch["x"].to(device), batch["cond"].to(device)
            bar.update(len(x))
            if diffusion:
                n = len(x)
                keys = [(seed, int(picks[done + b]), k) for b in range(n) for k in range(samples)]
                with autocast:
                    y = sample(model, x.repeat_interleave(samples, 0), cond.repeat_interleave(samples, 0),
                               initial_noise(keys, x.shape[-2:]), steps).float().cpu()
                size = (batch["relief"] * batch["scale"]).float().reshape(n, 1, 1, 1)
                z = batch["z_a"][:, None] + y.reshape(n, samples, *y.shape[-2:]) * size
                preds.append(z[:, 0].numpy())
                means.append(z.mean(1).numpy())
                spreads.append(z.std(1, correction=0).numpy())
                done += n
            else:
                with autocast:
                    pred = model(x, cond).float().cpu()
                preds.append(denormalize(pred, batch["z_a"], batch["relief"], batch["scale"]).numpy())
            if name == next(iter(loaded)):
                for key in raw:
                    raw[key].append(batch[key].numpy())
        bar.close()
        if diffusion:
            single, mean = f"{name} (1 sample)", f"{name} (mean of {samples})"
            out[single], out[mean] = np.concatenate(preds), np.concatenate(means)
            out["spread"][name] = np.concatenate(spreads)
            out["methods"] += [single, mean]
            out["kind"].update({single: "sample", mean: "mean"})
        else:
            out[name] = np.concatenate(preds)
            out["methods"].append(name)
            out["kind"][name] = "regression"
    out.update({key: np.concatenate(value) for key, value in raw.items()})
    times, run = data.times, out["run"]
    out["start_time"], out["dT"] = times[out["start"]], times[out["end"]] - times[out["start"]]
    out["no change"] = out["z_a"].copy()
    out["K_ref"], out["m"] = params["K_ref"][run], params["m"][run]
    out["D_over_K"], out["U_over_K"] = (params["D"] / params["K_ref"])[run], (params["U"] / params["K_ref"])[run]
    return out


def _models(pairs: dict) -> list[str]:
    """Every method except the no-change baseline."""
    return pairs["methods"][1:]


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
    for method in pairs["methods"]:
        mse = ((pairs[method] - pairs["z_b"]) ** 2).mean()
        result[method] = {
            "rmse": float(np.sqrt(mse)),
            "skill": float(1 - mse / mse_zero),
            "median_skill": float(np.median(_per_pair_skill(pairs, method))),
            "erosion_ratio": float(np.abs(pairs[method] - pairs["z_a"]).mean() / true_change),
        }
    if show:
        w = max(16, *map(len, pairs["methods"]))
        print(f"{len(pairs['z_a'])} pairs; simulator's mean absolute change {true_change:.3f} m\n")
        print(f"{'method':>{w}} {'RMSE (m)':>9} {'skill':>7} {'median skill':>13} {'erosion ratio':>14}")
        for method, r in result.items():
            print(f"{method:>{w}} {r['rmse']:9.3f} {r['skill']:7.3f} {r['median_skill']:13.3f} {r['erosion_ratio']:14.3f}")
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
    methods = pairs["methods"]
    areas = {name: np.stack([drainage_area(z) for z in pairs[name]]) for name in ("z_b", *methods)}
    threshold = float(np.quantile(areas["z_b"], quantile))
    channels = {name: a >= threshold for name, a in areas.items()}
    sim = channels["z_b"]
    sim_wide = np.stack([_dilate(c) for c in sim])
    result = {"threshold": threshold, "simulator_pits": float(np.mean([pits(z).sum() for z in pairs["z_b"]]))}
    for method in methods:
        mine = channels[method]
        mine_wide = np.stack([_dilate(c) for c in mine])
        result[method] = {
            "iou": float(np.mean([_iou(a, b) for a, b in zip(mine, sim)])),
            "iou_1": float(np.mean([_iou(a, b) for a, b in zip(mine_wide, sim_wide)])),
            "log_area_rmse": float(np.sqrt(((np.log10(areas[method]) - np.log10(areas["z_b"])) ** 2).mean())),
            "pits": float(np.mean([pits(z).sum() for z in pairs[method]])),
        }
    if show:
        w = max(16, *map(len, methods))
        print(f"channel threshold: drainage area >= {threshold:.0f} cells (simulator's {quantile:.0%} quantile)")
        print(f"simulator: {result['simulator_pits']:.1f} pits per terrain\n")
        print(f"{'method':>{w}} {'channel IoU':>12} {'IoU, 1 cell':>12} {'log10 area RMSE':>16} {'pits':>7}")
        for method in methods:
            r = result[method]
            print(f"{method:>{w}} {r['iou']:12.3f} {r['iou_1']:12.3f} {r['log_area_rmse']:16.3f} {r['pits']:7.1f}")
        fig, axes = plt.subplots(1, len(methods), figsize=(5.5 * len(methods), 5.8), layout="constrained", squeeze=False)
        for ax, method in zip(axes[0], methods):
            image = np.ones((*sim[example].shape, 3))
            image[sim[example] & ~channels[method][example]] = (0.85, 0.2, 0.2)
            image[channels[method][example] & ~sim[example]] = (0.2, 0.4, 0.9)
            image[sim[example] & channels[method][example]] = (0.1, 0.1, 0.1)
            ax.imshow(image, origin="lower")
            ax.set_title(f"pair {example}, {method}", fontsize=9)
            ax.set_axis_off()
        fig.suptitle("channels: black both, red simulator only, blue method only", fontsize=10)
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
    """Power spectrum of the change (end minus start), averaged over pairs, for the simulator and each method.

    Shows at which scales a method has too little change (smoothed away) or too much (noise). The ratio
    to the simulator is 1 where the method matches it. The change is zero at the fixed edges, so no
    window is needed.
    """
    result = {}
    for name in ("z_b", *_models(pairs)):
        spectra = [radial_power_spectrum(z - z_a)[1] for z, z_a in zip(pairs[name], pairs["z_a"])]
        result["simulator" if name == "z_b" else name] = np.mean(spectra, axis=0)
    result["wavelength"] = radial_power_spectrum(pairs["z_a"][0])[0]
    if show:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2), layout="constrained")
        for name in ("simulator", *_models(pairs)):
            ax1.loglog(result["wavelength"], result[name], label=name)
        for i, name in enumerate(_models(pairs), start=1):
            ax2.semilogx(result["wavelength"], result[name] / result["simulator"], color=f"C{i}", label=name)
        ax2.axhline(1, color="k", lw=0.8)
        for ax in (ax1, ax2):
            ax.set_xlabel("wavelength (cells)")
            ax.invert_xaxis()
            ax.legend(fontsize=8)
        ax1.set_ylabel("power of the change")
        ax2.set_ylabel("power / simulator's power")
        ax2.set_ylim(0, 2)
        plt.show()
    return result


def error_by_condition(pairs: dict, bins: int = 6, show: bool = True) -> dict:
    """Per-pair skill of each method against each factor in ``FACTORS``, as medians in equal-count bins.

    Shows where a method fails instead of averaging it away. Returns {factor: {"edges", method: medians}}.
    Factors with one value (for example U_over_K when every run has no uplift) are skipped.
    """
    skills = {m: _per_pair_skill(pairs, m) for m in _models(pairs)}
    result = {}
    for factor in FACTORS:
        values = pairs[factor]
        if np.ptp(values) == 0:
            continue
        edges = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)))
        which = np.clip(np.searchsorted(edges, values, side="right") - 1, 0, len(edges) - 2)
        result[factor] = {"edges": edges}
        for m, skill in skills.items():
            result[factor][m] = np.array([np.median(skill[which == b]) if (which == b).any() else np.nan
                                          for b in range(len(edges) - 1)])
    if show:
        low = min(s.min() for s in skills.values())
        fig, axes = plt.subplots(1, len(result), figsize=(3.4 * len(result), 3.4), layout="constrained", squeeze=False)
        for ax, (factor, r) in zip(axes[0], result.items()):
            centers = (r["edges"][:-1] + r["edges"][1:]) / 2
            for i, (m, skill) in enumerate(skills.items(), start=1):
                ax.scatter(pairs[factor], skill, s=3, alpha=0.2, color=f"C{i}")
                ax.plot(centers, r[m], "o-", color=f"C{i}", label=m)
            if factor == "K_ref":
                ax.set_xscale("log")
            ax.set_xlabel(factor)
            ax.set_ylim(max(-0.5, low - 0.05), 1.02)
        axes[0, 0].set_ylabel("skill per pair (lines: medians)")
        axes[0, 0].legend(fontsize=7)
        plt.show()
    return result


def sample_spread(pairs: dict, bins: int = 10, show: bool = True, example: int = 0) -> dict:
    """Where each diffusion model's samples disagree, and whether that is where the errors are.

    The spread is the per-pixel standard deviation across a diffusion model's samples (m). Interior
    pixels of all pairs are pooled and grouped into ``bins`` equal-count bins of spread. For each method,
    the mean absolute error in each bin shows whether errors are larger where the samples disagree.
    ``correlation`` is the Pearson correlation of spread and absolute error over all interior pixels.
    Returns {diffusion name: {"mean_spread", "edges", "centers", method: {"error", "correlation"}}}.
    """
    result = {}
    for name, spread in pairs["spread"].items():
        s = spread[:, 1:-1, 1:-1].ravel()
        edges = np.quantile(s, np.linspace(0, 1, bins + 1))
        which = np.clip(np.searchsorted(edges, s, side="right") - 1, 0, bins - 1)
        counts = np.bincount(which, minlength=bins)
        r = {"mean_spread": float(s.mean()), "edges": edges,
             "centers": np.bincount(which, s, minlength=bins) / counts}
        for m in _models(pairs):
            err = np.abs(pairs[m] - pairs["z_b"])[:, 1:-1, 1:-1].ravel()
            r[m] = {"error": np.bincount(which, err, minlength=bins) / counts,
                    "correlation": float(np.corrcoef(s, err)[0, 1])}
        result[name] = r
    if show:
        for name, r in result.items():
            w = max(16, *map(len, _models(pairs)))
            print(f"{name}: mean spread across samples {r['mean_spread']:.3f} m\n")
            print(f"{'method':>{w}} {'corr(spread, |error|)':>22}")
            for m in _models(pairs):
                print(f"{m:>{w}} {r[m]['correlation']:22.3f}")
            maps = [(pairs["spread"][name][example], f"{name}: spread across samples (m)")]
            maps += [(np.abs(pairs[m] - pairs["z_b"])[example], f"|error| of {m} (m)") for m in _models(pairs)]
            fig, axes = plt.subplots(1, 1 + len(maps), figsize=(4.6 * (1 + len(maps)), 4.2), layout="constrained")
            for i, m in enumerate(_models(pairs), start=1):
                axes[0].plot(r["centers"], r[m]["error"], "o-", color=f"C{i}", label=m)
            axes[0].plot(r["centers"], r["centers"], "k--", lw=0.8, label="error = spread")
            axes[0].set(xscale="log", yscale="log", xlabel=f"spread of {name} (m)", ylabel="mean |error| (m)",
                        title="error against spread, interior pixels")
            axes[0].legend(fontsize=7)
            limit = max(np.quantile(img, 0.99) for img, _ in maps)
            for ax, (img, title) in zip(axes[1:], maps):
                im = ax.imshow(img, origin="lower", cmap="magma", vmin=0, vmax=limit)
                ax.set_title(f"pair {example}, {title}", fontsize=8)
                ax.set_axis_off()
            plt.colorbar(im, ax=axes[1:], shrink=0.8)
            plt.show()
    return result


def error_by_elapsed(pairs: dict, bins: int = 10, quantile: float = 0.95, show: bool = True) -> dict:
    """Error of each method against the elapsed time of each pair, for models trained on many elapsed
    times.

    Pairs are grouped into ``bins`` equal-count bins of elapsed time. Per bin: RMSE (m), skill over the
    bin's pairs together, and mean channel IoU (channels as in ``hydrology_metrics``). Also returns,
    per method, the median skill per pair on a grid of start time against elapsed time; the plot shows
    the grid of the first method.
    """
    if len(np.unique(pairs["dT"])) < 2:
        if show:
            print(f"skipped: every pair has the same elapsed time ({pairs['dT'][0]:,.0f} yr); "
                  "this needs a model trained on many elapsed times")
        return {}
    methods = pairs["methods"]
    areas = {name: np.stack([drainage_area(z) for z in pairs[name]]) for name in ("z_b", *methods)}
    threshold = float(np.quantile(areas["z_b"], quantile))
    sim = areas["z_b"] >= threshold
    sq = {m: ((pairs[m] - pairs["z_b"]) ** 2).mean(axis=(1, 2)) for m in methods}
    iou = {m: np.array([_iou(a, b) for a, b in zip(areas[m] >= threshold, sim)]) for m in methods}
    dT = pairs["dT"]
    edges = np.unique(np.quantile(dT, np.linspace(0, 1, bins + 1)))
    which = np.clip(np.searchsorted(edges, dT, side="right") - 1, 0, len(edges) - 2)
    groups = [which == b for b in range(len(edges) - 1)]
    result = {"edges": edges, "centers": np.array([dT[g].mean() for g in groups])}
    for m in methods:
        result[m] = {
            "rmse": np.array([np.sqrt(sq[m][g].mean()) for g in groups]),
            "skill": np.array([1 - sq[m][g].sum() / sq["no change"][g].sum() for g in groups]),
            "iou": np.array([iou[m][g].mean() for g in groups]),
        }
    # Grid of start time against elapsed time. End times past the last snapshot leave cells empty.
    start_edges = np.linspace(pairs["start_time"].min(), pairs["start_time"].max() + 1, 6)
    dT_edges = np.linspace(dT.min(), dT.max() + 1, 6)
    result["grid"] = {"start_edges": start_edges, "dT_edges": dT_edges}
    for m in _models(pairs):
        skill = _per_pair_skill(pairs, m)
        grid = np.full((5, 5), np.nan)
        for i in range(5):
            for j in range(5):
                cell = ((pairs["start_time"] >= start_edges[i]) & (pairs["start_time"] < start_edges[i + 1])
                        & (dT >= dT_edges[j]) & (dT < dT_edges[j + 1]))
                if cell.any():
                    grid[i, j] = np.median(skill[cell])
        result["grid"][m] = grid
    if show:
        fig, axes = plt.subplots(1, 4, figsize=(19, 3.8), layout="constrained")
        for i, m in enumerate(methods):
            style = "s--" if m == "no change" else "o-"
            axes[0].plot(result["centers"], result[m]["rmse"], style, color=f"C{i}", label=m)
            axes[2].plot(result["centers"], result[m]["iou"], style, color=f"C{i}", label=m)
            if m != "no change":
                axes[1].plot(result["centers"], result[m]["skill"], "o-", color=f"C{i}", label=m)
        for ax, label in zip(axes[:3], ("RMSE (m)", "skill", f"channel IoU (top {1 - quantile:.0%} of cells)")):
            ax.set(xlabel="elapsed time (yr)", ylabel=label)
            ax.legend(fontsize=7)
        first = _models(pairs)[0]
        grid = result["grid"][first]
        im = axes[3].imshow(grid, origin="lower", aspect="auto", cmap="viridis",
                            extent=(dT_edges[0], dT_edges[-1], start_edges[0], start_edges[-1]))
        axes[3].set(xlabel="elapsed time (yr)", ylabel="start time (yr)", title=f"median skill per pair, {first}")
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

    A diffusion model draws one sample per call. Two calls of ``rollout_error`` with the same ``seed`` and
    the same run filter use the same runs, so their results can be compared directly.

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
    chosen, steps = np.sort(chosen), len(picks) - 1
    sim = np.asarray(snapshots[chosen][:, picks], dtype=np.float64)  # (runs, steps + 1, H, W)

    def sq_error(z):  # summed over runs, per frame
        return ((z - sim[:, 1:]) ** 2).mean(axis=(2, 3)).sum(0)

    sq = {"no change": sq_error(sim[:, :1])}
    # One call per frame, from the simulator's previous snapshot. All frames of all runs go in together.
    keys = [(seed, "one call", int(run), k, 0) for run in chosen for k in range(1, steps + 1)]
    sq["one call"] = sq_error(predict_batch(model, normalizer, sim[:, :-1].reshape(-1, *sim.shape[2:]), params,
                                            np.repeat(chosen, steps), dT, keys).reshape(sim[:, 1:].shape))
    sq["rollout"] = sq_error(batched_rollout(model, normalizer, sim[:, 0], params, chosen, dT, steps, seed=seed)[:, 1:])
    if "direct" in names:
        sq["direct"] = sq_error(batched_rollout(model, normalizer, sim[:, 0], params, chosen, dT, steps, "direct",
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


def _fine_detail(spectrum: dict, name: str, wavelength: float = 4.0) -> float:
    """Mean of the method's power over the simulator's, at wavelengths up to ``wavelength`` cells."""
    fine = spectrum["wavelength"] <= wavelength
    return float(np.mean(spectrum[name][fine] / spectrum["simulator"][fine]))


def step_sweep(checkpoint: Path, steps=(8, 16, 32, 64), regression: Path | None = None, split: str = "val",
               count: int = 256, samples: int = 4, seed: int = 0, show: bool = True,
               data_root: Path | None = None) -> dict:
    """Metrics of a diffusion checkpoint against the number of sampler steps.

    Every step count sees the same ``count`` pairs and the same noise, so only the step count differs.
    Per step count, for one sample and for the mean of ``samples``: RMSE (m), channel IoU, pits per
    terrain, and fine detail (power of the change at wavelengths up to 4 cells, over the simulator's;
    1 is right). Also the time per sample (ms), from the wall time of the whole prediction. A
    ``regression`` checkpoint adds its numbers as a reference.
    """
    common = dict(split=split, count=count, seed=seed, progress=False, data_root=data_root)
    predict_pairs(checkpoint, **{**common, "count": 1}, samples=1, steps=2)  # warm up the GPU
    result = {"steps": np.array(steps)}
    names = {"1 sample": "model (1 sample)", f"mean of {samples}": f"model (mean of {samples})"}
    rows = {label: {"rmse": [], "iou": [], "pits": [], "fine": []} for label in names}
    times = []
    for n in steps:
        start = time.perf_counter()
        pairs = predict_pairs(checkpoint, **common, samples=samples, steps=n)
        times.append(1000 * (time.perf_counter() - start) / (len(pairs["z_b"]) * samples))
        errors = error_metrics(pairs, show=False)
        hydro = hydrology_metrics(pairs, show=False)
        spectrum = spectrum_comparison(pairs, show=False)
        for label, method in names.items():
            rows[label]["rmse"].append(errors[method]["rmse"])
            rows[label]["iou"].append(hydro[method]["iou"])
            rows[label]["pits"].append(hydro[method]["pits"])
            rows[label]["fine"].append(_fine_detail(spectrum, method))
    for label in rows:
        result[label] = {key: np.array(value) for key, value in rows[label].items()}
    result["ms_per_sample"] = np.array(times)
    result["simulator_pits"] = hydro["simulator_pits"]
    result["no change"] = {"rmse": errors["no change"]["rmse"], "iou": hydro["no change"]["iou"]}
    if regression is not None:
        pairs = predict_pairs(regression, **common)
        errors, hydro = error_metrics(pairs, show=False), hydrology_metrics(pairs, show=False)
        result["regression"] = {"rmse": errors["model"]["rmse"], "iou": hydro["model"]["iou"],
                                "pits": hydro["model"]["pits"],
                                "fine": _fine_detail(spectrum_comparison(pairs, show=False), "model")}
    if show:
        print(f"{len(pairs['z_b'])} pairs, {samples} samples each; simulator: {result['simulator_pits']:.1f} pits")
        for label in rows:
            r = result[label]
            print(f"\ndiffusion, {label}")
            print(f"{'steps':>6} {'ms/sample':>10} {'RMSE (m)':>9} {'IoU':>6} {'pits':>6} {'fine':>5}")
            for i, n in enumerate(steps):
                print(f"{n:>6} {times[i]:>10.1f} {r['rmse'][i]:>9.3f} {r['iou'][i]:>6.3f} {r['pits'][i]:>6.1f}"
                      f" {r['fine'][i]:>5.2f}")
        if regression is not None:
            r = result["regression"]
            print(f"\nregression: RMSE {r['rmse']:.3f}, IoU {r['iou']:.3f}, pits {r['pits']:.1f}, fine {r['fine']:.2f}")
        panels = [("rmse", "RMSE (m)"), ("iou", "channel IoU"), ("pits", "pits per terrain"),
                  ("fine", "fine detail (1 = simulator)")]
        fig, axes = plt.subplots(1, 5, figsize=(20, 3.8), layout="constrained")
        for ax, (key, ylabel) in zip(axes, panels):
            for i, label in enumerate(rows):
                ax.plot(steps, result[label][key], "o-", color=f"C{i + 1}", label=f"diffusion, {label}")
            if regression is not None:
                ax.axhline(result["regression"][key], color="C0", ls="--", label="regression")
            if key == "pits":
                ax.axhline(result["simulator_pits"], color="k", ls=":", label="simulator")
            if key == "fine":
                ax.axhline(1, color="k", ls=":", label="simulator")
            if key == "iou":
                ax.axhline(result["no change"]["iou"], color="gray", ls=":", label="no change")
            ax.set_ylabel(ylabel)
        axes[4].plot(steps, times, "o-", color="k")
        axes[4].set_ylabel("ms per sample")
        for ax in axes:
            ax.set_xscale("log", base=2)
            ax.set_xticks(steps, [str(n) for n in steps])
            ax.set_xlabel("sampler steps")
        axes[0].legend(fontsize=8)
        fig.suptitle(f"{Path(checkpoint).stem}: metrics against sampler steps")
        plt.show()
    return result
