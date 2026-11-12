"""Pictures of model output next to the simulator: still comparisons and animations."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from ..data import ErosionPairs, Split
from ..diffusion import Denoiser
from ..hydrology import drainage_area
from ..paths import ANIMATIONS_DIR
from ..predict import default_lag, predict, predict_batch, rollout
from ..render import hillshade, render_frames
from ..train import load_checkpoint


@torch.no_grad()
def show_predictions(path: Path, count: int = 3, split: str = "val", data_root: Path | None = None, seed: int = 0) -> None:
    """Plot the model's prediction next to the simulator for ``count`` random pairs of ``split``.

    Columns: start, simulator end, model end, simulator change, model change, error. Changes and error
    share one color scale per row. A diffusion model shows one sample per pair, keyed by ``(seed, pair)``.
    """
    model, scaling, cfg, _ = load_checkpoint(path)
    split_data = Split.open(cfg.data_profile, split, data_root)
    params = split_data.params
    data = ErosionPairs(split_data, scaling, runs=split_data.select(**dict(cfg.runs)), lags=cfg.lags)
    fig, axes = plt.subplots(count, 6, figsize=(19, 3.3 * count), layout="constrained", squeeze=False)
    for row, i in zip(axes, np.random.default_rng(seed).choice(len(data), count, replace=False)):
        item = data[int(i)]
        z_a, z_b = item["z_a"].numpy().astype(np.float64), item["z_b"].numpy()
        dT = data.elapsed(int(i))
        z_pred = predict_batch(model, scaling, z_a[None], params, item["run"], dT, [(seed, int(i))], bf16=False)[0]
        low, high = min(z_a.min(), z_b.min()), max(z_a.max(), z_b.max())
        true_change, pred_change = z_b - z_a, z_pred - z_a
        limit = np.abs(true_change).max()
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
    split_data = Split.open(cfg.data_profile, split, data_root)
    params = split_data.params
    data = ErosionPairs(split_data, next(iter(loaded.values()))[1], runs=split_data.select(**dict(cfg.runs)),
                        lags=cfg.lags)
    picks = np.random.default_rng(seed).choice(len(data), count, replace=False)
    rows = []
    for i in picks:
        item = data[int(i)]
        z_a, z_b = item["z_a"].numpy().astype(np.float64), item["z_b"].numpy()
        dT = float(data.elapsed(int(i)))
        panels = [("start", z_a), (f"simulator change, ΔT={dT:,.0f} yr", z_b - z_a)]
        for name, (model, scaling, _, _) in loaded.items():
            z = predict(model, scaling, z_a, params, item["run"], dT, samples=samples, seed=seed, key=(int(i),))
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


def comparison_gif(
    checkpoint: Path,
    run: int,
    split: str = "val",
    lag: int | None = None,
    mode: str = "rollout",
    fps: float = 2.0,
    vert_exag: float = 1.5,
    out: Path | None = None,
    data_root: Path | None = None,
) -> Path:
    """Animate the simulator and the model side by side over one run, and save a GIF. Return its path.

    Frames are ``lag`` snapshots apart, starting at snapshot 0. ``lag`` defaults to the model's training
    elapsed time if it was trained on one, else 1. Top row: 2D (simulator, model). Bottom row: 3D, with
    rivers in blue. Both sides share one color scale and one camera.
    """
    model, scaling, cfg, _ = load_checkpoint(checkpoint)
    if lag is None:
        lag = default_lag(cfg)
    data = Split.open(cfg.data_profile, split, data_root)
    params, snapshots, times = data.params, data.snapshots, data.times
    picks = np.arange(0, len(times), lag)
    sim = np.asarray(snapshots[run, picks], dtype=np.float64)
    dT = float(times[lag] - times[0])
    pred = rollout(model, scaling, sim[0], params, run, dT, len(picks) - 1, mode)

    # One render pass over both sequences, so they share the camera and color scale.
    both = np.concatenate([sim, pred])
    areas = np.stack([drainage_area(z) for z in both])
    frames_3d = render_frames(both, areas, vert_exag=vert_exag, window_size=(800, 560))
    z_range = (both.min(), both.max())

    frames = []
    for i, t in enumerate(times[picks]):
        fig, axes = plt.subplots(1, 2, figsize=(16, 8.4), dpi=50)
        for ax, z, name in zip(axes, (sim[i], pred[i]), ("simulator", f"model ({mode})")):
            ax.imshow(hillshade(z, z_range), origin="lower")
            ax.set_title(f"{name}, t = {t:,.0f} yr", fontsize=20)
            ax.set_axis_off()
        rmse = np.sqrt(((pred[i] - sim[i]) ** 2).mean())
        fig.suptitle(f"run {run} ({split}), RMSE {rmse:.2f} m", fontsize=22)
        fig.canvas.draw()
        top = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        plt.close(fig)
        bottom = np.concatenate([frames_3d[i], frames_3d[len(picks) + i]], axis=1)
        top = np.asarray(Image.fromarray(top).resize((bottom.shape[1], round(top.shape[0] * bottom.shape[1] / top.shape[1]))))
        frames.append(Image.fromarray(np.concatenate([top, bottom], axis=0)))

    out = Path(out) if out is not None else ANIMATIONS_DIR / f"{cfg.objective}-{cfg.profile}-{split}-run{run}-{mode}.gif"
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0)
    return out
