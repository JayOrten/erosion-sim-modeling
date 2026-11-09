"""Use a trained model to erode terrain, and compare it with the simulator."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from .data import ARTIFACTS_DIR, Normalizer, cond_values, load_params
from .diffusion import DEFAULT_STEPS, Denoiser, initial_noise, sample
from .hydrology import drainage_area
from .render import hillshade, render_frames
from .train import load_checkpoint


@torch.no_grad()
def predict_batch(model, normalizer: Normalizer, z_a: np.ndarray, params: dict, runs, dT, keys=None,
                  steps: int = DEFAULT_STEPS, batch_size: int = 64, bf16: bool = True) -> np.ndarray:
    """One prediction per row: the model's heightmaps (m) after eroding ``z_a[i]`` for ``dT[i]`` years with
    run ``runs[i]``'s parameters. ``z_a`` is (N, H, W); ``runs`` and ``dT`` are length N, or one value
    for every row. Returns (N, H, W).

    The rows go through the model ``batch_size`` at a time. A diffusion model draws one sample per row,
    from the noise for ``keys[i]`` (a tuple), with ``steps`` sampler steps; regression ignores ``keys``
    and ``steps``. ``bf16`` runs the network in 16-bit autocast on the GPU (ignored on CPU).
    """
    z_a = np.asarray(z_a, dtype=np.float64)
    n = len(z_a)
    runs = np.broadcast_to(np.asarray(runs, dtype=np.int64), (n,))
    dT = np.broadcast_to(np.asarray(dT, dtype=np.float64), (n,))
    diffusion = isinstance(model, Denoiser)
    if diffusion and (keys is None or len(keys) != n):
        raise ValueError("a diffusion model needs one noise key per row")
    device = next(model.parameters()).device
    autocast = torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16 and device.type == "cuda")
    low = z_a.min(axis=(1, 2))
    relief = np.ptp(z_a, axis=(1, 2))
    scale = normalizer.target_scale(params["K_ref"][runs], dT)
    cond = normalizer.cond(cond_values(params, runs, dT, relief))
    out = np.empty_like(z_a)
    for i in range(0, n, batch_size):
        rows = slice(i, i + batch_size)
        x = torch.as_tensor((z_a[rows] - low[rows, None, None]) / relief[rows, None, None],
                            dtype=torch.float32, device=device)[:, None]
        c = torch.as_tensor(cond[rows], dtype=torch.float32, device=device)
        with autocast:
            if diffusion:
                y = sample(model, x, c, initial_noise(list(keys[rows]), z_a.shape[-2:]), steps)
            else:
                y = model(x, c)
        y = y.float()[:, 0].cpu().numpy()
        out[rows] = z_a[rows] + y * (relief[rows] * scale[rows])[:, None, None]
    return out


def predict(model, normalizer: Normalizer, z_a: np.ndarray, params: dict, run: int, dT: float, samples: int = 1,
            seed: int = 0, key: tuple = (), steps: int = DEFAULT_STEPS) -> np.ndarray:
    """Return the model's heightmap (m) after eroding ``z_a`` for ``dT`` years with run ``run``'s parameters.

    A regression model returns one heightmap, (H, W). A diffusion model returns ``samples`` of them,
    (samples, H, W). Sample ``i`` uses the noise for key ``(seed, *key, i)``, so it is reproducible;
    ``steps`` is the number of sampler steps. Regression ignores those arguments.
    """
    if isinstance(model, Denoiser):
        keys = [(seed, *key, i) for i in range(samples)]
        return predict_batch(model, normalizer, np.repeat(np.asarray(z_a)[None], samples, 0), params, run, dT,
                             keys, steps)
    return predict_batch(model, normalizer, np.asarray(z_a)[None], params, run, dT)[0]


def rollout(model, normalizer: Normalizer, z0: np.ndarray, params: dict, run: int, dT: float, steps: int,
            mode: str = "rollout", seed: int = 0) -> np.ndarray:
    """Model heightmaps at times 0, dT, 2 dT, ..., steps dT, shape (steps + 1, H, W).

    ``mode="rollout"`` feeds each prediction back in as the next start. ``mode="direct"`` predicts each
    time straight from ``z0``, which only works for elapsed times the model was trained on; its frames
    go through the model together.

    A diffusion model draws one sample per frame. In ``"direct"`` mode every frame uses the noise for
    key ``(seed, 0)``, so only the elapsed time changes between frames. In ``"rollout"`` mode frame ``k``
    uses key ``(seed, k, 0)``: each call is a new step from a new start.
    """
    return batched_rollout(model, normalizer, np.asarray(z0)[None], params, [run], dT, steps, mode, seed)[0]


def batched_rollout(model, normalizer: Normalizer, z0: np.ndarray, params: dict, runs, dT: float, steps: int,
                    mode: str = "rollout", seed: int = 0) -> np.ndarray:
    """``rollout`` for several runs at once: ``z0`` is (R, H, W), one start per run in ``runs``. Returns
    (R, steps + 1, H, W). Each run gets the same frames as ``rollout`` would give it alone."""
    if mode not in ("rollout", "direct"):
        raise ValueError("mode must be 'rollout' or 'direct'")
    z0 = np.asarray(z0, dtype=np.float64)
    runs = np.asarray(runs)
    frames = np.empty((len(z0), steps + 1, *z0.shape[1:]))
    frames[:, 0] = z0
    if mode == "direct":
        k = np.tile(np.arange(1, steps + 1), len(z0))
        keys = [(seed, 0)] * len(k)  # one noise field for every elapsed time, so frames agree
        z = predict_batch(model, normalizer, np.repeat(z0, steps, 0), params, np.repeat(runs, steps), k * dT, keys)
        frames[:, 1:] = z.reshape(len(z0), steps, *z0.shape[1:])
        return frames
    for k in range(1, steps + 1):
        keys = [(seed, k, 0)] * len(z0)
        frames[:, k] = predict_batch(model, normalizer, frames[:, k - 1], params, runs, dT, keys)
    return frames


def default_lag(cfg) -> int:
    """The model's training elapsed time, in snapshot steps, if it was trained on one; else 1."""
    return cfg.lags[0] if cfg.lags is not None and len(cfg.lags) == 1 else 1


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
    model, normalizer, cfg, _ = load_checkpoint(checkpoint)
    if lag is None:
        lag = default_lag(cfg)
    root = Path(data_root) if data_root is not None else ARTIFACTS_DIR / "data" / cfg.data_profile
    params, _ = load_params(root / split)
    snapshots = np.load(root / split / "snapshots.npy", mmap_mode="r")
    times = params["times"]
    picks = np.arange(0, len(times), lag)
    sim = np.asarray(snapshots[run, picks], dtype=np.float64)
    dT = float(times[lag] - times[0])
    pred = rollout(model, normalizer, sim[0], params, run, dT, len(picks) - 1, mode)

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

    out = Path(out) if out is not None else ARTIFACTS_DIR / "animations" / f"{cfg.objective}-{cfg.profile}-{split}-run{run}-{mode}.gif"
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0)
    return out
