"""Use a trained model to erode terrain, and compare it with the simulator."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from .data import ARTIFACTS_DIR, Normalizer, cond_values, load_params
from .hydrology import drainage_area
from .render import hillshade, render_frames
from .train import load_checkpoint


@torch.no_grad()
def predict(model, normalizer: Normalizer, z_a: np.ndarray, params: dict, run: int, dT: float) -> np.ndarray:
    """Return the model's heightmap (m) after eroding ``z_a`` for ``dT`` years with run ``run``'s parameters."""
    device = next(model.parameters()).device
    low, relief = float(z_a.min()), float(np.ptp(z_a))
    x = torch.as_tensor((z_a - low) / relief, dtype=torch.float32, device=device)[None, None]
    cond = normalizer.cond(cond_values(params, np.array([run]), np.array([dT]), relief)[0])
    cond = torch.as_tensor(cond, dtype=torch.float32, device=device)[None]
    scale = float(normalizer.target_scale(params["K_ref"][run], dT))
    y = model(x, cond).float()[0, 0].cpu().numpy()
    return z_a + y * relief * scale


def rollout(model, normalizer: Normalizer, z0: np.ndarray, params: dict, run: int, dT: float, steps: int,
            mode: str = "rollout") -> np.ndarray:
    """Model heightmaps at times 0, dT, 2 dT, ..., steps dT, shape (steps + 1, H, W).

    ``mode="rollout"`` feeds each prediction back in as the next start. ``mode="direct"`` predicts each
    time straight from ``z0``, which only works for elapsed times the model was trained on.
    """
    if mode not in ("rollout", "direct"):
        raise ValueError("mode must be 'rollout' or 'direct'")
    frames = [np.asarray(z0, dtype=np.float64)]
    for k in range(1, steps + 1):
        if mode == "rollout":
            frames.append(predict(model, normalizer, frames[-1], params, run, dT))
        else:
            frames.append(predict(model, normalizer, frames[0], params, run, k * dT))
    return np.stack(frames)


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

    out = Path(out) if out is not None else ARTIFACTS_DIR / "animations" / f"unet-{cfg.profile}-{split}-run{run}-{mode}.gif"
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0)
    return out
