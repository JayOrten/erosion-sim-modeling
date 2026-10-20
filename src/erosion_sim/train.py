"""Train the U-Net on snapshot pairs."""

import math
import time
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, Subset

from .config import Config
from .data import ARTIFACTS_DIR, COND_NAMES, ErosionPairs, Normalizer, select_runs
from .model import UNet, count_parameters


def checkpoint_path(cfg: Config) -> Path:
    return ARTIFACTS_DIR / f"unet-{cfg.profile}.pt"


def make_datasets(cfg: Config, data_root: Path | None = None) -> tuple[ErosionPairs, ErosionPairs, Normalizer]:
    """Training and validation pairs for ``cfg``, with a normalizer fitted on the training runs."""
    root = Path(data_root) if data_root is not None else ARTIFACTS_DIR / "data" / cfg.data_profile
    datasets = []
    for split in ("train", "val"):
        params = dict(np.load(root / split / "params.npz"))
        datasets.append((root / split, select_runs(params, **dict(cfg.runs))))
    (train_path, train_runs), (val_path, val_runs) = datasets
    normalizer = Normalizer.fit(train_path, scale=cfg.scale, runs=train_runs)
    train = ErosionPairs(train_path, normalizer, runs=train_runs, lags=cfg.lags)
    val = ErosionPairs(val_path, normalizer, runs=val_runs, lags=cfg.lags)
    return train, val, normalizer


def build_model(cfg: Config) -> UNet:
    return UNet(len(COND_NAMES), channels=cfg.channels, blocks=cfg.blocks, emb_dim=cfg.emb_dim, attention=cfg.attention)


def _lr_factor(step: int, cfg: Config) -> float:
    """Linear warmup, then cosine decay to zero."""
    if step < cfg.warmup_steps:
        return (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


@torch.no_grad()
def evaluate_loss(model, loader, device, bf16) -> dict:
    """Mean squared error of the model and of predicting no change, over every batch of ``loader``."""
    model.eval()
    sq_model = sq_zero = count = 0.0
    for batch in loader:
        x, cond, y = (batch[k].to(device, non_blocking=True) for k in ("x", "cond", "y"))
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
            pred = model(x, cond)
        sq_model += (pred.float() - y).pow(2).sum().item()
        sq_zero += y.pow(2).sum().item()
        count += y.numel()
    model.train()
    mse, mse_zero = sq_model / count, sq_zero / count
    return {"val_loss": mse, "val_skill": 1 - mse / mse_zero}


def train(cfg: Config, data_root: Path | None = None, show: bool = True, out: Path | None = None, **overrides) -> dict:
    """Train a model for ``cfg`` and save the checkpoint with the best validation loss.

    Keyword overrides replace ``cfg`` fields, for example ``train(PROFILES["fixed_dt"], steps=500)``.
    Returns the training history, plus the checkpoint path under ``"checkpoint"``.
    """
    cfg = replace(cfg, **overrides)
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    bf16 = cfg.bf16 and use_cuda
    if use_cuda:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    train_data, val_data, normalizer = make_datasets(cfg, data_root)
    rng = np.random.default_rng(cfg.seed)
    if cfg.max_train_samples is not None:
        train_data = Subset(train_data, rng.choice(len(train_data), cfg.max_train_samples, replace=False))
    val_subset = Subset(val_data, rng.choice(len(val_data), min(cfg.val_samples, len(val_data)), replace=False))

    loader_args = dict(batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=use_cuda,
                       persistent_workers=cfg.num_workers > 0)
    sampler = RandomSampler(train_data, replacement=True, num_samples=cfg.steps * cfg.batch_size)
    train_loader = DataLoader(train_data, sampler=sampler, drop_last=True, **loader_args)
    val_loader = DataLoader(val_subset, shuffle=False, **loader_args)

    model = build_model(cfg).to(device)
    step_model = torch.compile(model) if cfg.compile and use_cuda else model
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, fused=use_cuda)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: _lr_factor(step, cfg))
    out = Path(out) if out is not None else checkpoint_path(cfg)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"{cfg.profile}: {count_parameters(model):,} parameters, {len(train_data):,} training pairs, "
          f"{len(val_subset):,} validation pairs, device {device}", flush=True)
    history = {"step": [], "train_loss": [], "lr": [], "eval_step": [], "val_loss": [], "val_skill": []}
    best = math.inf
    running, seen, start = 0.0, 0, time.perf_counter()
    for step, batch in enumerate(train_loader, start=1):
        x, cond, y = (batch[k].to(device, non_blocking=True) for k in ("x", "cond", "y"))
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
            pred = step_model(x, cond)
        loss = (pred.float() - y).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        running += loss.item()
        seen += 1

        if step % cfg.log_every == 0 or step == cfg.steps:
            history["step"].append(step)
            history["train_loss"].append(running / seen)
            history["lr"].append(sched.get_last_lr()[0])
            rate = step * cfg.batch_size / (time.perf_counter() - start)
            print(f"step {step}/{cfg.steps}  train loss {running / seen:.4f}  lr {history['lr'][-1]:.2e}  "
                  f"{rate:,.0f} samples/s", flush=True)
            running, seen = 0.0, 0
        if step % cfg.eval_every == 0 or step == cfg.steps:
            result = evaluate_loss(step_model, val_loader, device, bf16)
            history["eval_step"].append(step)
            history["val_loss"].append(result["val_loss"])
            history["val_skill"].append(result["val_skill"])
            saved = result["val_loss"] < best
            if saved:
                best = result["val_loss"]
                torch.save({"model": model.state_dict(), "config": asdict(cfg), "normalizer": normalizer.to_dict(),
                            "step": step, "history": history}, out)
            print(f"  validation loss {result['val_loss']:.4f}  skill vs no change {result['val_skill']:.3f}"
                  f"{'  (saved)' if saved else ''}", flush=True)

    history["checkpoint"] = out
    history["seconds"] = time.perf_counter() - start
    if show:
        plot_history(history)
    return history


def plot_history(history: dict) -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4), layout="constrained")
    a.semilogy(history["step"], history["train_loss"], label="train")
    a.semilogy(history["eval_step"], history["val_loss"], "o-", label="validation")
    a.set(xlabel="step", ylabel="MSE of normalized change", title="loss")
    a.legend()
    b.plot(history["eval_step"], history["val_skill"], "o-")
    b.axhline(0, color="gray", lw=0.8)
    b.set(xlabel="step", ylabel="1 - MSE / MSE(no change)", title="validation skill", ylim=(min(-0.1, *history["val_skill"]), 1))
    plt.show()


def load_checkpoint(path: Path, device: str | None = None) -> tuple[UNet, Normalizer, Config, dict]:
    """Return (model in eval mode, normalizer, config, full checkpoint) from a saved checkpoint."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = Config(**ckpt["config"])
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    norm = ckpt["normalizer"]
    normalizer = Normalizer(tuple(norm["cond_low"]), tuple(norm["cond_high"]), norm["scale"], norm["coef"], norm["power"])
    return model, normalizer, cfg, ckpt


@torch.no_grad()
def show_predictions(path: Path, count: int = 3, split: str = "val", data_root: Path | None = None, seed: int = 0) -> None:
    """Plot the model's prediction next to the simulator for ``count`` random pairs of ``split``.

    Columns: start, simulator end, model end, simulator change, model change, error. Changes and error
    share one color scale per row.
    """
    from matplotlib.colors import LightSource

    from .data import denormalize

    model, normalizer, cfg, _ = load_checkpoint(path)
    device = next(model.parameters()).device
    root = Path(data_root) if data_root is not None else ARTIFACTS_DIR / "data" / cfg.data_profile
    params = dict(np.load(root / split / "params.npz"))
    data = ErosionPairs(root / split, normalizer, runs=select_runs(params, **dict(cfg.runs)), lags=cfg.lags)
    light = LightSource(azdeg=315, altdeg=45)

    def shade(z, low, high):
        return light.shade(z, cmap=plt.cm.terrain, blend_mode="soft", vmin=low, vmax=high)

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
            (shade(z_a, low, high), f"run {item['run']}: start, t={data.times[item['start']]:,.0f} yr", {}),
            (shade(z_b, low, high), f"simulator, ΔT={dT:,.0f} yr", {}),
            (shade(z_pred, low, high), "model", {}),
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
