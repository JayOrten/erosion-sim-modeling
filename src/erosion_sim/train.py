"""Train the U-Net on snapshot pairs."""

import math
import time
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler, Subset

from .config import Config
from .data import ARTIFACTS_DIR, COND_NAMES, ErosionPairs, Normalizer, select_runs
from .model import UNet, count_parameters


def checkpoint_path(cfg: Config) -> Path:
    return ARTIFACTS_DIR / f"unet-{cfg.profile}.pt"


def last_checkpoint_path(best: Path) -> Path:
    """Where the latest state goes, next to the best checkpoint: unet-<profile>-last.pt."""
    return best.with_name(f"{best.stem}-last{best.suffix}")


# Config fields that may differ when resuming: they do not change what is trained.
_RESUMABLE_CHANGES = {"num_workers", "compile", "log_every", "patience"}


class _OrderSampler(Sampler):
    """Yields the indices of a fixed array in order."""

    def __init__(self, order: np.ndarray):
        self.order = order

    def __iter__(self):
        return iter(self.order.tolist())

    def __len__(self) -> int:
        return len(self.order)


def training_order(size: int, count: int, seed: int) -> np.ndarray:
    """The first ``count`` training indices: shuffled passes over ``size`` pairs, without replacement.

    Fixed by ``seed``, so a resumed run continues the same order.
    """
    rng = np.random.default_rng(seed)
    passes = [rng.permutation(size) for _ in range(math.ceil(count / size))]
    return np.concatenate(passes)[:count]


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


def train(cfg: Config, data_root: Path | None = None, show: bool = True, out: Path | None = None, resume: bool = False,
          **overrides) -> dict:
    """Train a model for ``cfg`` and save the checkpoint with the best validation loss.

    At every evaluation, the latest state (with the optimizer) is also saved to ``unet-<profile>-last.pt``.
    Stops early after ``cfg.patience`` evaluations without a better validation loss.
    With ``resume=True``, continues from ``unet-<profile>-last.pt``: model, optimizer, learning-rate
    schedule, history, and position in the training order. The config must match the saved one, except
    for the fields in ``_RESUMABLE_CHANGES``.
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
    if cfg.epochs is not None:
        cfg = replace(cfg, steps=math.ceil(cfg.epochs * len(train_data) / cfg.batch_size))

    loader_args = dict(batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=use_cuda,
                       persistent_workers=cfg.num_workers > 0)
    model = build_model(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, fused=use_cuda)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: _lr_factor(step, cfg))
    out = Path(out) if out is not None else checkpoint_path(cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    last = last_checkpoint_path(out)

    history = {"step": [], "train_loss": [], "lr": [], "eval_step": [], "val_loss": [], "val_skill": []}
    best, since_best, done = math.inf, 0, 0
    if resume:
        saved = torch.load(last, map_location=device, weights_only=False)
        changed = {k for k, v in asdict(cfg).items() if saved["config"].get(k) != v} - _RESUMABLE_CHANGES
        if changed:
            raise ValueError(f"config differs from {last.name} in {sorted(changed)}; cannot resume")
        model.load_state_dict(saved["model"])
        opt.load_state_dict(saved["optimizer"])
        sched.load_state_dict(saved["scheduler"])
        history, done = saved["history"], saved["step"]
        best = min(history["val_loss"])
        since_best = len(history["val_loss"]) - 1 - int(np.argmin(history["val_loss"]))
        print(f"resuming from {last.name} at step {done}", flush=True)

    order = training_order(len(train_data), cfg.steps * cfg.batch_size, cfg.seed)[done * cfg.batch_size:]
    train_loader = DataLoader(train_data, sampler=_OrderSampler(order), drop_last=True, **loader_args)
    val_loader = DataLoader(val_subset, shuffle=False, **loader_args)
    step_model = torch.compile(model) if cfg.compile and use_cuda else model

    print(f"{cfg.profile}: {count_parameters(model):,} parameters, {len(train_data):,} training pairs, "
          f"{len(val_subset):,} validation pairs, device {device}", flush=True)
    running, seen, start = 0.0, 0, time.perf_counter()
    for step, batch in enumerate(train_loader, start=done + 1):
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
            rate = (step - done) * cfg.batch_size / (time.perf_counter() - start)
            print(f"step {step}/{cfg.steps}  train loss {running / seen:.4f}  lr {history['lr'][-1]:.2e}  "
                  f"{rate:,.0f} samples/s", flush=True)
            running, seen = 0.0, 0
        if step % cfg.eval_every == 0 or step == cfg.steps:
            result = evaluate_loss(step_model, val_loader, device, bf16)
            history["eval_step"].append(step)
            history["val_loss"].append(result["val_loss"])
            history["val_skill"].append(result["val_skill"])
            saved = result["val_loss"] < best
            state = {"model": model.state_dict(), "config": asdict(cfg), "normalizer": normalizer.to_dict(),
                     "step": step, "history": history}
            if saved:
                best, since_best = result["val_loss"], 0
                torch.save(state, out)
            else:
                since_best += 1
            torch.save({**state, "optimizer": opt.state_dict(), "scheduler": sched.state_dict()}, last)
            print(f"  validation loss {result['val_loss']:.4f}  skill vs no change {result['val_skill']:.3f}"
                  f"{'  (saved best)' if saved else ''}", flush=True)
            if cfg.patience is not None and since_best >= cfg.patience:
                print(f"stopping early: no better validation loss in {since_best} evaluations", flush=True)
                break

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

