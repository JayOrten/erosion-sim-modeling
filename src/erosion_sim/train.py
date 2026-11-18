"""Train a model into its folder (``paths.model_dir``), and load saved checkpoints."""

import copy
import json
import math
import time
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler, Subset

from . import data as erosion_data
from .config import NOT_TRAINING, Config, config_to_json
from . import flow
from .diffusion import VAL_SIGMAS, Denoiser, denoising_loss, step_generator, training_loss
from .dla import data as dla_data
from .model import build_backbone, count_parameters
from .normalization import Scaling
from .paths import git_state




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


def build_model(cfg: Config, sigma_data: float = 1.0) -> torch.nn.Module:
    """The backbone alone for regression, or the backbone inside an EDM denoiser or a flow matching model."""
    if cfg.backbone == "unet":
        net = dict(channels=cfg.channels, blocks=cfg.blocks, emb_dim=cfg.emb_dim, attention=cfg.attention)
    else:
        net = dict(patch=cfg.patch, depth=cfg.dit_depth, width=cfg.dit_width, heads=cfg.dit_heads)
    common = dict(sigma_data=sigma_data, fixed_edges=cfg.fixed_edges, in_channels=cfg.in_channels,
                  backbone=cfg.backbone, **net)
    if cfg.objective == "edm":
        return Denoiser(cfg.cond_dim, **common)
    if cfg.objective == "flow_matching":
        return flow.FlowModel(cfg.cond_dim, **common)
    return build_backbone(cfg.backbone, cfg.cond_dim, cfg.in_channels, cfg.fixed_edges, **net)


@torch.no_grad()
def update_ema(ema: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    """Move each weight of ``ema`` a fraction ``1 - decay`` of the way toward ``model``'s. Buffers are copied."""
    for e, p in zip(ema.parameters(), model.parameters()):
        e.lerp_(p, 1 - decay)
    for e, b in zip(ema.buffers(), model.buffers()):
        e.copy_(b)


def _lr_factor(step: int, cfg: Config) -> float:
    """Linear warmup, then cosine decay to zero."""
    if step < cfg.warmup_steps:
        return (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


@torch.no_grad()
def evaluate_loss(model, loader, device, bf16, objective: str = "regression", seed: int = 0,
                  baseline_is_input: bool = False) -> dict:
    """Validation loss and skill over every batch of ``loader``. ``loader`` must not shuffle.

    - Regression: ``val_loss`` is the MSE of the prediction.
    - EDM: ``val_loss`` is the EDM loss. Pair ``k`` is noised at ``VAL_SIGMAS[k % 8]``, with noise
      drawn from ``seed``, so every evaluation uses the same noise. Not comparable with regression.
    - Flow matching: ``val_loss`` is the velocity MSE, pair ``k`` at time ``flow.VAL_TIMES[k % 8]``, with
      fixed noise as for EDM.

    ``val_skill`` is ``1 - MSE / MSE(no change)``. No change is a zero target for erosion (the target is a
    change), and the input itself for DLA (``baseline_is_input``; the target is the later cluster). For EDM
    and flow matching it scores the one-step estimate at those noise levels or times, not a full sample.
    """
    model.eval()
    sq_model = sq_zero = count = 0.0
    loss_sum, pairs = 0.0, 0
    generator = torch.Generator(device=device).manual_seed(seed)
    for batch in loader:
        x, cond, y = (batch[k].to(device, non_blocking=True) for k in ("x", "cond", "y"))
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
            if objective == "edm":
                levels = torch.tensor(VAL_SIGMAS, device=device)[torch.arange(pairs, pairs + len(y), device=device) % len(VAL_SIGMAS)]
                noise = torch.randn(y.shape, device=device, generator=generator)
                loss, pred = denoising_loss(model, x, cond, y, levels * model.sigma_data, noise)
                loss_sum += loss.sum().item()
            elif objective == "flow_matching":
                times = torch.tensor(flow.VAL_TIMES, device=device)[torch.arange(pairs, pairs + len(y), device=device) % len(flow.VAL_TIMES)]
                noise = torch.randn(y.shape, device=device, generator=generator)
                loss, pred = flow.flow_loss(model, x, cond, y, times, noise)
                loss_sum += loss.sum().item()
            else:
                pred = model(x, cond)
        sq_model += (pred.float() - y).pow(2).sum().item()
        sq_zero += (y - x[:, :1] if baseline_is_input else y).pow(2).sum().item()
        count += y.numel()
        pairs += len(y)
    model.train()
    mse, mse_zero = sq_model / count, sq_zero / count
    val_loss = mse if objective == "regression" else loss_sum / pairs
    return {"val_loss": val_loss, "val_skill": 1 - mse / mse_zero}


def _write_json(path: Path, value) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, default=str))
    tmp.replace(path)  # atomic, so an interruption never leaves a half-written file


def train(cfg: Config, out: Path, data_root: Path | None = None, show: bool = False, **overrides) -> dict:
    """Train a model for ``cfg`` in the folder ``out``.

    Writes ``config.json`` (the config and git commit) at the start, ``best.pt`` whenever the validation
    loss improves, ``last.pt`` and ``history.json`` at every evaluation, and ``done.json`` at the end.

    With ``cfg.ema_decay`` set, a second copy of the model keeps an exponential moving average (EMA) of
    the training weights. Validation, early stopping, and the saved ``"model"`` use the EMA copy; the last
    checkpoint also keeps the training weights, as ``"train_model"``, for resuming.

    If ``last.pt`` exists, training resumes from it: model, optimizer, learning-rate schedule, history,
    and position in the training order. The config must match the saved one, except for the fields in
    ``config.NOT_TRAINING``. A resumed run gives the same result as an uninterrupted one. So if a job dies,
    submitting it again continues from the last evaluation.

    Stops early after ``cfg.patience`` evaluations without a better validation loss.
    Keyword overrides replace ``cfg`` fields, for example ``train(PROFILES["smoke"], steps=500)``.
    Returns the training history, plus the model folder under ``"folder"``.
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

    if cfg.task == "dla":
        train_data, val_data, sigma_data = dla_data.make_datasets(cfg, data_root)
        scaling = None
    else:
        train_data, val_data, scaling, sigma_data = erosion_data.make_datasets(cfg, data_root)
    rng = np.random.default_rng(cfg.seed)
    if cfg.max_train_samples is not None:
        train_data = Subset(train_data, rng.choice(len(train_data), cfg.max_train_samples, replace=False))
    val_subset = Subset(val_data, rng.choice(len(val_data), min(cfg.val_samples, len(val_data)), replace=False))
    if cfg.epochs is not None:
        cfg = replace(cfg, steps=math.ceil(cfg.epochs * len(train_data) / cfg.batch_size))

    loader_args = dict(batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=use_cuda,
                       persistent_workers=cfg.num_workers > 0)
    model = build_model(cfg, sigma_data).to(device)
    ema = copy.deepcopy(model).requires_grad_(False) if cfg.ema_decay is not None else None
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, fused=use_cuda)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: _lr_factor(step, cfg))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    best_path, last_path = out / "best.pt", out / "last.pt"

    history = {"step": [], "train_loss": [], "lr": [], "eval_step": [], "val_loss": [], "val_skill": [],
               "objective": cfg.objective}
    best, since_best, done = math.inf, 0, 0
    if last_path.exists():
        saved = torch.load(last_path, map_location=device, weights_only=False)
        saved_cfg = asdict(Config(**saved["config"]))  # fills fields added since the save with defaults
        changed = {k for k, v in asdict(cfg).items() if saved_cfg[k] != v} - NOT_TRAINING
        if changed:
            raise ValueError(f"config differs from {last_path} in {sorted(changed)}; cannot resume")
        model.load_state_dict(saved["train_model"] if ema is not None else saved["model"])
        if ema is not None:
            ema.load_state_dict(saved["model"])
        opt.load_state_dict(saved["optimizer"])
        sched.load_state_dict(saved["scheduler"])
        history, done = saved["history"], saved["step"]
        best = min(history["val_loss"], default=math.inf)
        since_best = len(history["val_loss"]) - 1 - int(np.argmin(history["val_loss"])) if history["val_loss"] else 0
        print(f"resuming from {last_path} at step {done}", flush=True)
    else:
        _write_json(out / "config.json", {"config": config_to_json(cfg), **git_state()})

    order = training_order(len(train_data), cfg.steps * cfg.batch_size, cfg.seed)[done * cfg.batch_size:]
    train_loader = DataLoader(train_data, sampler=_OrderSampler(order), drop_last=True, **loader_args)
    val_loader = DataLoader(val_subset, shuffle=False, **loader_args)
    step_model = torch.compile(model) if cfg.compile and use_cuda else model
    eval_model = ema if ema is not None else step_model

    def state(step: int) -> dict:
        return {"model": (ema if ema is not None else model).state_dict(), "config": asdict(cfg),
                "scaling": asdict(scaling) if scaling is not None else None, "sigma_data": sigma_data,
                "step": step, "history": history}

    def save_last(step: int) -> None:
        resume_state = {"optimizer": opt.state_dict(), "scheduler": sched.state_dict()}
        if ema is not None:
            resume_state["train_model"] = model.state_dict()
        tmp = last_path.with_suffix(".tmp")
        torch.save({**state(step), **resume_state}, tmp)
        tmp.replace(last_path)
        _write_json(out / "history.json", history)

    print(f"{cfg.profile} ({out.name}): {count_parameters(model):,} parameters, {len(train_data):,} training items, "
          f"{len(val_subset):,} validation items, device {device}", flush=True)
    running, seen, start, step = 0.0, 0, time.perf_counter(), done
    for step, batch in enumerate(train_loader, start=done + 1):
        x, cond, y = (batch[k].to(device, non_blocking=True) for k in ("x", "cond", "y"))
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
            if cfg.objective == "edm":
                loss = training_loss(step_model, x, cond, y, step_generator(cfg.seed, step, device))
            elif cfg.objective == "flow_matching":
                loss = flow.training_loss(step_model, x, cond, y, step_generator(cfg.seed, step, device))
            else:
                loss = (step_model(x, cond).float() - y).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if ema is not None:
            update_ema(ema, model, cfg.ema_decay)
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
            result = evaluate_loss(eval_model, val_loader, device, bf16, cfg.objective, cfg.seed,
                                   baseline_is_input=cfg.task == "dla")
            history["eval_step"].append(step)
            history["val_loss"].append(result["val_loss"])
            history["val_skill"].append(result["val_skill"])
            improved = result["val_loss"] < best
            if improved:
                best, since_best = result["val_loss"], 0
                torch.save(state(step), best_path)
            else:
                since_best += 1
            save_last(step)
            print(f"  validation loss {result['val_loss']:.4f}  skill vs no change {result['val_skill']:.3f}"
                  f"{'  (saved best)' if improved else ''}", flush=True)
            if cfg.patience is not None and since_best >= cfg.patience:
                print(f"stopping early: no better validation loss in {since_best} evaluations", flush=True)
                break

    seconds = time.perf_counter() - start
    history["seconds"] = history.get("seconds", 0.0) + seconds
    _write_json(out / "history.json", history)
    _write_json(out / "done.json", {"step": step, "best_val_loss": best, "seconds": history["seconds"]})
    if show:
        plot_history(history)
    return {**history, "folder": out}


def plot_history(history: dict) -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4), layout="constrained")
    a.semilogy(history["step"], history["train_loss"], label="train")
    a.semilogy(history["eval_step"], history["val_loss"], "o-", label="validation")
    objective = history.get("objective")
    a.set(xlabel="step", title="loss",
          ylabel={"edm": "EDM loss", "flow_matching": "velocity MSE"}.get(objective, "MSE of the target"))
    a.legend()
    b.plot(history["eval_step"], history["val_skill"], "o-")
    b.axhline(0, color="gray", lw=0.8)
    b.set(xlabel="step", ylabel="1 - MSE / MSE(no change)",
          title="validation skill" + (" (one-step estimate)" if objective != "regression" else ""), ylim=(min(-0.1, *history["val_skill"]), 1))
    plt.show()


def load_checkpoint(path: Path, device: str | None = None) -> tuple[torch.nn.Module, Scaling | None, Config, dict]:
    """Return (model in eval mode, scaling, config, full checkpoint) from a model folder (its ``best.pt``)
    or a checkpoint file. ``scaling`` is None for DLA models."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    path = Path(path)
    ckpt = torch.load(path / "best.pt" if path.is_dir() else path, map_location=device, weights_only=False)
    cfg = Config(**ckpt["config"])
    scaling = Scaling.from_dict(ckpt["scaling"]) if ckpt.get("scaling") is not None else None
    model = build_model(cfg, ckpt["sigma_data"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, scaling, cfg, ckpt
