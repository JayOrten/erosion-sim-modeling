"""Train a model into its folder (``paths.model_dir``)."""

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

from ..config import Config, config_to_json
from ..data.pairs import make_datasets as make_erosion_datasets
from ..dla.data import make_datasets as make_dla_datasets
from ..models.layers import count_parameters
from ..objectives import consistency, flow
from ..objectives.edm import VAL_SIGMAS, denoising_loss, training_loss
from ..objectives.residual import residual_rms
from ..paths import code_version, model_dir
from ..seeding import step_generator
from .checkpoints import build_model, load_checkpoint
from .run import check_stored_config


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
                  baseline_is_input: bool = False, base=None, teacher=None) -> dict:
    """Validation loss and skill over every batch of ``loader``. ``loader`` must not shuffle.

    - Regression: ``val_loss`` is the MSE of the prediction.
    - EDM: ``val_loss`` is the EDM loss. Pair ``k`` is noised at ``VAL_SIGMAS[k % 8]``, with noise
      drawn from ``seed``, so every evaluation uses the same noise. Not comparable with regression.
    - Flow matching: ``val_loss`` is the velocity MSE, pair ``k`` at time ``flow.VAL_TIMES[k % 8]``, with
      fixed noise as for EDM.
    - Consistency: ``val_loss`` is the consistency loss against ``teacher``, pair ``k`` at training level
      ``k % (SCALES - 1)``, with the student as its own target.

    With ``base`` (a residual model), the target is the residual ``y - base(x)``.

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
        if base is not None:
            y = y - base(x, cond).float()
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
            if objective == "consistency":
                n = torch.arange(pairs, pairs + len(y), device=device) % (consistency.SCALES - 1)
                noise = torch.randn(y.shape, device=device, generator=generator)
                loss = consistency.consistency_loss(model, model, teacher, x, cond, y, n, noise)
                level = consistency.training_levels(model.sigma_data, device)[n + 1]
                pred = model(x, cond, y + level.reshape(-1, 1, 1, 1) * noise, level)
                loss_sum += loss.sum().item()
            elif objective == "edm":
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


def train(cfg: Config, out: Path, data_root: Path | None = None, show: bool = False, models_root: Path | None = None,
          **overrides) -> dict:
    """Train a model for ``cfg`` in the folder ``out``.

    Writes ``config.json`` (the config) at the start, ``best.pt`` whenever the validation
    loss improves, ``last.pt`` and ``history.json`` at every evaluation, and ``done.json`` at the end.

    With ``cfg.ema_decay`` set, a second copy of the model keeps an exponential moving average (EMA) of
    the training weights. Validation, early stopping, and the saved ``"model"`` use the EMA copy; the last
    checkpoint also keeps the training weights, as ``"train_model"``, for resuming.

    If ``last.pt`` exists, training resumes from it: model, optimizer, learning-rate schedule, history,
    and position in the training order. The config must match the one in ``config.json``, except for the
    fields in ``config.NOT_TRAINING`` (``run.check_stored_config``). A resumed run gives the same result as an uninterrupted one. So if a job dies,
    submitting it again continues from the last evaluation.

    With ``cfg.residual_of``, the model samples the residual of that regression model; with ``cfg.teacher``,
    it is a consistency student distilled from that EDM model, starting from its weights. Either way it
    uses the other model's ``Scaling``. Models are found by name under ``models_root`` (default
    ``artifacts/models``).

    Stops early after ``cfg.patience`` evaluations without a better validation loss.
    Keyword overrides replace ``cfg`` fields, for example ``train(model_config("smoke"), out, steps=500)``.
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
        train_data, val_data, sigma_data = make_dla_datasets(cfg, data_root)
        scaling = None
    else:
        train_data, val_data, scaling, sigma_data = make_erosion_datasets(cfg, data_root)
    base = teacher = None
    if cfg.residual_of or cfg.teacher:
        other, other_scaling, other_cfg, _ = load_checkpoint(model_dir(cfg.residual_of or cfg.teacher, models_root),
                                                             device=str(device))
        same = ("task", "cell", "window", "context_cell", "inputs", "data_profile")
        if any(getattr(other_cfg, k) != getattr(cfg, k) for k in same):
            raise ValueError(f"{cfg.residual_of or cfg.teacher} must match this model's {', '.join(same)}")
        other.requires_grad_(False)
        if scaling is not None:  # share the other model's units
            scaling = train_data.scaling = val_data.scaling = other_scaling
        if cfg.residual_of:
            if other_cfg.objective != "regression":
                raise ValueError("residual_of must name a regression model")
            base, sigma_data = other, residual_rms(other, train_data, seed=cfg.seed)
        else:
            if other_cfg.objective != "edm":
                raise ValueError("teacher must name an EDM model")
            teacher, sigma_data = other, other.sigma_data
    rng = np.random.default_rng(cfg.seed)
    val_subset = Subset(val_data, rng.choice(len(val_data), min(cfg.val_samples, len(val_data)), replace=False))

    loader_args = dict(batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=use_cuda,
                       persistent_workers=cfg.num_workers > 0)
    model = build_model(cfg, sigma_data).to(device)
    if teacher is not None and not (Path(out) / "last.pt").exists():
        model.load_state_dict(teacher.state_dict())  # the student starts as the teacher
    ema = copy.deepcopy(model).requires_grad_(False) if cfg.ema_decay is not None else None
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, fused=use_cuda)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: _lr_factor(step, cfg))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    best_path, last_path = out / "best.pt", out / "last.pt"

    history = {"step": [], "train_loss": [], "lr": [], "eval_step": [], "val_loss": [], "val_skill": [],
               "objective": cfg.objective, "sessions": []}
    best, since_best, done = math.inf, 0, 0
    if last_path.exists():
        check_stored_config(out, cfg)
        saved = torch.load(last_path, map_location=device, weights_only=False)
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
        _write_json(out / "config.json", {"config": config_to_json(cfg), "code": code_version()})
    # One entry per training session (the first, and each resume), so a model resumed under changed code says so.
    history["sessions"].append({"start_step": done, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                                               **code_version()})

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

    print(f"{cfg.name}: {count_parameters(model):,} parameters, {len(train_data):,} training items, "
          f"{len(val_subset):,} validation items, device {device}", flush=True)
    running, seen, start, step = 0.0, 0, time.perf_counter(), done
    for step, batch in enumerate(train_loader, start=done + 1):
        x, cond, y = (batch[k].to(device, non_blocking=True) for k in ("x", "cond", "y"))
        weight = batch["weight"].to(device, non_blocking=True) if "weight" in batch else None
        if base is not None:
            with torch.no_grad():
                y = y - base(x, cond).float()
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
            if cfg.objective == "consistency":
                loss = consistency.training_loss(step_model, ema if ema is not None else model, teacher, x, cond, y,
                                                 step_generator(cfg.seed, step, device), weight)
            elif cfg.objective == "edm":
                loss = training_loss(step_model, x, cond, y, step_generator(cfg.seed, step, device), weight)
            elif cfg.objective == "flow_matching":
                loss = flow.training_loss(step_model, x, cond, y, step_generator(cfg.seed, step, device), weight)
            else:
                error = (step_model(x, cond).float() - y).pow(2)
                loss = (error if weight is None else error * weight).mean()
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
                                   baseline_is_input=cfg.task == "dla", base=base, teacher=teacher)
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
