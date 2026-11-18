"""Training settings: one frozen ``Config`` per model, plus a few named profiles.

A trained model's folder holds its config as ``config.json`` (``config_to_json``), so the settings it was
trained with can always be read back (``config_from_json``).
"""

import json
from dataclasses import asdict, dataclass, fields, replace

TASKS = ("erosion", "dla")
BACKBONES = ("unet", "dit")
OBJECTIVES = ("regression", "edm", "flow_matching")
LEVELS = (None, "coarse", "fine")
DLA_CONDITIONING = ("p", "x")  # normalization.dla_conditioning


@dataclass(frozen=True)
class Config:
    """Settings for one training run. A registered experiment's model is saved in
    ``artifacts/models/<experiment>/<label>/`` (``paths.model_dir``)."""

    profile: str  # a readable name
    data_profile: str  # dataset under artifacts/data/
    # "erosion": snapshot pairs from datagen/ (data.ErosionPairs). "dla": cluster pairs (dla.data.DLAPairs).
    task: str = "erosion"
    backbone: str = "unet"
    # "regression": predict the target directly with MSE. "edm": sample it with an EDM denoiser
    # (diffusion.py). "flow_matching": sample it with a flow matching model (flow.py).
    objective: str = "regression"
    # Which runs to use, as (column, condition) pairs; see table.select. Applied to train and val. Empty
    # uses every run. Example: (("fractal", "fbm"), ("index", (0, 2667))).
    data: tuple[tuple[str, object], ...] = ()
    # Elapsed times to train on, in snapshot steps (1 = 1,000 yr in the full dataset). None uses all.
    lags: tuple[int, ...] | None = None
    # Edge cells change by zero, as in the simulator. False for world models, whose window edges are not
    # boundaries.
    fixed_edges: bool = True
    # World level (docs/plans/infinite-plan.md, step 5). None trains on whole runs (ErosionPairs).
    # "coarse": whole runs as block x block means (world.data.ErosionCoarse). "fine": crop x crop pieces
    # with the true coarse change as a second input channel (world.data.ErosionCrops).
    level: str | None = None
    crop: int = 64
    block: int = 4
    # Coarse level only: crop the block means to this many coarse cells per side. None uses whole runs.
    coarse_crop: int | None = None
    # World levels: keep crops this many fine cells away from the run's edge, where the simulator holds
    # heights fixed and drains water (infinite-plan step 9a).
    margin: int = 0
    # Training items per run (DLA) or per (run, start, end) pair (world levels), each with its own random
    # draw: t and x for DLA, crop and flip for world levels.
    repeats: int = 1
    # DLA: range of the start step t, both ends included. None uses 1 to 90% of the particle count.
    t_range: tuple[int, int] | None = None
    max_train_samples: int | None = None  # train on only this many fixed pairs (for the overfit test)
    # Model. U-Net settings apply when backbone = "unet", DiT settings when backbone = "dit" (dit.py).
    channels: tuple[int, ...] = (32, 64, 128, 256)
    blocks: int = 1
    emb_dim: int = 256
    attention: bool = True
    # DiT S: 8 blocks of width 240, about 8.6 M parameters, near the U-Net's 8.4 M. Patch 8 gives 1,024
    # tokens at 256²; DLA uses patch 4 at 128².
    patch: int = 8
    dit_depth: int = 8
    dit_width: int = 240
    dit_heads: int = 4
    # Optimization
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 1e-4
    steps: int = 20_000
    epochs: float | None = None  # if set, replaces steps: this many passes over the training pairs
    warmup_steps: int = 500
    # Exponential moving average of the weights, used for validation and saved as the model. Each step
    # moves the average 1 - ema_decay of the way to the training weights, so it covers about
    # 1 / (1 - ema_decay) steps. None turns it off.
    ema_decay: float | None = 0.999
    # Validation
    eval_every: int = 1_000
    val_samples: int = 2_048  # fixed validation pairs, drawn once
    patience: int | None = None  # stop after this many evaluations without a better validation loss
    log_every: int = 100
    # Hardware
    num_workers: int = 8
    compile: bool = True  # torch.compile; ignored on CPU
    bf16: bool = True  # 16-bit autocast; ignored on CPU
    seed: int = 0

    def __post_init__(self):
        for name, allowed in (("task", TASKS), ("backbone", BACKBONES)):
            if getattr(self, name) not in allowed:
                raise ValueError(f"{name} must be one of {allowed}, got {getattr(self, name)!r}")
        if self.task == "dla" and (self.level is not None or self.fixed_edges):
            raise ValueError("DLA models have no world level and no fixed edges")
        if self.objective not in OBJECTIVES:
            raise ValueError(f"objective must be one of {OBJECTIVES}, got {self.objective!r}")
        if self.level not in LEVELS:
            raise ValueError(f"level must be one of {LEVELS}, got {self.level!r}")
        if self.margin % self.block:
            raise ValueError("margin must be a multiple of block")

    @property
    def cond_dim(self) -> int:
        """Length of the conditioning vector."""
        from .normalization import PARAMETER_NAMES

        return len(DLA_CONDITIONING) if self.task == "dla" else len(PARAMETER_NAMES)

    @property
    def in_channels(self) -> int:
        """Input channels besides the noisy target: the start heightmap, plus the coarse change at the fine level."""
        return 2 if self.level == "fine" else 1


PROFILES = {
    # Seconds on CPU, for tests. Point data_root at a small dataset.
    "smoke": Config(
        profile="smoke",
        data_profile="smoke",
        channels=(8, 16),
        emb_dim=16,
        batch_size=4,
        steps=6,
        warmup_steps=2,
        ema_decay=0.5,
        eval_every=3,
        val_samples=8,
        log_every=2,
        num_workers=0,
        compile=False,
    ),
    # The DLA version, on erosion_sim.dla's dla_smoke dataset.
    "smoke_dla": Config(
        profile="smoke_dla",
        data_profile="dla_smoke",
        task="dla",
        objective="edm",
        fixed_edges=False,
        repeats=20,
        channels=(8, 16),
        emb_dim=16,
        batch_size=4,
        steps=6,
        warmup_steps=2,
        ema_decay=0.5,
        eval_every=3,
        val_samples=8,
        log_every=2,
        num_workers=0,
        compile=False,
    ),
}


def get_config(profile: str, objective: str | None = None) -> Config:
    """The settings for ``profile``, with its objective replaced by ``objective`` if given."""
    cfg = PROFILES[profile]
    return cfg if objective is None else replace(cfg, objective=objective)


# Fields that do not change the trained weights. A stored config may differ from a requested one in these.
NOT_TRAINING = frozenset({"profile", "num_workers", "log_every", "compile", "patience"})


def _to_json(value):
    if isinstance(value, (set, frozenset)):  # tagged, so a set never reads back as a range tuple
        return {"one_of": sorted((_to_json(v) for v in value), key=lambda v: json.dumps(v, sort_keys=True))}
    if isinstance(value, (tuple, list)):
        return [_to_json(v) for v in value]
    return value


def _from_json(value):
    if isinstance(value, dict) and set(value) == {"one_of"}:
        return frozenset(_from_json(v) for v in value["one_of"])
    if isinstance(value, list):
        return tuple(_from_json(v) for v in value)
    return value


def config_to_json(cfg: Config) -> dict:
    """``cfg`` as plain JSON values. Tuples become lists; sets become ``{"one_of": [...]}``."""
    return {k: _to_json(v) for k, v in asdict(cfg).items()}


def config_from_json(values: dict) -> Config:
    """The inverse of ``config_to_json``. Fields missing from ``values`` (added since it was written) take
    their defaults; fields no longer in ``Config`` are dropped."""
    known = {f.name for f in fields(Config)}
    return Config(**{k: _from_json(v) for k, v in values.items() if k in known})


def training_differences(stored: Config, requested: Config) -> list[str]:
    """Names of the fields, other than ``NOT_TRAINING``, in which two configs differ."""
    a, b = asdict(stored), asdict(requested)
    return sorted(k for k in a if a[k] != b[k] and k not in NOT_TRAINING)

