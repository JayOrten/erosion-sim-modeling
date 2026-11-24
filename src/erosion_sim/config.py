"""Training settings: one frozen ``Config`` per model, read from ``configs/models/<name>.yaml``.

A model file lists every field below except ``name``, which is the file name: each file is the complete
record of its model, and does not depend on the defaults here.
A trained model's folder holds its full config as ``config.json`` (``config_to_json``), so the settings it
was trained with can always be read back (``config_from_json``).
"""

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml

from .paths import CONFIGS_DIR

TASKS = ("erosion", "dla")
BACKBONES = ("unet", "dit", "fno")
OBJECTIVES = ("regression", "edm", "flow_matching", "consistency")
DLA_CONDITIONING = ("p", "x")  # normalization.dla_conditioning


@dataclass(frozen=True)
class Config:
    """Settings for one model, trained into ``artifacts/models/<name>/`` (``paths.model_dir``)."""

    name: str  # the model's name: its file name in configs/models/
    data_profile: str  # dataset under artifacts/data/, such as "erosion_master/main_256" or "dla_master"
    # "erosion": snapshot pairs from datagen/ (data.ErosionPairs). "dla": cluster pairs (dla.data.DLAPairs).
    task: str = "erosion"
    backbone: str = "unet"
    # "regression": predict the target directly with MSE. "edm": sample it with an EDM denoiser
    # (objectives/edm.py). "flow_matching": sample it with a flow matching model (objectives/flow.py).
    # "consistency": a student distilled from the EDM model at `teacher` (objectives/consistency.py).
    objective: str = "regression"
    # Models this one is built on, by name, such as "erosion_regression". `residual_of`: a regression model whose residual this generative model samples
    # (residual.py, experiment B8). `teacher`: the EDM model a consistency student is distilled from.
    residual_of: str | None = None
    teacher: str | None = None
    # Which runs to use: column -> condition; see table.select. Applied to train and val. Empty uses every
    # run. Example: {"fractal": "fbm", "index": {"range": [0, 2667]}}.
    data: dict = field(default_factory=dict)
    # Train on only the first `max_runs` of the selected runs, in table order (nested subsets, experiment
    # B4). Validation keeps every selected run.
    max_runs: int | None = None
    # Edge cells change by zero, as in the simulator. False for world levels, whose window edges are not
    # boundaries, and for DLA.
    fixed_edges: bool = True
    # Erosion items (data.ErosionPairs). Track B: whole runs at 1 m cells (window None). Track C: a world
    # level sees windows of `window` cells, each a `cell` × `cell` block of the 1 m runs, at least `margin`
    # 1 m cells from the run's edge. `context_cell` is the cell size of the level above, whose true change
    # is an extra input channel; None for the top level. `context_noise` adds noise to that channel in
    # training (experiment C4).
    cell: int = 1
    window: int | None = None
    context_cell: int | None = None
    margin: int = 0
    context_noise: float = 0.0
    # Training items per pair (erosion windows) or per run (DLA), each with its own random draw. `augment`
    # flips and rotates erosion items at random.
    repeats: int = 1
    augment: bool = False
    # Extra erosion input channels, from data.INPUTS ("map", "drainage"), and the drainage-weighted loss
    # exponent (experiment B7; 0 is the plain loss).
    inputs: tuple[str, ...] = ()
    drainage_weight: float = 0.0
    # DLA only: the range of the start step t, both ends included (dla.data.DLAPairs). None for erosion.
    t_range: tuple[int, int] | None = None
    # Model. Each backbone reads only its own settings: U-Net (model.py), DiT (dit.py), FNO (fno.py).
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
    # FNO (fno.py, the optional arm of B3): 4 layers of width 64 keeping 32 × 32 Fourier modes.
    fno_modes: int = 32
    fno_width: int = 64
    fno_depth: int = 4
    # Optimization
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 1e-4
    steps: int = 20_000
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
        if self.task == "dla" and self.fixed_edges:
            raise ValueError("DLA models have no fixed edges")
        if (self.task == "dla") != (self.t_range is not None):
            raise ValueError("t_range is set for DLA models, and only for them")
        if self.window is not None and self.fixed_edges:
            raise ValueError("window items are crops whose edges are not boundaries: set fixed_edges=False")
        if self.objective not in OBJECTIVES:
            raise ValueError(f"objective must be one of {OBJECTIVES}, got {self.objective!r}")
        if self.residual_of is not None and self.objective not in ("edm", "flow_matching"):
            raise ValueError("a residual model samples the residual: use edm or flow_matching")
        if (self.objective == "consistency") != (self.teacher is not None):
            raise ValueError("a consistency student needs a teacher, and only a consistency student takes one")
        if self.context_cell is not None and self.context_cell % self.cell:
            raise ValueError("context_cell must be a multiple of cell")

    @property
    def cond_dim(self) -> int:
        """Length of the conditioning vector."""
        from .data.normalization import PARAMETER_NAMES

        return len(DLA_CONDITIONING) if self.task == "dla" else len(PARAMETER_NAMES)

    @property
    def in_channels(self) -> int:
        """Input channels besides the noisy target: the start terrain (or DLA cluster), the extra inputs,
        and the context channel of a level with a parent."""
        return 1 + len(self.inputs) + (self.context_cell is not None)


def check_fields(path: Path, values: dict, required: set[str]) -> None:
    """Raise ``ValueError`` unless the config file at ``path`` has exactly the fields ``required``."""
    missing, unknown = sorted(required - set(values)), sorted(set(values) - required)
    if missing or unknown:
        raise ValueError(f"{path}: " + "; ".join(
            text for text in (f"missing fields {', '.join(missing)}" if missing else "",
                              f"unknown fields {', '.join(unknown)}" if unknown else "") if text))


def data_names(root: Path | None = None) -> list[str]:
    """Every dataset defined in ``configs/data/``, by path without ``.yaml``: ``dla_master``,
    ``erosion_master/main_256``. The same path under ``artifacts/data/`` holds its splits."""
    folder = (Path(root) if root is not None else CONFIGS_DIR) / "data"
    return sorted(str(p.relative_to(folder).with_suffix("")) for p in folder.rglob("*.yaml"))


def check_data_profile(name: str, root: Path | None = None) -> None:
    """Raise ``KeyError`` unless ``name`` is a dataset in ``configs/data/``."""
    if name not in data_names(root):
        raise KeyError(f"no dataset {name!r} in configs/data/; datasets: {', '.join(data_names(root))}")


def config_names(kind: str, root: Path | None = None) -> list[str]:
    """Every name defined in ``configs/<kind>/`` (``models``, ``evaluations``, or ``worlds``). ``root``
    replaces ``configs/``."""
    return sorted(f.stem for f in ((Path(root) if root is not None else CONFIGS_DIR) / kind).glob("*.yaml"))


def model_config(name: str, root: Path | None = None) -> Config:
    """The config in ``configs/models/<name>.yaml`` (``root`` replaces ``configs/``). The file must list
    every field but ``name``. Lists become tuples, as in ``config_from_json``."""
    path = (Path(root) if root is not None else CONFIGS_DIR) / "models" / f"{name}.yaml"
    if not path.exists():
        raise KeyError(f"no model {name!r}: {path} does not exist")
    values = yaml.safe_load(path.read_text()) or {}
    check_fields(path, values, {f.name for f in fields(Config)} - {"name"})
    return Config(name=name, **{k: from_json_value(v) for k, v in values.items()})


# Fields that do not change the trained weights. A stored config may differ from a requested one in these.
NOT_TRAINING = frozenset({"name", "num_workers", "log_every", "compile", "patience"})


def to_json_value(value):
    """Tuples become lists, at any depth."""
    if isinstance(value, (tuple, list)):
        return [to_json_value(v) for v in value]
    if isinstance(value, dict):
        return {k: to_json_value(v) for k, v in value.items()}
    return value


def from_json_value(value):
    """Lists become tuples, at any depth, as the dataclasses use them."""
    if isinstance(value, list):
        return tuple(from_json_value(v) for v in value)
    if isinstance(value, dict):
        return {k: from_json_value(v) for k, v in value.items()}
    return value


def config_to_json(cfg: Config) -> dict:
    """``cfg`` as plain JSON values."""
    return {k: to_json_value(v) for k, v in asdict(cfg).items()}


def config_from_json(values: dict) -> Config:
    """The inverse of ``config_to_json``."""
    return Config(**{k: from_json_value(v) for k, v in values.items()})


def training_differences(stored: Config, requested: Config) -> list[str]:
    """Names of the fields, other than ``NOT_TRAINING``, in which two configs differ."""
    a, b = asdict(stored), asdict(requested)
    return sorted(k for k in a if a[k] != b[k] and k not in NOT_TRAINING)

