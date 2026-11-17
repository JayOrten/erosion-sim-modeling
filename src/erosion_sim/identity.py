"""A model's identity: a hash of its config, and the folder it is saved in.

Two configs that differ only in fields that do not change the trained weights (``NOT_IDENTITY``) are
the same model. So when two experiments ask for the same model, the second finds it already trained.

A model folder, ``artifacts/models/<profile>-<hash>/``, holds:

- ``config.json``: the config, its hash, and the git commit it was trained at.
- ``best.pt``: the checkpoint with the best validation loss (``train.load_checkpoint``).
- ``last.pt``: the latest state, with the optimizer, for resuming.
- ``history.json``: losses so far.
- ``done.json``: written when training ends, not when it is interrupted. Marks a finished model.
"""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from .config import Config
from .paths import MODELS_DIR

# Fields that do not change the trained weights. ``profile`` is only a label.
NOT_IDENTITY = frozenset({"profile", "num_workers", "log_every", "compile"})
HASH_LENGTH = 8


def to_json(value):
    """A config value as JSON. Sets become ``{"one_of": [...]}`` (sorted), so they never collide with a
    range tuple, which becomes a list."""
    if isinstance(value, (set, frozenset)):
        return {"one_of": sorted((to_json(v) for v in value), key=lambda v: json.dumps(v, sort_keys=True))}
    if isinstance(value, (tuple, list)):
        return [to_json(v) for v in value]
    if isinstance(value, dict):
        return {str(k): to_json(v) for k, v in value.items()}
    return value


def from_json(value):
    """The inverse of ``to_json``, for the value types ``Config`` holds: lists become tuples."""
    if isinstance(value, dict) and set(value) == {"one_of"}:
        return frozenset(from_json(v) for v in value["one_of"])
    if isinstance(value, list):
        return tuple(from_json(v) for v in value)
    return value


def identity_fields(cfg: Config) -> dict:
    return {k: to_json(v) for k, v in asdict(cfg).items() if k not in NOT_IDENTITY}


def config_hash(cfg: Config) -> str:
    """The first ``HASH_LENGTH`` hex digits of the SHA-256 of the config's identity fields as canonical JSON."""
    payload = json.dumps(identity_fields(cfg), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()[:HASH_LENGTH]


def model_dir(cfg: Config, root: Path | None = None) -> Path:
    """The folder of ``cfg``'s model. An existing folder with the same hash is reused whatever its label;
    otherwise ``<root>/<profile>-<hash>``. Not created here."""
    root = Path(root) if root is not None else MODELS_DIR
    digest = config_hash(cfg)
    existing = sorted(root.glob(f"*-{digest}"))
    return existing[0] if existing else root / f"{cfg.profile}-{digest}"


def config_from_json(fields: dict) -> Config:
    """A ``Config`` from its ``to_json`` form, as stored in ``config.json`` and checkpoints."""
    return Config(**{k: from_json(v) for k, v in fields.items()})


def is_finished(folder: Path) -> bool:
    return (Path(folder) / "done.json").exists()
