"""Train a model by name: ``configs/models/<name>.yaml`` into ``artifacts/models/<name>/``."""

import json
from pathlib import Path

from ..config import (
    Config,
    check_data_profile,
    config_from_json,
    model_config,
    training_differences,
)
from ..logs import logged
from ..paths import is_finished, model_dir


class ConfigChanged(RuntimeError):
    """A stored model was trained with settings its config file no longer gives."""


def check_stored_config(folder: Path, cfg: Config) -> None:
    """Raise ``ConfigChanged`` if ``folder`` holds a model whose ``config.json`` differs from ``cfg`` in a
    field that changes training. A folder without ``config.json`` passes."""
    path = Path(folder) / "config.json"
    if not path.exists():
        return
    changed = training_differences(config_from_json(json.loads(path.read_text())["config"]), cfg)
    if changed:
        raise ConfigChanged(f"{folder} was trained with different {', '.join(changed)} than "
                            f"configs/models/{cfg.name}.yaml now gives. Undo the edit and put the change in a "
                            "new model file, or delete the folder to train it again.")


def trained_model(name: str, models_root: Path | None = None, configs_root: Path | None = None) -> Path:
    """The folder of model ``name``, after checking it finished training with its current config."""
    folder = model_dir(name, models_root)
    check_stored_config(folder, model_config(name, configs_root))
    if not is_finished(folder):
        raise RuntimeError(f"model {name} is not trained yet: run `erosion-sim train {name}`")
    return folder


def train_model(name: str, data_root: Path | None = None, models_root: Path | None = None,
                configs_root: Path | None = None) -> Path:
    """Train model ``name``, resuming if interrupted. Does nothing if it is finished. Output is also appended
    to ``train.log`` in its folder. Returns the folder."""
    from .train import train

    cfg = model_config(name, configs_root)
    check_data_profile(cfg.data_profile, configs_root)
    folder = model_dir(name, models_root)
    check_stored_config(folder, cfg)
    if is_finished(folder):
        print(f"{folder} is already trained", flush=True)
        return folder
    with logged(folder / "train.log"):
        train(cfg, data_root=data_root, out=folder, models_root=models_root)
    return folder
