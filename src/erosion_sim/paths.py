"""Where datasets, checkpoints, and animations live.

Everything generated goes under one root folder: ``$EROSION_ARTIFACTS`` if set, else the repo's
``artifacts/``, which is gitignored. On the cluster, the variable points at large shared storage.

- ``artifacts/data/<profile>/<split>/``: datasets from ``datagen/``.
- ``artifacts/models/<experiment>/<label>/``: one trained model per folder (``model_dir``).
- ``artifacts/results/<name>/``: one evaluation per folder (``experiments/results.py``).
"""

import os
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = Path(os.environ.get("EROSION_ARTIFACTS") or REPO_DIR / "artifacts")
DATA_DIR = ARTIFACTS_DIR / "data"
MODELS_DIR = ARTIFACTS_DIR / "models"
RESULTS_DIR = ARTIFACTS_DIR / "results"
DATAGEN_DIR = REPO_DIR / "datagen"


def data_root(profile: str, root: Path | None = None) -> Path:
    """The folder holding a dataset profile's splits: ``root`` if given (tests pass a temporary one),
    else ``artifacts/data/<profile>``."""
    return Path(root) if root is not None else DATA_DIR / profile


def model_dir(experiment: str, label: str, root: Path | None = None) -> Path:
    """The folder of model ``label`` of experiment ``experiment``: ``<root or artifacts/models>/<experiment>/<label>``.

    It holds ``config.json`` (the config), ``best.pt``, ``last.pt``, ``history.json``,
    and, once training ends, ``done.json``.
    """
    return (Path(root) if root is not None else MODELS_DIR) / experiment / label


def model_path(name: str | Path, root: Path | None = None) -> Path:
    """A model folder named in a config: absolute as given, otherwise under ``artifacts/models`` (or ``root``),
    for example ``"B2/regression"``."""
    path = Path(name)
    return path if path.is_absolute() else (Path(root) if root is not None else MODELS_DIR) / path


def is_finished(folder: Path) -> bool:
    """Whether the model in ``folder`` finished training (``done.json`` exists)."""
    return (Path(folder) / "done.json").exists()

