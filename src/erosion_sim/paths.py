"""Where datasets, models, and results live.

Everything generated goes under one folder: the ``"artifacts"`` path in the repo's ``local.json`` if that
file exists, else the repo's ``artifacts/``. ``local.json`` is gitignored, so each machine has its own:

    {"artifacts": "/path/to/large/storage/erosion"}

- ``artifacts/data/<profile>/<split>/``: datasets from ``datagen/``.
- ``artifacts/models/<model>/``: one trained model per folder (``model_dir``).
- ``artifacts/results/<evaluation>/<predictor>/``: one evaluation of one predictor per folder
  (``evaluation/results.py``).

Configs for models, evaluations, and worlds are in the repo's ``configs/`` (``CONFIGS_DIR``).
"""

import json
import subprocess
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[2]


def _artifacts_dir() -> Path:
    local = REPO_DIR / "local.json"
    if local.exists():
        return Path(json.loads(local.read_text())["artifacts"]).expanduser()
    return REPO_DIR / "artifacts"


ARTIFACTS_DIR = _artifacts_dir()
DATA_DIR = ARTIFACTS_DIR / "data"
MODELS_DIR = ARTIFACTS_DIR / "models"
RESULTS_DIR = ARTIFACTS_DIR / "results"
DATAGEN_DIR = REPO_DIR / "datagen"
CONFIGS_DIR = REPO_DIR / "configs"


def data_root(profile: str, root: Path | None = None) -> Path:
    """The folder holding a dataset profile's splits: ``root`` if given (tests pass a temporary one),
    else ``artifacts/data/<profile>``."""
    return Path(root) if root is not None else DATA_DIR / profile


def model_dir(name: str, root: Path | None = None) -> Path:
    """The folder of model ``name``: ``<root or artifacts/models>/<name>``.

    It holds ``config.json`` (the config), ``best.pt``, ``last.pt``, ``history.json``, ``train.log``,
    and, once training ends, ``done.json``.
    """
    return (Path(root) if root is not None else MODELS_DIR) / name


def results_dir(evaluation: str, predictor: str, root: Path | None = None) -> Path:
    """The results of one predictor under one evaluation: ``<root or artifacts/results>/<evaluation>/<predictor>``."""
    return (Path(root) if root is not None else RESULTS_DIR) / evaluation / predictor


def is_finished(folder: Path) -> bool:
    """Whether the model in ``folder`` finished training (``done.json`` exists)."""
    return (Path(folder) / "done.json").exists()


def code_version() -> dict:
    """The repo's commit, and whether ``src/``, ``datagen/``, or ``terrain/`` had uncommitted changes.
    Recorded with every dataset, model, and result; never checked. Both values are None outside a git
    checkout."""
    def git(*args) -> str:
        return subprocess.run(["git", "-C", str(REPO_DIR), *args], capture_output=True, text=True,
                              check=True).stdout.strip()

    try:
        changes = git("status", "--porcelain", "--", "src", "datagen", "terrain")
        return {"commit": git("rev-parse", "HEAD"), "uncommitted_changes": bool(changes)}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "uncommitted_changes": None}

