"""Where datasets, checkpoints, and animations live.

Everything generated goes under one root folder: ``$EROSION_ARTIFACTS`` if set, else the repo's
``artifacts/``, which is gitignored. On the cluster, the variable points at large shared storage.

- ``artifacts/data/<profile>/<split>/``: datasets from ``datagen/``.
- ``artifacts/models/<experiment>/<label>/``: one trained model per folder (``model_dir``).
- ``artifacts/results/<name>/``: one evaluation per folder (``results.py``).
- ``artifacts/checkpoints/<objective>-<profile>.pt``: models trained before that, still read by ``world/``.
- ``artifacts/animations/``: GIFs.
"""

import os
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = Path(os.environ.get("EROSION_ARTIFACTS") or REPO_DIR / "artifacts")
DATA_DIR = ARTIFACTS_DIR / "data"
CHECKPOINTS_DIR = ARTIFACTS_DIR / "checkpoints"  # models trained before models/ existed; used by world/
MODELS_DIR = ARTIFACTS_DIR / "models"
RESULTS_DIR = ARTIFACTS_DIR / "results"
ANIMATIONS_DIR = ARTIFACTS_DIR / "animations"
DATAGEN_DIR = REPO_DIR / "datagen"


def data_root(profile: str, root: Path | None = None) -> Path:
    """The folder holding a dataset profile's splits: ``root`` if given (tests pass a temporary one),
    else ``artifacts/data/<profile>``."""
    return Path(root) if root is not None else DATA_DIR / profile


def dataset_dir(profile: str, split: str) -> Path:
    return data_root(profile) / split


def model_dir(experiment: str, label: str, root: Path | None = None) -> Path:
    """The folder of model ``label`` of experiment ``experiment``: ``<root or artifacts/models>/<experiment>/<label>``.

    It holds ``config.json`` (the config and the git commit), ``best.pt``, ``last.pt``, ``history.json``,
    and, once training ends, ``done.json``.
    """
    return (Path(root) if root is not None else MODELS_DIR) / experiment / label


def is_finished(folder: Path) -> bool:
    """Whether the model in ``folder`` finished training (``done.json`` exists)."""
    return (Path(folder) / "done.json").exists()


def checkpoint_path(profile: str, objective: str = "regression") -> Path:
    """Best checkpoint for a profile trained with an objective: ``artifacts/checkpoints/<objective>-<profile>.pt``."""
    return CHECKPOINTS_DIR / f"{objective}-{profile}.pt"


def git_state() -> dict:
    """The repo's current commit, and whether ``src/`` has uncommitted changes. Recorded with every model
    and result. ``commit`` is None outside a git checkout."""
    import subprocess

    def git(*args) -> str:
        return subprocess.run(["git", "-C", str(REPO_DIR), *args], capture_output=True, text=True, check=True).stdout

    try:
        return {"commit": git("rev-parse", "HEAD").strip(), "dirty": bool(git("status", "--porcelain", "--", "src").strip())}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
