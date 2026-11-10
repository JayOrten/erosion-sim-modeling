"""Where datasets, checkpoints, and animations live.

Everything generated goes under ``artifacts/``, which is gitignored:

- ``artifacts/data/<profile>/<split>/``: datasets from ``datagen/``.
- ``artifacts/checkpoints/<objective>-<profile>.pt``: trained models, plus ``-last.pt`` for resuming.
- ``artifacts/animations/``: GIFs.
"""

from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = REPO_DIR / "artifacts"
DATA_DIR = ARTIFACTS_DIR / "data"
CHECKPOINTS_DIR = ARTIFACTS_DIR / "checkpoints"
ANIMATIONS_DIR = ARTIFACTS_DIR / "animations"
DATAGEN_DIR = REPO_DIR / "datagen"


def data_root(profile: str, root: Path | None = None) -> Path:
    """The folder holding a dataset profile's splits: ``root`` if given (tests pass a temporary one),
    else ``artifacts/data/<profile>``."""
    return Path(root) if root is not None else DATA_DIR / profile


def dataset_dir(profile: str, split: str) -> Path:
    return data_root(profile) / split


def checkpoint_path(profile: str, objective: str = "regression") -> Path:
    """Best checkpoint for a profile trained with an objective: ``artifacts/checkpoints/<objective>-<profile>.pt``."""
    return CHECKPOINTS_DIR / f"{objective}-{profile}.pt"


def last_checkpoint_path(best: Path) -> Path:
    """The resumable checkpoint saved next to ``best``: ``<name>-last.pt``."""
    best = Path(best)
    return best.with_name(f"{best.stem}-last{best.suffix}")
