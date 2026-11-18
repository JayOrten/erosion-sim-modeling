"""Results folders: what an evaluation suite writes, and what notebooks and plots read.

One folder per experiment, ``artifacts/results/<name>/``:

- ``results.json``: scalars, nested as the suite chooses (for DLA: model, condition, metric).
- ``arrays.npz``: curves and per-sample values. Keys are paths joined with ``/``, such as
  ``"A1_base/p=0.1/fractal_dimension"``.
- ``examples.npz``: a few inputs, samples, and simulator outputs for figures, keyed the same way.
- ``manifest.json``: the models (folder names and hashes), the suite's settings, the test data, the git
  commit, the Slurm job ID, and the time. Written last, so it marks a finished result.

Plots read a folder with ``read_results``; they never recompute.
"""

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .paths import RESULTS_DIR, git_state


def results_dir(name: str) -> Path:
    return RESULTS_DIR / name


class DirtyRepo(RuntimeError):
    """Results would not match any commit."""


def write_results(folder: Path, scalars: dict, arrays: dict[str, np.ndarray], examples: dict[str, np.ndarray],
                  manifest: dict, allow_dirty: bool = False) -> Path:
    """Write a results folder. ``manifest`` gets the git state, the Slurm job ID, and the time added.

    Refuses if ``src/`` has uncommitted changes, unless ``allow_dirty`` (for tests and trial runs; the
    manifest then records ``"dirty": true``).
    """
    state = git_state()
    if state["dirty"] and not allow_dirty:
        raise DirtyRepo("src/ has uncommitted changes, so these results would not match any commit. "
                        "Commit first, or pass allow_dirty for a trial run.")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "manifest.json").unlink(missing_ok=True)
    (folder / "results.json").write_text(json.dumps(scalars, indent=2))
    np.savez_compressed(folder / "arrays.npz", **arrays)
    np.savez_compressed(folder / "examples.npz", **examples)
    manifest = {**manifest, **state, "slurm_job": os.environ.get("SLURM_JOB_ID"),
                "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return folder


@dataclass
class Results:
    folder: Path
    scalars: dict
    arrays: dict[str, np.ndarray]
    examples: dict[str, np.ndarray]
    manifest: dict


def read_results(folder: Path | str) -> Results:
    """A finished results folder, or a name under ``artifacts/results/``."""
    folder = Path(folder)
    if not folder.is_absolute() and not folder.exists():
        folder = results_dir(str(folder))
    if not (folder / "manifest.json").exists():
        raise FileNotFoundError(f"{folder} has no manifest.json: not a finished results folder")
    with np.load(folder / "arrays.npz") as f:
        arrays = dict(f)
    with np.load(folder / "examples.npz") as f:
        examples = dict(f)
    return Results(folder, json.loads((folder / "results.json").read_text()), arrays, examples,
                   json.loads((folder / "manifest.json").read_text()))


def list_results(root: Path | None = None) -> list[dict]:
    """One row per finished results folder: name, suite, models, commit, whether it was dirty, and time."""
    root = Path(root) if root is not None else RESULTS_DIR
    rows = []
    for manifest in sorted(root.glob("*/manifest.json")):
        m = json.loads(manifest.read_text())
        rows.append({"name": manifest.parent.name, "suite": m.get("suite"),
                     "models": ", ".join(f"{k}={v['folder']}" for k, v in m.get("models", {}).items()),
                     "commit": (m.get("commit") or "")[:8], "dirty": m.get("dirty"), "time": m.get("time")})
    return rows
