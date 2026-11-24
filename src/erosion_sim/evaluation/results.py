"""Results folders: what an evaluation suite writes, and what notebooks and plots read.

One folder per evaluation and predictor, ``artifacts/results/<evaluation>/<predictor>/``:

- ``results.json``: scalars, nested as the suite chooses (for DLA: model, condition, metric).
- ``arrays.npz``: curves and per-sample values. Keys are paths joined with ``/``, such as
  ``"A1_base/p=0.1/fractal_dimension"``.
- ``examples.npz``: a few inputs, samples, and simulator outputs for figures, keyed the same way.
- ``manifest.json``: the evaluation's config file contents, the predictor, the models it used (each one's
  name, training config, code version, and final step and validation loss; ``model_records``), the suite's
  settings, the test data, the code version, the Slurm job ID, and the time. Written last, so it marks a
  finished result.

Notebooks read results with ``load``, which combines several predictors of one evaluation into one
``Results``. Plots read a ``Results``; they never recompute.
"""

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..paths import RESULTS_DIR, code_version, results_dir


def model_records(models: dict[str, Path]) -> dict[str, dict]:
    """What a results manifest keeps of each scored model (name -> model folder): the folder name, its full
    training config and code version (``config.json``), and how training
    ended (``done.json``: final step, best validation loss, seconds). A copy, so the manifest still says what
    was scored if the folder is later retrained or deleted."""
    records = {}
    for label, folder in models.items():
        folder = Path(folder)
        stored = json.loads((folder / "config.json").read_text())
        done = folder / "done.json"
        records[label] = {"folder": folder.name, "config": stored["config"],
                          "code": stored.get("code"),
                          "training": json.loads(done.read_text()) if done.exists() else None}
    return records


def write_results(folder: Path, scalars: dict, arrays: dict[str, np.ndarray], examples: dict[str, np.ndarray],
                  manifest: dict) -> Path:
    """Write a results folder. ``manifest`` gets the code version, the Slurm job ID, and the time added."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "manifest.json").unlink(missing_ok=True)
    (folder / "results.json").write_text(json.dumps(scalars, indent=2))
    np.savez_compressed(folder / "arrays.npz", **arrays)
    np.savez_compressed(folder / "examples.npz", **examples)
    manifest = {**manifest, "code": code_version(), "slurm_job": os.environ.get("SLURM_JOB_ID"),
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
    """One finished results folder."""
    folder = Path(folder)
    if not (folder / "manifest.json").exists():
        raise FileNotFoundError(f"{folder} has no manifest.json: not a finished results folder")
    with np.load(folder / "arrays.npz") as f:
        arrays = dict(f)
    with np.load(folder / "examples.npz") as f:
        examples = dict(f)
    return Results(folder, json.loads((folder / "results.json").read_text()), arrays, examples,
                   json.loads((folder / "manifest.json").read_text()))


def _merge(a, b):
    """``a`` with ``b``'s entries added, nested dicts merged. Where both have a value, ``a``'s is kept."""
    if not (isinstance(a, dict) and isinstance(b, dict)):
        return a
    out = dict(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if k in out else v
    return out


PER_PREDICTOR = ("predictor", "code", "slurm_job", "time")


def load(evaluation: str, *predictors: str, root: Path | None = None) -> Results:
    """The results of ``predictors`` under ``evaluation``, combined into one ``Results``; with no predictors,
    every finished one. Every predictor was scored on the same test data with the same settings, so
    shared keys (the simulator's values, the baselines) are the same in each and kept once.

    ``manifest["predictors"]`` keeps each predictor's own manifest. The top-level manifest has the merged
    ``models`` and ``entries``, and the shared ``evaluation``, ``settings``, and ``test``."""
    folder = (Path(root) if root is not None else RESULTS_DIR) / evaluation
    if not predictors:
        predictors = tuple(sorted(m.parent.name for m in folder.glob("*/manifest.json")))
        if not predictors:
            raise FileNotFoundError(f"{folder} holds no finished results")
    parts = [read_results(results_dir(evaluation, p, root)) for p in predictors]
    scalars, arrays, examples, manifest = {}, {}, {}, {}
    for part in parts:
        scalars = _merge(scalars, part.scalars)
        arrays = {**part.arrays, **arrays}
        examples = {**part.examples, **examples}
        manifest = _merge(manifest, {k: v for k, v in part.manifest.items() if k not in PER_PREDICTOR})
    manifest["predictors"] = {p: part.manifest for p, part in zip(predictors, parts)}
    return Results(folder, scalars, arrays, examples, manifest)


def list_results(root: Path | None = None) -> list[dict]:
    """One row per finished results folder: evaluation, predictor, suite, and time."""
    root = Path(root) if root is not None else RESULTS_DIR
    rows = []
    for manifest in sorted(root.glob("*/*/manifest.json")):
        m = json.loads(manifest.read_text())
        rows.append({"evaluation": manifest.parent.parent.name, "predictor": manifest.parent.name,
                     "suite": m.get("suite"), "time": m.get("time")})
    return rows
