"""Run registry entries (``registry.py``): train their models, evaluate them, or submit both to Slurm.

- ``train_model(name, label)``: one model of an experiment, in its model folder.
- ``evaluate(name)``: the experiment's suite on its trained models, into ``artifacts/results/<name>/``.
- ``submit(name)``: one Slurm training job per model that is neither finished nor already queued, then
  one evaluation job that waits for them (``--dependency=afterok``).

A model borrowed from another experiment (``registry.Ref``) lives in, and is trained into, that
experiment's folder. Before using a stored model, its ``config.json`` is checked against the registry: if
the entry was edited after training, nothing runs until the old folder is removed or the edit undone.
"""

import json
import subprocess
from pathlib import Path

from ..config import Config, config_from_json, training_differences
from ..paths import REPO_DIR, is_finished, model_dir
from . import registry
from .results import results_dir

TRAIN_SCRIPT = "slurm/train.sbatch"
EVALUATE_SCRIPT = "slurm/evaluate.sbatch"


class ConfigChanged(RuntimeError):
    """A stored model was trained with settings the registry no longer asks for."""


def check_stored_config(folder: Path, cfg: Config) -> None:
    """Raise ``ConfigChanged`` if ``folder`` holds a model whose ``config.json`` differs from ``cfg`` in a
    field that changes training. A folder without ``config.json`` passes."""
    path = Path(folder) / "config.json"
    if not path.exists():
        return
    changed = training_differences(config_from_json(json.loads(path.read_text())["config"]), cfg)
    if changed:
        raise ConfigChanged(f"{folder} was trained with different {', '.join(changed)} than the registry now "
                            "gives. Undo the edit, or delete the folder to train it again.")


def model_folder(name: str, label: str, models_root: Path | None = None) -> tuple[Path, Config, str, str]:
    """Model ``label`` of experiment ``name``: its folder, config, and owning experiment and label, after
    checking any stored config against the registry."""
    owner, owner_label, cfg = registry.resolve(name, label)
    folder = model_dir(owner, owner_label, models_root)
    check_stored_config(folder, cfg)
    return folder, cfg, owner, owner_label


def train_model(name: str, label: str, data_root: Path | None = None, models_root: Path | None = None) -> Path:
    """Train model ``label`` of experiment ``name`` (in its owner's folder), resuming if interrupted.
    Returns its folder."""
    from ..training.train import train

    folder, cfg, _, _ = model_folder(name, label, models_root)
    if is_finished(folder):
        print(f"{folder} is already trained", flush=True)
        return folder
    train(cfg, data_root=data_root, out=folder)
    return folder


def evaluate(name: str, data_root: Path | None = None, models_root: Path | None = None, out: Path | None = None,
             device: str | None = None) -> Path:
    """Run experiment ``name``'s suite on its models and write the results folder."""
    from ..dla.suite import evaluate_agreement, evaluate_dla
    from ..erosion.suite import evaluate_erosion
    from ..world.suite import evaluate_world

    exp = registry.get(name)
    folders = {label: model_folder(name, label, models_root)[0] for label in exp.models}
    missing = [label for label, folder in folders.items() if not is_finished(folder)]
    if missing:
        raise RuntimeError(f"{name}: models not trained yet: {', '.join(missing)}")
    out = Path(out) if out is not None else results_dir(name)
    args = dict(data_root=data_root, device=device)
    if exp.settings is not None:
        args["settings"] = exp.settings
    suite = {"dla": evaluate_dla, "dla_agreement": evaluate_agreement, "erosion": evaluate_erosion,
             "world": evaluate_world}[exp.suite]
    return suite(folders, out, **args)


def _job_id(sbatch_output: str) -> str:
    return sbatch_output.strip().split(";")[0]  # --parsable prints "id" or "id;cluster"


def queued_job(job_name: str, run=subprocess.run) -> str | None:
    """The ID of one of your queued or running jobs with this name, or None."""
    try:
        out = run(["squeue", "--me", "--noheader", f"--name={job_name}", "--format=%i"], capture_output=True,
                  text=True, check=True).stdout.split()
    except (OSError, subprocess.CalledProcessError):
        return None
    return out[0] if out else None


def submit(name: str, dry_run: bool = False, force_evaluate: bool = False, run=subprocess.run) -> list[list[str]]:
    """Submit experiment ``name``: training jobs for missing models, then its evaluation job.

    - Finished models are skipped. A model already queued or running (job name ``train-<owner>-<label>``,
      submitted by any experiment) is waited on, not trained twice. A model borrowed from another
      experiment that is not trained yet is trained into that experiment's folder.
    - Logs go into the model folder and the results folder: ``slurm-<job>.out``.
    - Evaluation is skipped if the results already exist, unless ``force_evaluate``.
    - ``dry_run`` only prints.

    Returns the ``sbatch`` commands, in order.
    """
    exp = registry.get(name)
    commands, waits = [], []

    def sbatch(args: list[str]) -> str:
        command = ["sbatch", "--parsable", *args]
        commands.append(command)
        print(" ".join(command), flush=True)
        if dry_run:
            return f"<job {len(commands)}>"
        job = _job_id(run(command, capture_output=True, text=True, check=True, cwd=REPO_DIR).stdout)
        print(f"  -> job {job}", flush=True)
        return job

    for label in exp.models:
        folder, _, owner, owner_label = model_folder(name, label)
        if is_finished(folder):
            print(f"{label}: trained ({owner}/{owner_label})", flush=True)
            continue
        job_name = f"train-{owner}-{owner_label}"
        queued = None if dry_run else queued_job(job_name, run)
        if queued:
            print(f"{label}: already queued as job {queued}; waiting on it", flush=True)
            waits.append(queued)
            continue
        if not dry_run:
            folder.mkdir(parents=True, exist_ok=True)
        waits.append(sbatch([f"--job-name={job_name}", f"--output={folder}/slurm-%j.out",
                             TRAIN_SCRIPT, owner, owner_label]))

    out = results_dir(name)
    if (out / "manifest.json").exists() and not force_evaluate:
        print(f"{name}: results exist ({out}); pass force_evaluate to redo them", flush=True)
        return commands
    if not dry_run:
        out.mkdir(parents=True, exist_ok=True)
    dependency = [f"--dependency=afterok:{':'.join(waits)}"] if waits else []
    sbatch([f"--job-name=evaluate-{name}", f"--output={out}/slurm-%j.out", *dependency,
            EVALUATE_SCRIPT, name])
    return commands
