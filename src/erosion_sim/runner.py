"""Run registry entries (``experiments.py``): train their models, evaluate them, or submit both to Slurm.

- ``train_model(name, label)``: one model of an experiment, in its model folder.
- ``evaluate(name)``: the experiment's suite on its trained models, into ``artifacts/results/<name>/``.
- ``submit(name)``: one Slurm training job per model that is neither finished nor already queued, then
  one evaluation job that waits for them (``--dependency=afterok``).
"""

import subprocess
from pathlib import Path

from . import experiments
from .identity import config_hash, is_finished, model_dir
from .paths import REPO_DIR, git_state
from .results import results_dir

TRAIN_SCRIPT = "slurm/train.sbatch"
EVALUATE_SCRIPT = "slurm/evaluate.sbatch"


def train_model(name: str, label: str, data_root: Path | None = None, models_root: Path | None = None) -> Path:
    """Train model ``label`` of experiment ``name``, resuming if interrupted. Returns its folder."""
    from .train import train

    exp = experiments.get(name)
    if label not in exp.models:
        raise KeyError(f"{name} has no model {label!r}; models: {', '.join(exp.models)}")
    cfg = exp.models[label]
    folder = model_dir(cfg, models_root)
    if is_finished(folder):
        print(f"{folder} is already trained", flush=True)
        return folder
    train(cfg, data_root=data_root, out=folder)
    return folder


def evaluate(name: str, data_root: Path | None = None, models_root: Path | None = None, out: Path | None = None,
             allow_dirty: bool = False, device: str | None = None) -> Path:
    """Run experiment ``name``'s suite on its models and write the results folder."""
    from .dla.suite import evaluate_agreement, evaluate_dla

    exp = experiments.get(name)
    folders = {label: model_dir(cfg, models_root) for label, cfg in exp.models.items()}
    missing = [label for label, folder in folders.items() if not is_finished(folder)]
    if missing:
        raise RuntimeError(f"{name}: models not trained yet: {', '.join(missing)}")
    out = Path(out) if out is not None else results_dir(name)
    args = dict(data_root=data_root, allow_dirty=allow_dirty, device=device)
    if exp.settings is not None:
        args["settings"] = exp.settings
    suite = {"dla": evaluate_dla, "dla_agreement": evaluate_agreement}[exp.suite]
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

    - Finished models are skipped. A model already queued or running (its job name is ``train-<hash>``,
      from any experiment) is waited on, not trained twice.
    - Each job gets ``EROSION_COMMIT``, so it refuses to run if the repo moves before it starts.
    - Logs go into the model folder and the results folder: ``slurm-<job>.out``.
    - Evaluation is skipped if the results already exist, unless ``force_evaluate``.
    - Refuses if ``src/`` has uncommitted changes, except with ``dry_run``, which only prints.

    Returns the ``sbatch`` commands, in order.
    """
    exp = experiments.get(name)
    state = git_state()
    if state["dirty"] and not dry_run:
        raise RuntimeError("src/ has uncommitted changes; commit before submitting")
    export = f"--export=ALL,EROSION_COMMIT={state['commit']}"
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

    for label, cfg in exp.models.items():
        folder = model_dir(cfg)
        if is_finished(folder):
            print(f"{label}: trained ({folder.name})", flush=True)
            continue
        job_name = f"train-{config_hash(cfg)}"
        queued = None if dry_run else queued_job(job_name, run)
        if queued:
            print(f"{label}: already queued as job {queued}; waiting on it", flush=True)
            waits.append(queued)
            continue
        if not dry_run:
            folder.mkdir(parents=True, exist_ok=True)
        waits.append(sbatch([f"--job-name={job_name}", f"--output={folder}/slurm-%j.out", export,
                             TRAIN_SCRIPT, name, label]))

    out = results_dir(name)
    if (out / "manifest.json").exists() and not force_evaluate:
        print(f"{name}: results exist ({out}); pass force_evaluate to redo them", flush=True)
        return commands
    if not dry_run:
        out.mkdir(parents=True, exist_ok=True)
    dependency = [f"--dependency=afterok:{':'.join(waits)}"] if waits else []
    sbatch([f"--job-name=evaluate-{name}", f"--output={out}/slurm-%j.out", export, *dependency,
            EVALUATE_SCRIPT, name])
    return commands
