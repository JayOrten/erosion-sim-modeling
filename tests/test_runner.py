import subprocess

import numpy as np
import pytest

import erosion_sim.runner as runner
from erosion_sim import experiments
from erosion_sim.identity import config_hash, model_dir
from erosion_sim.results import read_results


def test_registry_entries_are_consistent():
    for name, exp in experiments.EXPERIMENTS.items():
        assert exp.name == name and exp.suite in experiments.SUITES and exp.models
    # A5 and A6 reuse A1's model: same config, so the same folder.
    a1 = experiments.get("A1").models["base"]
    assert config_hash(experiments.get("A5").models["best"]) == config_hash(a1)
    assert config_hash(experiments.get("A6").models["best"]) == config_hash(a1)
    with pytest.raises(KeyError):
        experiments.get("nope")


def test_smoke_experiments_train_and_evaluate(dla_root, tmp_path):
    models = tmp_path / "models"
    folder = runner.train_model("smoke_A", "base", data_root=dla_root, models_root=models)
    assert runner.train_model("smoke_A", "base", data_root=dla_root, models_root=models) == folder  # reused
    with pytest.raises(KeyError):
        runner.train_model("smoke_A", "nope", data_root=dla_root, models_root=models)

    out = runner.evaluate("smoke_A", data_root=dla_root, models_root=models, out=tmp_path / "r1",
                          allow_dirty=True, device="cpu")
    assert read_results(out).manifest["suite"] == "dla"

    out = runner.evaluate("smoke_A_agreement", data_root=dla_root, models_root=models, out=tmp_path / "r2",
                          allow_dirty=True, device="cpu")
    results = read_results(out)
    jumps = results.manifest["jumps"]
    assert jumps == [2, 5, 10, 20, 40]  # fractions of the 40 particles after the 20-particle start
    rows = results.scalars["base"]["containment"]
    assert set(rows) == {"shared", "fresh"} and len(rows["shared"]) == len(jumps) * (len(jumps) - 1) // 2
    assert all(0 <= v <= 1 for v in rows["shared"].values())
    assert set(results.scalars["base"]["chain"]) == {"1x40", "2x20", "4x10"}
    assert results.examples["base/containment/shared"].shape[0] == len(jumps)


def test_evaluate_needs_trained_models(dla_root, tmp_path):
    with pytest.raises(RuntimeError, match="not trained"):
        runner.evaluate("smoke_A", data_root=dla_root, models_root=tmp_path / "empty", out=tmp_path / "r")


def test_missing_fraction():
    from erosion_sim.dla.suite import missing_fraction

    small = np.zeros((2, 4, 4), np.uint8)
    small[:, 0, :2] = 1
    large = small.copy()
    large[1, 0, 1] = 0
    np.testing.assert_allclose(missing_fraction(small, large), [0.0, 0.5])


class FakeSlurm:
    """Stands in for subprocess.run: records sbatch commands, answers squeue from ``queued``."""

    def __init__(self, queued=None):
        self.calls, self.queued, self.next = [], queued or {}, 100

    def __call__(self, command, **kwargs):
        self.calls.append(command)
        if command[0] == "squeue":
            name = next(a for a in command if a.startswith("--name=")).split("=", 1)[1]
            return subprocess.CompletedProcess(command, 0, stdout=self.queued.get(name, ""))
        self.next += 1
        return subprocess.CompletedProcess(command, 0, stdout=f"{self.next};cluster\n")


@pytest.fixture
def clean_repo(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "git_state", lambda: {"commit": "abc123", "dirty": False})
    monkeypatch.setattr(runner, "results_dir", lambda name: tmp_path / "results" / name)
    monkeypatch.setattr(runner, "model_dir", lambda cfg: model_dir(cfg, tmp_path / "models"))
    return tmp_path


def test_submit_trains_then_evaluates_after_training(clean_repo):
    slurm = FakeSlurm()
    commands = runner.submit("smoke_A", run=slurm)
    train, evaluate = commands
    assert train[:2] == ["sbatch", "--parsable"] and train[-3:] == ["slurm/train.sbatch", "smoke_A", "base"]
    assert "--export=ALL,EROSION_COMMIT=abc123" in train
    cfg = experiments.get("smoke_A").models["base"]
    assert f"--job-name=train-{config_hash(cfg)}" in train
    assert evaluate[-2:] == ["slurm/evaluate.sbatch", "smoke_A"]
    assert "--dependency=afterok:101" in evaluate


def test_submit_waits_on_a_model_already_queued(clean_repo):
    cfg = experiments.get("smoke_A").models["base"]
    slurm = FakeSlurm(queued={f"train-{config_hash(cfg)}": "77\n"})
    commands = runner.submit("smoke_A_agreement", run=slurm)
    assert len(commands) == 1 and "--dependency=afterok:77" in commands[0]


def test_submit_skips_trained_models_and_existing_results(clean_repo):
    cfg = experiments.get("smoke_A").models["base"]
    folder = model_dir(cfg, clean_repo / "models")
    folder.mkdir(parents=True)
    (folder / "done.json").write_text("{}")
    commands = runner.submit("smoke_A", run=FakeSlurm())
    assert len(commands) == 1 and not any(a.startswith("--dependency") for a in commands[0])
    (clean_repo / "results" / "smoke_A" / "manifest.json").write_text("{}")
    assert runner.submit("smoke_A", run=FakeSlurm()) == []
    assert len(runner.submit("smoke_A", run=FakeSlurm(), force_evaluate=True)) == 1


def test_submit_refuses_a_dirty_repo_but_dry_run_prints(clean_repo, monkeypatch):
    monkeypatch.setattr(runner, "git_state", lambda: {"commit": "abc123", "dirty": True})
    with pytest.raises(RuntimeError, match="uncommitted"):
        runner.submit("smoke_A", run=FakeSlurm())
    slurm = FakeSlurm()
    assert len(runner.submit("smoke_A", dry_run=True, run=slurm)) == 2 and slurm.calls == []
