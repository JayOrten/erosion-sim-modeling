import json
import shutil

import numpy as np
import pytest
import yaml
from conftest import CONFIGS

from erosion_sim.config import (
    check_data_profile,
    config_names,
    data_names,
    model_config,
)
from erosion_sim.evaluation.evaluate import (
    EvaluationChanged,
    _settings,
    evaluate,
    evaluation_config,
    predictor_config,
)
from erosion_sim.evaluation.results import list_results, load, read_results
from erosion_sim.paths import CONFIGS_DIR
from erosion_sim.training.run import ConfigChanged, train_model


@pytest.fixture(scope="module")
def all_configs(tmp_path_factory):
    """The repo's configs/ and the tests' own configs, in one folder: tests use both."""
    root = tmp_path_factory.mktemp("configs")
    shutil.copytree(CONFIGS_DIR, root, dirs_exist_ok=True)
    shutil.copytree(CONFIGS, root, dirs_exist_ok=True)
    return root


@pytest.mark.parametrize("which", ["repo", "tests"])
def test_every_config_file_loads(which, all_configs):
    # Each folder's files load, and every name they refer to exists (tests' files may use the repo's).
    own = CONFIGS_DIR if which == "repo" else CONFIGS
    refs = CONFIGS_DIR if which == "repo" else all_configs
    for name in config_names("models", own):
        cfg = model_config(name, own)
        assert cfg.name == name
        check_data_profile(cfg.data_profile, refs)
        for other in (cfg.residual_of, cfg.teacher):
            assert other is None or other in config_names("models", refs), (name, other)
    for name in config_names("evaluations", own):
        values = evaluation_config(name, own)
        check_data_profile(values["data_profile"], refs)
        _settings(values["suite"], {k: v for k, v in values.items() if k != "suite"})
    assert not set(config_names("predictors", refs)) & set(config_names("models", refs))
    for name in config_names("predictors", own):
        spec = predictor_config(name, refs)
        for model in spec.get("levels", []) + ([spec["model"]] if "model" in spec else []):
            assert model in config_names("models", refs), (name, model)
    from erosion_sim.dla.config import data_config

    for name in data_names(own):
        if name.startswith("dla"):
            assert data_config(name, own).profile == name
    with pytest.raises(KeyError):
        check_data_profile("main_256", refs)


def test_every_cascade_is_a_valid_stack_that_fits_its_runs(all_configs):
    from erosion_sim.world.world import check_stack

    def run_size(profile: str) -> int:
        group = profile.split("/")[1]
        return 32 if group == "smoke" else int(group.rsplit("_", 1)[1])

    for name in config_names("predictors", all_configs):
        spec = predictor_config(name, all_configs)
        if spec["kind"] != "cascade":
            continue
        configs = [model_config(level, all_configs) for level in spec["levels"]]
        check_stack(configs)
        for cfg in configs:
            assert cfg.window * cfg.cell + 2 * cfg.margin <= run_size(cfg.data_profile), (name, cfg.name)
            assert not cfg.fixed_edges


def test_config_files_must_list_every_field(roots):
    configs = roots["configs_root"]
    path = configs / "models" / "smoke.yaml"
    values = yaml.safe_load(path.read_text())
    path.write_text(yaml.safe_dump({**values, "learning_rate": 0.1}))
    with pytest.raises(ValueError, match="unknown fields learning_rate"):
        model_config("smoke", configs)
    del values["lr"]
    path.write_text(yaml.safe_dump(values))
    with pytest.raises(ValueError, match="missing fields lr"):
        model_config("smoke", configs)
    with pytest.raises(KeyError):
        model_config("nope", configs)
    path = configs / "evaluations" / "smoke_dla.yaml"
    values = yaml.safe_load(path.read_text())
    del values["seed"]
    path.write_text(yaml.safe_dump(values))
    with pytest.raises(ValueError, match="missing fields seed"):
        evaluation_config("smoke_dla", configs)
    path = configs / "predictors" / "smoke_cascade.yaml"
    path.write_text(yaml.safe_dump({"kind": "cascade", "levels": ["smoke_coarse", "smoke_fine"]}))
    with pytest.raises(ValueError, match="missing fields splits, steps"):
        predictor_config("smoke_cascade", configs)
    path = configs / "predictors" / "smoke_model_chunks.yaml"
    path.write_text(yaml.safe_dump({"kind": "baseline", "method": "model_chunks", "chunk": 16}))
    with pytest.raises(ValueError, match="missing fields model, steps"):
        predictor_config("smoke_model_chunks", configs)


@pytest.fixture
def roots(tmp_path):
    """Copies of configs/ and the tests' configs in one folder, and empty models and results folders, so
    tests can edit config files."""
    configs = tmp_path / "configs"
    shutil.copytree(CONFIGS_DIR, configs)
    shutil.copytree(CONFIGS, configs, dirs_exist_ok=True)
    return {"configs_root": configs, "models_root": tmp_path / "models", "results_root": tmp_path / "results"}


def _edit(configs, kind: str, name: str, **changes):
    path = configs / kind / f"{name}.yaml"
    path.write_text(yaml.safe_dump({**yaml.safe_load(path.read_text()), **changes}))


def test_train_and_evaluate_by_name(dla_root, roots):
    model_args = {k: roots[k] for k in ("models_root", "configs_root")}
    folder = train_model("smoke_dla", data_root=dla_root, **model_args)
    assert folder == roots["models_root"] / "smoke_dla"
    assert train_model("smoke_dla", data_root=dla_root, **model_args) == folder  # finished: reused
    with pytest.raises(KeyError):
        train_model("nope", data_root=dla_root, **model_args)

    out = evaluate("smoke_dla", "smoke_dla", data_root=dla_root, device="cpu", **roots)
    assert out == roots["results_root"] / "smoke_dla" / "smoke_dla"
    manifest = read_results(out).manifest
    assert manifest["suite"] == "dla" and manifest["evaluation"]["data_profile"] == "dla_smoke"
    assert manifest["predictor"] == {"name": "smoke_dla", "kind": "model"}
    # The manifest keeps a copy of what was scored, and the code version.
    record = manifest["models"]["smoke_dla"]
    assert record["folder"] == "smoke_dla" and record["config"]["name"] == "smoke_dla"
    assert record["training"]["step"] > 0 and "commit" in record["code"] and "commit" in manifest["code"]
    # Logs, and a training session per run.
    assert "step" in (folder / "train.log").read_text() and (out / "evaluate.log").exists()
    assert [s["start_step"] for s in json.loads((folder / "history.json").read_text())["sessions"]] == [0]
    # Finished results are not replaced by accident.
    with pytest.raises(FileExistsError):
        evaluate("smoke_dla", "smoke_dla", data_root=dla_root, device="cpu", **roots)
    evaluate("smoke_dla", "smoke_dla", data_root=dla_root, device="cpu", force=True, **roots)

    out = evaluate("smoke_dla_agreement", "smoke_dla", data_root=dla_root, device="cpu", **roots)
    results = read_results(out)
    jumps = results.manifest["jumps"]
    assert jumps == [2, 5, 10, 20, 40]  # fractions of the 40 particles after the 20-particle start
    rows = results.scalars["smoke_dla"]["containment"]
    assert set(rows) == {"shared", "fresh"} and len(rows["shared"]) == len(jumps) * (len(jumps) - 1) // 2
    assert all(0 <= v <= 1 for v in rows["shared"].values())
    assert set(results.scalars["smoke_dla"]["chain"]) == {"1x40", "2x20", "4x10"}
    assert {(r["evaluation"], r["predictor"]) for r in list_results(roots["results_root"])} == {
        ("smoke_dla", "smoke_dla"), ("smoke_dla_agreement", "smoke_dla")}


def test_load_combines_predictors(dla_root, roots):
    model_args = {k: roots[k] for k in ("models_root", "configs_root")}
    for name in ("smoke_dla", "smoke_dla_flow"):
        train_model(name, data_root=dla_root, **model_args)
        evaluate("smoke_dla", name, data_root=dla_root, device="cpu", **roots)
    both = load("smoke_dla", "smoke_dla", "smoke_dla_flow", root=roots["results_root"])
    assert set(both.manifest["entries"]) == set(both.manifest["models"]) == {"smoke_dla", "smoke_dla_flow"}
    assert set(both.manifest["predictors"]) == {"smoke_dla", "smoke_dla_flow"}
    assert {k.split("/")[0] for k in both.arrays} == {"smoke_dla", "smoke_dla_flow", "simulator", "radial_edges"}
    alone = read_results(roots["results_root"] / "smoke_dla" / "smoke_dla_flow")
    for key, value in alone.arrays.items():
        np.testing.assert_array_equal(both.arrays[key], value)
    assert set(load("smoke_dla", root=roots["results_root"]).manifest["predictors"]) == {"smoke_dla", "smoke_dla_flow"}


def test_changed_configs_are_refused(dla_root, roots):
    model_args = {k: roots[k] for k in ("models_root", "configs_root")}
    train_model("smoke_dla", data_root=dla_root, **model_args)
    evaluate("smoke_dla", "smoke_dla", data_root=dla_root, device="cpu", **roots)

    # An evaluation file edited after it has results.
    _edit(roots["configs_root"], "evaluations", "smoke_dla", batch_size=4)
    with pytest.raises(EvaluationChanged):
        evaluate("smoke_dla", "smoke_dla_flow", data_root=dla_root, device="cpu", **roots)

    # A model file edited after training. A field that does not change training is allowed.
    _edit(roots["configs_root"], "models", "smoke_dla", num_workers=3)
    train_model("smoke_dla", data_root=dla_root, **model_args)
    _edit(roots["configs_root"], "models", "smoke_dla", lr=1e-3)
    with pytest.raises(ConfigChanged, match="lr"):
        train_model("smoke_dla", data_root=dla_root, **model_args)
    with pytest.raises(ConfigChanged):
        evaluate("smoke_dla_agreement", "smoke_dla", data_root=dla_root, device="cpu", **roots)


def test_evaluate_checks_the_predictor(dla_root, roots):
    with pytest.raises(RuntimeError, match="not trained"):
        evaluate("smoke_dla", "smoke_dla", data_root=dla_root, **roots)
    with pytest.raises(ValueError, match="does not score a cascade"):
        evaluate("smoke_dla", "smoke_cascade", data_root=dla_root, **roots)
    with pytest.raises(ValueError, match="does not score a model"):
        evaluate("smoke_world", "smoke_dla", data_root=dla_root, **roots)
    with pytest.raises(ValueError, match="does not score a analytical_chunks baseline"):
        evaluate("smoke_erosion", "smoke_analytical_chunks", data_root=dla_root, **roots)
    with pytest.raises(KeyError, match="no predictor"):
        evaluate("smoke_dla", "nope", data_root=dla_root, **roots)


def test_missing_fraction():
    from erosion_sim.dla.suite import missing_fraction

    small = np.zeros((2, 4, 4), np.uint8)
    small[:, 0, :2] = 1
    large = small.copy()
    large[1, 0, 1] = 0
    np.testing.assert_allclose(missing_fraction(small, large), [0.0, 0.5])


def test_cascades_and_baselines_are_built_from_their_files(roots, monkeypatch):
    import erosion_sim.world.suite as world_suite
    from erosion_sim.paths import model_dir

    for name in ("smoke_coarse", "smoke_fine", "smoke_erosion"):
        folder = model_dir(name, roots["models_root"])
        folder.mkdir(parents=True)
        (folder / "done.json").write_text("{}")
    _edit(roots["configs_root"], "predictors", "smoke_cascade", steps=3)
    seen = {}
    monkeypatch.setattr(world_suite, "evaluate_world", lambda models, out, **kw: seen.update(models=models, **kw))
    evaluate("smoke_world", "smoke_cascade", **roots)
    assert set(seen["models"]) == {"smoke_coarse", "smoke_fine"} and "baselines" not in seen
    assert seen["settings"].cascades == (("smoke_cascade", ("smoke_coarse", "smoke_fine"),
                                          (("steps", 3), ("splits", (5.0,)))),)
    assert seen["settings"].data_profile == "erosion_smoke/smoke"
    assert seen["record"]["predictor"]["kind"] == "cascade"

    seen.clear()
    evaluate("smoke_world", "smoke_model_chunks", **roots)
    assert set(seen["models"]) == {"smoke_erosion"} and seen["settings"].cascades == ()
    assert seen["baselines"] == {"smoke_model_chunks": {"kind": "baseline", "method": "model_chunks",
                                                        "model": "smoke_erosion", "chunk": 16, "steps": 2}}
