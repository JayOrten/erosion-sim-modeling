"""Run a named evaluation on one predictor, into ``artifacts/results/<evaluation>/<predictor>/``.

An evaluation is a file ``configs/evaluations/<name>.yaml``: the suite (``dla``, ``dla_agreement``,
``erosion``, or ``world``), the test dataset (``data_profile``), and every field of that suite's
``Settings`` (but ``worlds``, which comes from the predictor). Every predictor scored under one evaluation sees the same test data and settings, so their
results can be compared.

A predictor is what an evaluation scores:

- a trained model, by name (``configs/models/<name>.yaml``), for the ``dla``, ``dla_agreement``, and
  ``erosion`` suites;
- a world, by name (``configs/worlds/<name>.yaml``), for the ``world`` suite. A world file lists its
  levels' models, coarsest first (``levels``), and its sampler's ``steps`` and ``splits``. The
  evaluation's own ``steps`` is used for the levels alone and the ``model_chunks`` baseline.

Two rules keep results consistent:

- An evaluation file may not change once it has results. ``evaluate`` compares the file with the copy in
  every existing result and refuses on a difference: put the change in a new evaluation file.
- A finished result is not replaced unless ``force`` is given.
"""

import json
from dataclasses import fields, replace
from pathlib import Path

import yaml

from ..config import check_fields, config_names, from_json_value
from ..paths import CONFIGS_DIR, results_dir
from ..records import logged
from ..training.run import trained_model

SUITES = ("dla", "dla_agreement", "erosion", "world")
WORLD_SAMPLER = ("steps", "splits")


class EvaluationChanged(RuntimeError):
    """An evaluation's file differs from the one its existing results were made with."""


def _path(kind: str, name: str, root: Path | None) -> Path:
    path = (Path(root) if root is not None else CONFIGS_DIR) / kind / f"{name}.yaml"
    if not path.exists():
        raise KeyError(f"no {kind[:-1]} {name!r}: {path} does not exist")
    return path


def _settings_class(suite: str):
    from ..dla.suite import AgreementSettings, Settings
    from ..erosion.suite import Settings as ErosionSettings
    from ..world.suite import Settings as WorldSettings

    return {"dla": Settings, "dla_agreement": AgreementSettings, "erosion": ErosionSettings,
            "world": WorldSettings}[suite]


def evaluation_config(name: str, root: Path | None = None) -> dict:
    """The contents of ``configs/evaluations/<name>.yaml``, checked: ``suite``, ``data_profile``, and
    every field of the suite's ``Settings`` but ``worlds``."""
    path = _path("evaluations", name, root)
    values = yaml.safe_load(path.read_text()) or {}
    if values.get("suite") not in SUITES:
        raise ValueError(f"{path}: suite must be one of {SUITES}, got {values.get('suite')!r}")
    check_fields(path, values, {"suite", "data_profile"}
                 | {f.name for f in fields(_settings_class(values["suite"]))} - {"worlds"})
    return values


def world_config(name: str, root: Path | None = None) -> dict:
    """The contents of ``configs/worlds/<name>.yaml``, checked: ``levels``, ``steps``, and ``splits``."""
    path = _path("worlds", name, root)
    values = yaml.safe_load(path.read_text()) or {}
    check_fields(path, values, {"levels", *WORLD_SAMPLER})
    return values


def _settings(suite: str, values: dict):
    """The suite's ``Settings`` from an evaluation file's values (lists become tuples)."""
    return _settings_class(suite)(**{k: from_json_value(v) for k, v in values.items()})


def evaluate(evaluation: str, predictor: str, force: bool = False, data_root: Path | None = None,
             models_root: Path | None = None, results_root: Path | None = None, configs_root: Path | None = None,
             device: str | None = None) -> Path:
    """Score ``predictor`` under ``evaluation`` and write its results folder. Output is also appended to
    ``evaluate.log`` there. Returns the folder.

    ``data_root`` replaces the test dataset's folder, and the ``*_root`` arguments replace the other
    folders; tests use them."""
    from ..dla.suite import evaluate_agreement, evaluate_dla
    from ..erosion.suite import evaluate_erosion
    from ..world.suite import evaluate_world

    values = evaluation_config(evaluation, configs_root)
    out = results_dir(evaluation, predictor, results_root)
    for manifest in sorted(out.parent.glob("*/manifest.json")):
        if json.loads(manifest.read_text()).get("evaluation") != values:
            raise EvaluationChanged(f"configs/evaluations/{evaluation}.yaml differs from the one used for "
                                    f"{manifest.parent}. Undo the edit and put the change in a new evaluation file.")
    if (out / "manifest.json").exists() and not force:
        raise FileExistsError(f"{out} already holds results; pass force to replace them")

    suite = values["suite"]
    settings = {k: v for k, v in values.items() if k != "suite"}
    is_world = predictor in config_names("worlds", configs_root)
    if is_world and predictor in config_names("models", configs_root):
        raise ValueError(f"{predictor} is both a model and a world; rename one")
    if (suite == "world") != is_world:
        raise ValueError(f"the {suite} suite scores {'worlds' if suite == 'world' else 'models'}; "
                         f"{predictor} is a {'world' if is_world else 'model'}")

    def model(name: str) -> Path:
        return trained_model(name, models_root, configs_root)

    if is_world:
        world = world_config(predictor, configs_root)
        levels = tuple(world["levels"])
        sampler = tuple((k, from_json_value(world[k])) for k in WORLD_SAMPLER)
        models = {name: model(name) for name in levels}
        if settings.get("chunk_model"):
            models[settings["chunk_model"]] = model(settings["chunk_model"])
        kwargs = {"settings": replace(_settings(suite, settings), worlds=((predictor, levels, sampler),))}
        record = {"predictor": {"name": predictor, "kind": "world", **world}}
    else:
        models = {predictor: model(predictor)}
        data_profile = settings.pop("data_profile")
        kwargs = {"settings": _settings(suite, settings)}
        if suite == "erosion":
            kwargs["settings"] = replace(kwargs["settings"], data_profile=data_profile)
        else:
            kwargs["data_profile"] = data_profile
        record = {"predictor": {"name": predictor, "kind": "model"}}

    run = {"dla": evaluate_dla, "dla_agreement": evaluate_agreement, "erosion": evaluate_erosion,
           "world": evaluate_world}[suite]
    with logged(out / "evaluate.log"):
        return run(models, out, data_root=data_root, device=device, record={"evaluation": values, **record},
                   **kwargs)
