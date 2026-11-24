"""Run a named evaluation on one predictor, into ``artifacts/results/<evaluation>/<predictor>/``.

An evaluation is a file ``configs/evaluations/<name>.yaml``: the suite (``dla``, ``dla_agreement``,
``erosion``, or ``world``), the test dataset (``data_profile``), and every field of that suite's
``Settings`` (but ``cascades``, which comes from the predictor). Every predictor scored under one
evaluation sees the same test data and settings, so their results can be compared.

A predictor is anything an evaluation scores:

- a trained model, by name (``configs/models/<name>.yaml``);
- a file in ``configs/predictors/<name>.yaml`` with a ``kind``:
  - ``cascade``: models stacked coarse to fine, each level conditioned on the coarser level's change
    (track C). Fields: ``levels`` (model names, coarsest first), and the sampler's ``steps`` and
    ``splits``.
  - ``baseline``: a method that is not a trained model. Field ``method``, plus that method's fields
    (``BASELINE_FIELDS``).

Which suite scores which kind (``SCORES``): the DLA suites score models; the erosion suite scores models
and the ``no_change`` and ``analytical`` baselines; the world suite scores cascades and every baseline.

Two rules keep results consistent:

- An evaluation file may not change once it has results. ``evaluate`` compares the file with the copy in
  every existing result and refuses on a difference: put the change in a new evaluation file.
- A finished result is not replaced unless ``force`` is given.
"""

import json
from dataclasses import fields, replace
from pathlib import Path

import yaml

from ..config import check_data_profile, check_fields, config_names, from_json_value
from ..logs import logged
from ..paths import CONFIGS_DIR, results_dir
from ..training.run import trained_model

SUITES = ("dla", "dla_agreement", "erosion", "world")
KINDS = ("model", "cascade", "baseline")
BASELINE_FIELDS = {"no_change": (), "analytical": (), "analytical_chunks": ("chunk",),
                   "analytical_blended": ("chunk",), "simulator_chunks": ("chunk",),
                   "model_chunks": ("model", "chunk", "steps")}
# suite -> (kinds it scores, baseline methods it scores)
SCORES = {"dla": (("model",), ()), "dla_agreement": (("model",), ()),
          "erosion": (("model", "baseline"), ("no_change", "analytical")),
          "world": (("cascade", "baseline"), tuple(BASELINE_FIELDS))}


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
    """The contents of ``configs/evaluations/<name>.yaml``, checked: ``suite`` and every field of the
    suite's ``Settings`` but ``cascades``. ``data_profile`` (the test dataset) must be set."""
    path = _path("evaluations", name, root)
    values = yaml.safe_load(path.read_text()) or {}
    if values.get("suite") not in SUITES:
        raise ValueError(f"{path}: suite must be one of {SUITES}, got {values.get('suite')!r}")
    check_fields(path, values, {"suite"} | {f.name for f in fields(_settings_class(values["suite"]))} - {"cascades"})
    if not values["data_profile"]:
        raise ValueError(f"{path}: data_profile (the test dataset) must be set")
    return values


def predictor_config(name: str, root: Path | None = None) -> dict:
    """What predictor ``name`` is: ``{"kind": "model"}`` for a model, else the contents of
    ``configs/predictors/<name>.yaml``, checked."""
    is_model = name in config_names("models", root)
    if name in config_names("predictors", root):
        if is_model:
            raise ValueError(f"{name} is both a model and a predictor file; rename one")
        path = _path("predictors", name, root)
        values = yaml.safe_load(path.read_text()) or {}
        kind = values.get("kind")
        if kind == "cascade":
            check_fields(path, values, {"kind", "levels", "steps", "splits"})
        elif kind == "baseline":
            if values.get("method") not in BASELINE_FIELDS:
                raise ValueError(f"{path}: method must be one of {tuple(BASELINE_FIELDS)}, got {values.get('method')!r}")
            check_fields(path, values, {"kind", "method", *BASELINE_FIELDS[values["method"]]})
        else:
            raise ValueError(f"{path}: kind must be cascade or baseline, got {kind!r}")
        return values
    if is_model:
        return {"kind": "model"}
    raise KeyError(f"no predictor {name!r}: not in configs/models/ or configs/predictors/")


def _settings(suite: str, values: dict):
    """The suite's ``Settings`` from an evaluation file's values (lists become tuples)."""
    return _settings_class(suite)(**{k: from_json_value(v) for k, v in values.items()})


def evaluate(evaluation: str, predictor: str, force: bool = False, data_root: Path | None = None,
             models_root: Path | None = None, results_root: Path | None = None, configs_root: Path | None = None,
             device: str | None = None) -> Path:
    """Score ``predictor`` under ``evaluation`` and write its results folder. Output is also appended to
    ``evaluate.log`` there. Returns the folder.

    The ``*_root`` arguments replace ``artifacts/data``, ``artifacts/models``, ``artifacts/results``, and
    ``configs``; tests use them."""
    from ..dla.suite import evaluate_agreement, evaluate_dla
    from ..erosion.suite import evaluate_erosion
    from ..world.suite import evaluate_world

    values = evaluation_config(evaluation, configs_root)
    check_data_profile(values["data_profile"], configs_root)
    out = results_dir(evaluation, predictor, results_root)
    for manifest in sorted(out.parent.glob("*/manifest.json")):
        if json.loads(manifest.read_text()).get("evaluation") != values:
            raise EvaluationChanged(f"configs/evaluations/{evaluation}.yaml differs from the one used for "
                                    f"{manifest.parent}. Undo the edit and put the change in a new evaluation file.")
    if (out / "manifest.json").exists() and not force:
        raise FileExistsError(f"{out} already holds results; pass force to replace them")

    suite = values["suite"]
    spec = predictor_config(predictor, configs_root)
    kinds, methods = SCORES[suite]
    if spec["kind"] not in kinds or (spec["kind"] == "baseline" and spec["method"] not in methods):
        what = f"{spec['method']} baseline" if spec["kind"] == "baseline" else spec["kind"]
        raise ValueError(f"the {suite} suite does not score a {what} ({predictor}); it scores {', '.join(kinds)}"
                         + (f" (baselines: {', '.join(methods)})" if "baseline" in kinds else ""))

    def model(name: str) -> Path:
        return trained_model(name, models_root, configs_root)

    settings = _settings(suite, {k: v for k, v in values.items() if k != "suite"})
    kwargs = {}
    models, baselines = {}, {}
    if spec["kind"] == "model":
        models[predictor] = model(predictor)
    elif spec["kind"] == "cascade":
        levels = tuple(spec["levels"])
        models = {name: model(name) for name in levels}
        sampler = (("steps", spec["steps"]), ("splits", from_json_value(spec["splits"])))
        settings = replace(settings, cascades=((predictor, levels, sampler),))
    else:
        baselines[predictor] = spec
        if spec["method"] == "model_chunks":
            models[spec["model"]] = model(spec["model"])
    if baselines:
        kwargs["baselines"] = baselines

    run = {"dla": evaluate_dla, "dla_agreement": evaluate_agreement, "erosion": evaluate_erosion,
           "world": evaluate_world}[suite]
    record = {"evaluation": values, "predictor": {"name": predictor, **spec}}
    with logged(out / "evaluate.log"):
        return run(models, out, settings=settings, data_root=data_root, device=device, record=record, **kwargs)
