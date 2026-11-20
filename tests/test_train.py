import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from erosion_sim.config import (
    PROFILES,
    config_from_json,
    config_to_json,
    get_config,
    training_differences,
)
from erosion_sim.paths import is_finished
from erosion_sim.training import train as train_module
from erosion_sim.training.checkpoints import load_checkpoint
from erosion_sim.training.train import train


def test_smoke_training_fills_the_model_folder(data_root, tmp_path):
    out = tmp_path / "model"
    history = train(PROFILES["smoke"], data_root=data_root, out=out)
    assert history["folder"] == out
    assert len(history["val_loss"]) == 2
    assert all(torch.isfinite(torch.tensor(history["train_loss"])))
    assert {p.name for p in out.iterdir()} == {"config.json", "best.pt", "last.pt", "history.json", "done.json"}
    assert is_finished(out)
    saved = json.loads((out / "config.json").read_text())
    assert config_from_json(saved["config"]) == PROFILES["smoke"]

    model, scaling, cfg, ckpt = load_checkpoint(out, device="cpu")
    assert cfg == PROFILES["smoke"]
    assert len(scaling.parameters.low) == len(scaling.parameters.high) == 4
    x, cond = torch.rand(2, 1, 16, 16), torch.rand(2, 4)
    with torch.no_grad():
        assert model(x, cond).shape == (2, 1, 16, 16)


def test_epochs_and_early_stopping(data_root, tmp_path):
    # One epoch of the fake dataset is 4 runs x 15 pairs / batch 4 = 15 steps.
    history = train(PROFILES["smoke"], data_root=data_root, out=tmp_path / "a", epochs=1.0, eval_every=5,
                    log_every=5)
    assert history["eval_step"][-1] == 15
    # With a learning rate of zero the validation loss never improves after the first evaluation.
    history = train(PROFILES["smoke"], data_root=data_root, out=tmp_path / "b", lr=0.0, warmup_steps=0,
                    steps=20, eval_every=1, patience=2)
    assert history["eval_step"] == [1, 2, 3]
    last = torch.load(tmp_path / "b" / "last.pt", weights_only=False)
    assert last["step"] == 3 and "optimizer" in last
    assert is_finished(tmp_path / "b")


def _interrupt_at_second_evaluation(monkeypatch) -> None:
    """Make training die at its second evaluation, as a killed job would, after one save at the first."""
    evaluate, calls = train_module.evaluate_loss, []

    def interrupted(*a, **k):
        calls.append(1)
        if len(calls) == 2:
            raise KeyboardInterrupt
        return evaluate(*a, **k)

    monkeypatch.setattr(train_module, "evaluate_loss", interrupted)


@pytest.mark.parametrize("objective", ["regression", "edm"])
def test_a_rerun_resumes_to_the_same_result(data_root, tmp_path, monkeypatch, objective):
    cfg = get_config("smoke", objective)
    args = dict(data_root=data_root, steps=8, eval_every=4, log_every=2)
    full = train(cfg, out=tmp_path / "full", **args)

    with monkeypatch.context() as m:
        _interrupt_at_second_evaluation(m)
        with pytest.raises(KeyboardInterrupt):
            train(cfg, out=tmp_path / "split", **args)
    assert not is_finished(tmp_path / "split")
    assert torch.load(tmp_path / "split" / "last.pt", weights_only=False)["step"] == 4

    with pytest.raises(ValueError):
        train(cfg, out=tmp_path / "split", **args, lr=1e-2)
    resumed = train(cfg, out=tmp_path / "split", **args)
    assert is_finished(tmp_path / "split")
    assert resumed["eval_step"] == full["eval_step"] == [4, 8]
    assert [x["start_step"] for x in resumed["sessions"]] == [0, 4]  # the first session and the resume
    np.testing.assert_allclose(resumed["train_loss"], full["train_loss"], rtol=1e-5)
    np.testing.assert_allclose(resumed["val_loss"], full["val_loss"], rtol=1e-5)
    a = torch.load(tmp_path / "full" / "best.pt", weights_only=False)["model"]
    b = torch.load(tmp_path / "split" / "best.pt", weights_only=False)["model"]
    for k in a:
        torch.testing.assert_close(a[k], b[k], rtol=1e-5, atol=1e-6)


def test_unknown_settings_are_rejected():
    for field, value in (("objective", "gan"), ("task", "fluid"), ("backbone", "mlp")):
        with pytest.raises(ValueError):
            replace(PROFILES["smoke"], **{field: value})
    with pytest.raises(ValueError):
        replace(PROFILES["smoke_dla"], fixed_edges=True)


def test_free_edges_edm_trains_and_loads(data_root, tmp_path):
    cfg = replace(get_config("smoke", "edm"), fixed_edges=False)
    train(cfg, data_root=data_root, out=tmp_path / "m")
    model, _, loaded, ckpt = load_checkpoint(tmp_path / "m", device="cpu")
    assert loaded == cfg
    assert model.sigma_data == ckpt["sigma_data"]
    assert not model.fixed_edges and not model.net.fixed_edges


@pytest.mark.parametrize("cell, context_cell, inputs", [(2, None, ()), (1, 2, ("map", "drainage"))])
def test_window_levels_train_and_load(data_root, tmp_path, cell, context_cell, inputs):
    cfg = replace(get_config("smoke", "edm"), fixed_edges=False, cell=cell, window=8 // cell, context_cell=context_cell,
                  inputs=inputs, augment=True, repeats=2)
    history = train(cfg, data_root=data_root, out=tmp_path / "m")
    assert all(torch.isfinite(torch.tensor(history["train_loss"])))
    model, _, loaded, _ = load_checkpoint(tmp_path / "m", device="cpu")
    assert loaded == cfg and model.net.stem.in_channels == cfg.in_channels + 1
    size = 8 // cell
    x = torch.rand(2, cfg.in_channels, size, size)
    with torch.no_grad():
        assert model(x, torch.rand(2, 4), torch.randn(2, 1, size, size), torch.tensor([1.0, 2.0])).shape == (2, 1, size, size)


@pytest.mark.parametrize("objective", ["regression", "edm", "flow_matching"])
def test_drainage_weighted_loss_trains(data_root, tmp_path, objective):
    cfg = replace(get_config("smoke", objective), drainage_weight=0.5)
    history = train(cfg, data_root=data_root, out=tmp_path / "m")
    assert all(np.isfinite(history["train_loss"]))


def test_dla_model_trains_and_loads(dla_root, tmp_path):
    cfg = PROFILES["smoke_dla"]
    history = train(cfg, data_root=dla_root, out=tmp_path / "m")
    assert all(np.isfinite(history["train_loss"] + history["val_loss"]))
    assert all(np.isfinite(history["val_skill"]))
    model, scaling, loaded, ckpt = load_checkpoint(tmp_path / "m", device="cpu")
    assert scaling is None and loaded == cfg and 0 < ckpt["sigma_data"] < 1
    size = 48
    with torch.no_grad():
        out = model(torch.rand(2, 1, size, size), torch.rand(2, 2), torch.randn(2, 1, size, size), torch.tensor([1.0, 2.0]))
    assert out.shape == (2, 1, size, size)


def test_ema_is_saved_as_the_model_and_training_weights_kept_for_resuming(data_root, tmp_path):
    train(PROFILES["smoke"], data_root=data_root, out=tmp_path / "a")
    last = torch.load(tmp_path / "a" / "last.pt", weights_only=False)
    ema, trained = last["model"], last["train_model"]
    assert ema.keys() == trained.keys()
    assert any(not torch.equal(ema[k], trained[k]) for k in ema)

    train(PROFILES["smoke"], data_root=data_root, out=tmp_path / "b", ema_decay=None)
    last = torch.load(tmp_path / "b" / "last.pt", weights_only=False)
    assert "train_model" not in last


def test_update_ema_moves_a_fraction_of_the_way():
    from erosion_sim.training.train import update_ema

    ema, model = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
    before = ema.weight.clone()
    update_ema(ema, model, decay=0.75)
    torch.testing.assert_close(ema.weight, before + 0.25 * (model.weight - before))


def test_config_round_trips_through_json_and_reports_training_differences():
    cfg = replace(PROFILES["smoke"], data=(("noise", frozenset({"perlin", "value_cubic"})), ("index", (0, 2667)),
                                           ("U", 0.0)), lags=(1, 2))
    assert config_from_json(json.loads(json.dumps(config_to_json(cfg)))) == cfg
    as_range = replace(cfg, data=(("noise", ("perlin", "value_cubic")),))
    assert config_from_json(config_to_json(as_range)) == as_range != replace(as_range, data=cfg.data)
    stored = config_to_json(cfg)
    stored.pop("dit_width")  # a field added after the model was trained takes its default
    stored["removed_field"] = 1  # a field since removed is ignored
    assert config_from_json(stored) == cfg
    assert training_differences(cfg, replace(cfg, profile="other", num_workers=1, log_every=9)) == []
    assert training_differences(cfg, replace(cfg, lr=1e-3, seed=2)) == ["lr", "seed"]

