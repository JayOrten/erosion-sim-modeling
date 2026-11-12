from dataclasses import replace

import numpy as np
import pytest
import torch

from erosion_sim.config import PROFILES, get_config
from erosion_sim.train import load_checkpoint, train


def test_smoke_training_saves_a_loadable_checkpoint(data_root, tmp_path):
    out = tmp_path / "regression-smoke.pt"
    history = train(PROFILES["smoke"], data_root=data_root, show=False, out=out)
    assert out.exists()
    assert len(history["val_loss"]) == 2
    assert all(torch.isfinite(torch.tensor(history["train_loss"])))

    model, scaling, cfg, ckpt = load_checkpoint(out, device="cpu")
    assert cfg == PROFILES["smoke"]
    assert len(scaling.parameters.low) == len(scaling.parameters.high) == 5
    x, cond = torch.rand(2, 1, 16, 16), torch.rand(2, 5)
    with torch.no_grad():
        assert model(x, cond).shape == (2, 1, 16, 16)


def test_epochs_early_stopping_and_last_checkpoint(data_root, tmp_path):
    out = tmp_path / "regression-smoke.pt"
    # One epoch of the fake dataset is 4 runs x 15 pairs / batch 4 = 15 steps.
    history = train(PROFILES["smoke"], data_root=data_root, show=False, out=out, epochs=1.0, eval_every=5,
                    log_every=5)
    assert history["eval_step"][-1] == 15
    # With a learning rate of zero the validation loss never improves after the first evaluation.
    history = train(PROFILES["smoke"], data_root=data_root, show=False, out=out, lr=0.0, warmup_steps=0,
                    steps=20, eval_every=1, patience=2)
    assert history["eval_step"] == [1, 2, 3]
    last = torch.load(tmp_path / "regression-smoke-last.pt", weights_only=False)
    assert last["step"] == 3 and "optimizer" in last



def test_resume_continues_where_it_stopped(data_root, tmp_path, monkeypatch):
    import pytest

    import erosion_sim.train as train_module

    args = dict(data_root=data_root, show=False, out=tmp_path / "regression-smoke.pt", steps=8, eval_every=4, log_every=4)
    full = train(PROFILES["smoke"], **args)

    # Interrupt at the second evaluation (step 8), so the last checkpoint is from step 4.
    evaluate, calls = train_module.evaluate_loss, []

    def interrupted(*a, **k):
        calls.append(1)
        if len(calls) == 2:
            raise KeyboardInterrupt
        return evaluate(*a, **k)

    monkeypatch.setattr(train_module, "evaluate_loss", interrupted)
    with pytest.raises(KeyboardInterrupt):
        train(PROFILES["smoke"], **args)
    monkeypatch.setattr(train_module, "evaluate_loss", evaluate)

    with pytest.raises(ValueError):
        train(PROFILES["smoke"], **args, lr=1e-2, resume=True)
    resumed = train(PROFILES["smoke"], **args, resume=True)
    assert resumed["eval_step"] == full["eval_step"] == [4, 8]
    np.testing.assert_allclose(resumed["train_loss"], full["train_loss"], rtol=1e-5)
    np.testing.assert_allclose(resumed["val_loss"], full["val_loss"], rtol=1e-5)


def test_checkpoint_names_follow_the_objective():
    from dataclasses import replace

    import pytest

    from erosion_sim.paths import checkpoint_path

    assert checkpoint_path("smoke").name == "regression-smoke.pt"
    assert checkpoint_path("smoke", "diffusion").name == "diffusion-smoke.pt"
    with pytest.raises(ValueError):
        replace(PROFILES["smoke"], objective="gan")


def test_checkpoint_without_objective_loads_as_regression(data_root, tmp_path):
    out = tmp_path / "regression-smoke.pt"
    train(PROFILES["smoke"], data_root=data_root, show=False, out=out)
    ckpt = torch.load(out, weights_only=False)
    del ckpt["config"]["objective"]
    torch.save(ckpt, out)

    _, _, cfg, _ = load_checkpoint(out, device="cpu")
    assert cfg.objective == "regression"


def test_free_edges_diffusion_trains_and_loads(data_root, tmp_path):
    out = tmp_path / "diffusion-smoke.pt"
    cfg = replace(get_config("smoke", "diffusion"), fixed_edges=False)
    train(cfg, data_root=data_root, show=False, out=out)
    model, _, loaded, ckpt = load_checkpoint(out, device="cpu")
    assert loaded == cfg
    assert model.sigma_data == ckpt["sigma_data"]
    assert not model.fixed_edges and not model.net.fixed_edges


@pytest.mark.parametrize("level", ["coarse", "fine"])
def test_world_levels_train_and_load(data_root, tmp_path, level):
    out = tmp_path / f"diffusion-smoke-{level}.pt"
    cfg = replace(get_config("smoke", "diffusion"), fixed_edges=False, level=level, crop=8, block=2)
    history = train(cfg, data_root=data_root, show=False, out=out)
    assert all(torch.isfinite(torch.tensor(history["train_loss"])))
    model, _, loaded, _ = load_checkpoint(out, device="cpu")
    assert loaded == cfg and model.net.stem.in_channels == cfg.in_channels + 1
    size = 8  # fine crops are 8²; coarse items are 16² runs as 2 × 2 means
    x = torch.rand(2, cfg.in_channels, size, size)
    with torch.no_grad():
        assert model(x, torch.rand(2, 5), torch.randn(2, 1, size, size), torch.tensor([1.0, 2.0])).shape == (2, 1, size, size)


def test_ema_is_saved_as_the_model_and_training_weights_kept_for_resuming(data_root, tmp_path):
    out = tmp_path / "regression-smoke.pt"
    train(PROFILES["smoke"], data_root=data_root, show=False, out=out)
    last = torch.load(tmp_path / "regression-smoke-last.pt", weights_only=False)
    ema, trained = last["model"], last["train_model"]
    assert ema.keys() == trained.keys()
    assert any(not torch.equal(ema[k], trained[k]) for k in ema)

    train(PROFILES["smoke"], data_root=data_root, show=False, out=out, ema_decay=None)
    last = torch.load(tmp_path / "regression-smoke-last.pt", weights_only=False)
    assert "train_model" not in last


def test_update_ema_moves_a_fraction_of_the_way():
    from erosion_sim.train import update_ema

    ema, model = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
    before = ema.weight.clone()
    update_ema(ema, model, decay=0.75)
    torch.testing.assert_close(ema.weight, before + 0.25 * (model.weight - before))
