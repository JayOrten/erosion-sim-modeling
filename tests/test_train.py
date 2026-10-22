import numpy as np
import torch

from erosion_sim.config import PROFILES
from erosion_sim.train import load_checkpoint, train


def test_smoke_training_saves_a_loadable_checkpoint(data_root, tmp_path):
    out = tmp_path / "unet-smoke.pt"
    history = train(PROFILES["smoke"], data_root=data_root, show=False, out=out)
    assert out.exists()
    assert len(history["val_loss"]) == 2
    assert all(torch.isfinite(torch.tensor(history["train_loss"])))

    model, normalizer, cfg, ckpt = load_checkpoint(out, device="cpu")
    assert cfg == PROFILES["smoke"]
    x, cond = torch.rand(2, 1, 16, 16), torch.rand(2, 5)
    with torch.no_grad():
        assert model(x, cond).shape == (2, 1, 16, 16)


def test_epochs_early_stopping_and_last_checkpoint(data_root, tmp_path):
    out = tmp_path / "unet-smoke.pt"
    # One epoch of the fake dataset is 4 runs x 15 pairs / batch 4 = 15 steps.
    history = train(PROFILES["smoke"], data_root=data_root, show=False, out=out, epochs=1.0, eval_every=5,
                    log_every=5)
    assert history["eval_step"][-1] == 15
    # With a learning rate of zero the validation loss never improves after the first evaluation.
    history = train(PROFILES["smoke"], data_root=data_root, show=False, out=out, lr=0.0, warmup_steps=0,
                    steps=20, eval_every=1, patience=2)
    assert history["eval_step"] == [1, 2, 3]
    last = torch.load(tmp_path / "unet-smoke-last.pt", weights_only=False)
    assert last["step"] == 3 and "optimizer" in last



def test_resume_continues_where_it_stopped(data_root, tmp_path, monkeypatch):
    import pytest

    import erosion_sim.train as train_module

    args = dict(data_root=data_root, show=False, out=tmp_path / "unet-smoke.pt", steps=8, eval_every=4, log_every=4)
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
