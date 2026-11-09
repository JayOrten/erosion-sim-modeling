import numpy as np

from erosion_sim.config import PROFILES, get_config
from erosion_sim.evaluate import (error_by_condition, error_by_elapsed, error_metrics, hydrology_metrics, predict_pairs,
                                  radial_power_spectrum, rollout_error, sample_spread, show_comparison,
                                  spectrum_comparison, step_sweep, time_model)
from erosion_sim.train import train


def test_radial_power_spectrum():
    i = np.arange(32)
    wave = np.sin(2 * np.pi * i / 8)[None, :] * np.ones((32, 1))
    wavelength, power = radial_power_spectrum(wave)
    assert wavelength[power.argmax()] == 8


def test_accuracy_metrics(data_root, tmp_path):
    ckpt = tmp_path / "regression-smoke.pt"
    train(PROFILES["smoke"], data_root=data_root, show=False, out=ckpt)
    pairs = predict_pairs(ckpt, count=10, data_root=data_root)
    assert pairs["methods"] == ["no change", "model"]
    assert all(pairs[m].shape == pairs["z_b"].shape for m in pairs["methods"])
    errors = error_metrics(pairs, show=False)
    assert errors["no change"]["skill"] == 0.0
    hydro = hydrology_metrics(pairs, show=False)
    assert 0 <= hydro["model"]["iou"] <= hydro["model"]["iou_1"] <= 1
    assert np.isfinite(spectrum_comparison(pairs, show=False)["model"]).all()
    assert "K_ref" in error_by_condition(pairs, bins=2, show=False)
    assert len(error_by_elapsed(pairs, bins=2, show=False)["model"]["rmse"]) >= 1
    rollout = rollout_error(ckpt, runs=2, show=False, data_root=data_root)
    assert len(rollout["rollout"]["rmse"]) == len(rollout["elapsed"])
    assert "direct" in rollout  # smoke trains on every elapsed time


def test_regression_against_diffusion(data_root, tmp_path, monkeypatch):
    import matplotlib

    matplotlib.use("Agg")
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)
    reg, dif = tmp_path / "regression-smoke.pt", tmp_path / "diffusion-smoke.pt"
    train(PROFILES["smoke"], data_root=data_root, show=False, out=reg)
    train(get_config("smoke", "diffusion"), data_root=data_root, show=False, out=dif)
    checkpoints = {"regression": reg, "diffusion": dif}

    pairs = predict_pairs(checkpoints, count=6, samples=3, steps=3, batch_size=4, data_root=data_root)
    assert pairs["methods"] == ["no change", "regression", "diffusion (1 sample)", "diffusion (mean of 3)"]
    assert pairs["kind"]["diffusion (mean of 3)"] == "mean"
    assert all(pairs[m].shape == pairs["z_b"].shape for m in pairs["methods"])
    assert pairs["spread"]["diffusion"].shape == pairs["z_b"].shape and (pairs["spread"]["diffusion"] > 0).any()
    # The regression column matches a single-checkpoint run on the same pairs.
    alone = predict_pairs(reg, count=6, data_root=data_root)
    np.testing.assert_allclose(alone["model"], pairs["regression"], rtol=1e-5, atol=1e-5)
    errors = error_metrics(pairs)
    assert all(np.isfinite(errors[m]["rmse"]) for m in pairs["methods"])
    assert set(hydrology_metrics(pairs)) >= set(pairs["methods"])
    assert set(spectrum_comparison(pairs)) >= {"simulator", *pairs["methods"][1:]}
    assert "regression" in error_by_condition(pairs, bins=2)["K_ref"]
    assert "diffusion (1 sample)" in error_by_elapsed(pairs, bins=2)
    spread = sample_spread(pairs, bins=3)
    assert -1 <= spread["diffusion"]["regression"]["correlation"] <= 1
    show_comparison(checkpoints, count=2, data_root=data_root)

    from erosion_sim.train import load_checkpoint

    rollout = rollout_error(dif, runs=2, show=False, data_root=data_root)
    assert np.isfinite(rollout["rollout"]["rmse"]).all()
    model, _, _, _ = load_checkpoint(dif, device="cpu")
    assert time_model(model, size=(16, 16), batch_sizes=(1,), repeats=1, steps=2)[1] > 0


def test_step_sweep(data_root, tmp_path):
    reg, dif = tmp_path / "regression-smoke.pt", tmp_path / "diffusion-smoke.pt"
    train(PROFILES["smoke"], data_root=data_root, show=False, out=reg)
    train(get_config("smoke", "diffusion"), data_root=data_root, show=False, out=dif)
    sweep = step_sweep(dif, steps=(2, 4), regression=reg, count=4, samples=2, show=False, data_root=data_root)
    assert len(sweep["ms_per_sample"]) == 2 and (sweep["ms_per_sample"] > 0).all()
    assert all(np.isfinite(sweep[label]["rmse"]).all() for label in ("1 sample", "mean of 2"))
    assert np.isfinite(sweep["regression"]["fine"])
