import numpy as np

from erosion_sim.config import PROFILES
from erosion_sim.evaluate import (METHODS, error_by_condition, error_by_elapsed, error_metrics, hydrology_metrics,
                                  predict_pairs, radial_power_spectrum, rollout_error, spectrum_comparison)
from erosion_sim.train import train


def test_radial_power_spectrum():
    i = np.arange(32)
    wave = np.sin(2 * np.pi * i / 8)[None, :] * np.ones((32, 1))
    wavelength, power = radial_power_spectrum(wave)
    assert wavelength[power.argmax()] == 8


def test_accuracy_metrics(data_root, tmp_path):
    ckpt = tmp_path / "unet-smoke.pt"
    train(PROFILES["smoke"], data_root=data_root, show=False, out=ckpt)
    pairs = predict_pairs(ckpt, count=10, data_root=data_root)
    assert all(pairs[m].shape == pairs["z_b"].shape for m in METHODS)
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
