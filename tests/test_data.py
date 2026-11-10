import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from erosion_sim.data import COND_NAMES, ErosionPairs, Normalizer, denormalize, select_runs

from conftest import RUNS, SIZE, SNAPS


@pytest.mark.parametrize("scale", ["fitted", "relief"])
def test_items_have_expected_shapes_and_round_trip(dataset_path, scale):
    norm = Normalizer.fit(dataset_path, scale=scale)
    data = ErosionPairs(dataset_path, norm)
    assert len(data) == RUNS * SNAPS * (SNAPS - 1) // 2
    item = data[7]
    assert item["x"].shape == item["y"].shape == (1, SIZE, SIZE)
    assert item["cond"].shape == (len(COND_NAMES),)
    assert 0.0 <= item["x"].min() and item["x"].max() <= 1.0
    batch = next(iter(DataLoader(data, batch_size=5)))
    z_b = denormalize(batch["y"], batch["z_a"], batch["relief"], batch["scale"])
    torch.testing.assert_close(z_b, batch["z_b"], atol=1e-4, rtol=0)


def test_conditioning_is_in_unit_range_on_fitted_data(dataset_path):
    data = ErosionPairs(dataset_path, Normalizer.fit(dataset_path))
    cond = torch.stack([data[i]["cond"] for i in range(len(data))])
    assert cond.min() >= -1e-6 and cond.max() <= 1 + 1e-6


def test_fitted_scale_evens_out_target_size(dataset_path):
    data = ErosionPairs(dataset_path, Normalizer.fit(dataset_path, scale="fitted"), lags=[1, SNAPS - 1])
    rms = {}
    for i in range(len(data)):
        item = data[i]
        rms.setdefault(item["end"] - item["start"], []).append(item["y"].pow(2).mean().sqrt().item())
    short, long = np.mean(rms[1]), np.mean(rms[SNAPS - 1])
    assert 0.5 < short / long < 2.0


def test_lags_and_run_selection(dataset_path):
    params = dict(np.load(dataset_path / "params.npz"))
    runs = select_runs(params, U=(0, 0))
    np.testing.assert_array_equal(runs, [0, 1])
    np.testing.assert_array_equal(select_runs(params, D_over_K=(1.5, 3)), [2, 3])
    data = ErosionPairs(dataset_path, Normalizer.fit(dataset_path), runs=runs, lags=[2])
    assert len(data) == 2 * (SNAPS - 2)
    assert all(data[i]["end"] - data[i]["start"] == 2 for i in range(len(data)))
    assert {data[i]["run"] for i in range(len(data))} == {0, 1}


def test_fixed_height_scale_is_the_same_for_every_window(dataset_path):
    norm = Normalizer.fit(dataset_path, height=64.0)
    assert norm.height == 64.0
    z = np.random.default_rng(0).normal(size=(2, SIZE, SIZE)) * 10
    x, relief = norm.input(z)
    np.testing.assert_array_equal(relief, [64.0, 64.0])
    np.testing.assert_allclose(x.mean(axis=(1, 2)), 0, atol=1e-12)
    # Raising a window changes nothing; doubling its relief doubles the input instead of rescaling it.
    np.testing.assert_allclose(norm.input(z + 100)[0], x)
    np.testing.assert_allclose(norm.input(2 * z)[0], 2 * x)

    data = ErosionPairs(dataset_path, norm)
    batch = next(iter(DataLoader(data, batch_size=5)))
    torch.testing.assert_close(batch["relief"], torch.full((5,), 64.0, dtype=torch.float64))
    z_b = denormalize(batch["y"], batch["z_a"], batch["relief"], batch["scale"])
    torch.testing.assert_close(z_b, batch["z_b"], atol=1e-4, rtol=0)
