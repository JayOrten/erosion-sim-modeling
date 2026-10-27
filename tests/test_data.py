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
