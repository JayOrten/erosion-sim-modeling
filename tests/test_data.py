import json

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from erosion_sim.data import COND_NAMES, ErosionPairs, Normalizer, denormalize, select_runs

RUNS, SNAPS, SIZE = 4, 6, 16


@pytest.fixture
def dataset_path(tmp_path):
    """A tiny dataset in the format datagen writes: terrain that lowers at a rate set by K_ref."""
    rng = np.random.default_rng(0)
    K_ref = np.array([1e-4, 2e-4, 3e-4, 4e-4])
    z0 = rng.uniform(0, 64, (RUNS, 1, SIZE, SIZE))
    t = np.arange(SNAPS)[None, :, None, None] * 1000.0
    snapshots = (z0 * np.exp(-0.05 * K_ref[:, None, None, None] * t)).astype(np.float32)
    snapshots[:, :, 0, :] = snapshots[:, :1, 0, :]  # fixed edges, as in the simulator
    np.save(tmp_path / "snapshots.npy", snapshots)
    np.savez(
        tmp_path / "params.npz",
        times=np.arange(SNAPS) * 1000.0,
        K=K_ref * 1.2, K_ref=K_ref, D=K_ref * np.array([0.5, 1, 2, 3]), m=np.array([0.3, 0.4, 0.5, 0.6]),
        U=np.array([0.0, 0.0, 1e-5, 2e-5]), period=np.full(RUNS, 64.0), persistence=np.full(RUNS, 0.5),
        terrain_seed=np.arange(RUNS),
    )
    (tmp_path / "meta.json").write_text(json.dumps({"config": {}, "split": "train"}))
    return tmp_path


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
