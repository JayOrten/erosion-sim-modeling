import numpy as np
import torch
from torch.utils.data import DataLoader

from erosion_sim.data import ErosionPairs, select_runs
from erosion_sim.normalization import PARAMETER_NAMES

from conftest import RUNS, SIZE, SNAPS


def test_items_have_expected_shapes_and_round_trip(dataset, scaling):
    data = ErosionPairs(dataset, scaling)
    assert len(data) == RUNS * SNAPS * (SNAPS - 1) // 2
    item = data[7]
    assert item["x"].shape == item["y"].shape == (1, SIZE, SIZE)
    assert item["cond"].shape == (len(PARAMETER_NAMES),)
    batch = next(iter(DataLoader(data, batch_size=5)))
    K_ref = torch.as_tensor(dataset.params["K_ref"])[batch["run"]][:, None, None]
    dT = torch.as_tensor(dataset.times)[batch["end"]] - torch.as_tensor(dataset.times)[batch["start"]]
    z_b = batch["z_a"] + scaling.change.from_model(batch["y"][:, 0], K_ref, dT[:, None, None])
    torch.testing.assert_close(z_b.float(), batch["z_b"], atol=1e-4, rtol=0)


def test_conditioning_is_in_unit_range_on_fitted_data(dataset, scaling):
    data = ErosionPairs(dataset, scaling)
    cond = torch.stack([data[i]["cond"] for i in range(len(data))])
    assert cond.min() >= -1e-6 and cond.max() <= 1 + 1e-6


def test_lags_and_run_selection(dataset, scaling):
    params = dataset.params
    runs = select_runs(params, U=(0, 0))
    np.testing.assert_array_equal(runs, [0, 1])
    np.testing.assert_array_equal(select_runs(params, D_over_K=(1.5, 3)), [2, 3])
    data = ErosionPairs(dataset, scaling, runs=runs, lags=[2])
    assert len(data) == 2 * (SNAPS - 2)
    assert all(data[i]["end"] - data[i]["start"] == 2 for i in range(len(data)))
    assert {data[i]["run"] for i in range(len(data))} == {0, 1}
