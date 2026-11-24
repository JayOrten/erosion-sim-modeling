from dataclasses import replace

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from erosion_terrain import Terrain

from erosion_sim.config import model_config
from erosion_sim.data import hydrology
from erosion_sim.erosion import metrics, plots
from erosion_sim.erosion.baselines import (
    analytical_stream_power,
    blended,
    chunk_edge_mask,
    chunked,
)
from erosion_sim.erosion.suite import Settings, choose_pairs, evaluate_erosion
from erosion_sim.evaluation.results import read_results
from erosion_sim.training.train import train

matplotlib.use("Agg")


def test_fill_pits_and_drainage():
    z = Terrain(1, period=16).heights(0, 0, 48, 48)
    filled = hydrology.fill_pits(z)
    assert hydrology.pits(z).sum() > 0 and hydrology.pits(filled).sum() == 0 and np.all(filled >= z)
    area = hydrology.drainage_area(filled)
    edges = np.zeros(z.shape, dtype=bool)
    edges[0, :] = edges[-1, :] = edges[:, 0] = edges[:, -1] = True
    assert np.isclose(area[edges].sum(), z.size)  # every cell drains to an edge


def test_slope_area_exponent_recovers_a_power_law():
    # A valley down the middle column, flowing to the bottom edge, with steep sides draining into it. The
    # valley cell in row r drains about (r + 1) * n cells, and its slope is set to that area ** -0.5.
    n = 64
    area = (np.arange(n) + 1.0) * n
    drop = 0.5 * area ** -0.5
    valley = np.concatenate([np.cumsum(drop[:-1][::-1])[::-1], [0.0]])
    z = valley[:, None] + 1.0 * np.abs(np.arange(n) - n // 2)[None, :]
    theta = hydrology.slope_area_exponent(z, threshold=4 * n)
    assert abs(theta - 0.5) < 0.1


def test_analytical_stream_power():
    z = Terrain(2, period=16, height=32).heights(0, 0, 64, 64)
    filled = hydrology.fill_pits(z)
    np.testing.assert_allclose(analytical_stream_power(z, 3e-4, 0.4, 0.0), filled)
    later = analytical_stream_power(z, 3e-4, 0.4, 20_000.0)
    np.testing.assert_allclose(later[0], filled[0])  # edges stay fixed
    assert np.all(later <= filled + 1e-9) and (filled - later).mean() > 0
    much_later = analytical_stream_power(z, 3e-4, 0.4, 200_000.0)
    assert (filled - much_later).mean() > (filled - later).mean()
    soft = analytical_stream_power(z, 3e-4, 0.4, 20_000.0, K_map=np.full(z.shape, 3.0))
    assert (filled - soft).mean() > (filled - later).mean()


def test_chunk_helpers():
    z = np.random.default_rng(0).random((32, 32))
    np.testing.assert_allclose(chunked(lambda c, o: 2 * c, z, 16), 2 * z)
    np.testing.assert_allclose(blended(lambda c, o: 2 * c, z, 16), 2 * z)
    mask = chunk_edge_mask((8, 8), 4, 1)
    assert mask[3, 0] and mask[4, 7] and mask[0, 3] and not mask[0, 0] and not mask[7, 7]


def test_metrics_of_the_truth_are_perfect():
    z_a = Terrain(3, period=16).heights(0, 0, 32, 32)
    z_b = z_a - 0.1 * np.abs(z_a)
    truth = metrics.Truth(z_a, z_b, m=0.45)
    perfect = metrics.pair_metrics(z_b, truth)
    assert perfect["sq_error"] == 0 and perfect["iou"] == 1 and perfect["log_area_rmse"] == 0
    still = metrics.pair_metrics(z_a, truth)
    summary = metrics.summarize({k: np.array([perfect[k], still[k]]) for k in metrics.PAIR_METRICS})
    assert summary["pairs"] == 2 and 0 < summary["skill"] < 1
    assert metrics.summarize({k: np.array([still[k]]) for k in metrics.PAIR_METRICS})["skill"] == 0


def test_choose_pairs(dataset):
    random = choose_pairs(dataset, np.arange(4), Settings(pairs_per_run=3))
    assert random.shape == (12, 3) and np.all(random[:, 1] < random[:, 2])
    np.testing.assert_array_equal(random, choose_pairs(dataset, np.arange(4), Settings(pairs_per_run=3)))
    ends = choose_pairs(dataset, np.arange(2), Settings(ends_yr=(2000.0, 5000.0)))
    np.testing.assert_array_equal(ends, [[0, 0, 2], [0, 0, 5], [1, 0, 2], [1, 0, 5]])


@pytest.fixture(scope="module")
def two_models(tmp_path_factory):
    from conftest import write_fake_dataset

    root = tmp_path_factory.mktemp("erosion")
    for split, seed in (("train", 0), ("val", 1), ("test", 2)):
        write_fake_dataset(root / "data" / split, seed=seed)
    folders = {}
    for objective in ("regression", "edm"):
        cfg = replace(model_config("smoke"), objective=objective)
        folders[objective] = train(cfg, data_root=root / "data", out=root / objective)["folder"]
    return root, folders


def test_suite_writes_every_metric_and_breakdown(two_models, tmp_path):
    root, folders = two_models
    settings = Settings(pairs_per_run=3, samples=2, steps=2, fill_pits=True, batch_size=4)
    out = evaluate_erosion(folders, tmp_path / "r", settings, data_root=root / "data", device="cpu")
    results = read_results(out)
    names = list(results.manifest["entries"])
    assert names == ["no change", "analytical", "regression", "regression + pit filling", "edm", "edm + pit filling"]
    assert results.manifest["entries"]["edm"]["network_calls"] == 3
    rows = results.scalars["entries"]
    assert rows["no change"]["skill"] == 0 and rows["regression"]["pairs"] == 12
    assert "spread_correlation" in rows["edm"] and "spread_correlation" not in rows["regression"]
    assert rows["regression + pit filling"]["pits_per_km2"] == 0
    assert set(results.scalars["by"]) >= {"K_ref", "dT", "m"}
    for groups in results.scalars["by"].values():
        assert sum(g["no change"]["pairs"] for g in groups.values()) == 12
    assert results.arrays["regression/sq_error"].shape == (12,)
    assert results.examples["edm"].shape == (3, 16, 16)
    plots.table(results)
    plots.by_table(results, "dT")
    for figure in (plots.plot_by(results, "K_ref", show=False), plots.plot_spectrum(results, show=False),
                   plots.plot_examples(results, show=False)):
        assert figure is not None
    plt.close("all")


def test_suite_step_sweep_and_chains(two_models, tmp_path):
    root, folders = two_models
    settings = Settings(ends_yr=(2000.0, 4000.0), samples=1, step_sweep=(1, 2), baselines=(), batch_size=4,
                        chains=((2, 1000.0), (2, 2000.0)))
    out = evaluate_erosion({"edm": folders["edm"]}, tmp_path / "r", settings, data_root=root / "data",
                           device="cpu")
    results = read_results(out)
    info = results.manifest["entries"]
    assert [e for e in info if info[e]["kind"] == "model"] == ["edm@1", "edm@2"]
    assert info["edm chained 2x1000"]["network_calls"] == 2 * 63  # chains use the default 32 steps
    assert results.scalars["entries"]["edm chained 2x1000"]["pairs"] == 4  # the pairs ending at 2,000 yr
    assert results.scalars["entries"]["edm chained 2x2000"]["pairs"] == 4
    plots.plot_steps(results, show=False)
    plt.close("all")


def test_suite_rejects_world_levels(two_models, tmp_path):
    root, folders = two_models
    cfg = replace(model_config("smoke"), objective="edm", fixed_edges=False, window=8)
    level = train(cfg, data_root=root / "data", out=tmp_path / "level")["folder"]
    with pytest.raises(ValueError, match="world"):
        evaluate_erosion({"level": level}, tmp_path / "r", data_root=root / "data", device="cpu")
