import json

import matplotlib
import matplotlib.pyplot
import numpy as np
import pytest

from erosion_sim.config import PROFILES
from erosion_sim.dla import metrics, plots
from erosion_sim.dla.config import PROFILES as DATA_PROFILES
from erosion_sim.dla.data import DLASplit
from erosion_sim.dla.simulate import occupancy, run_seed, simulate
from erosion_sim.dla.suite import Settings, conditions, evaluate_dla
from erosion_sim.experiments.results import read_results, write_results
from erosion_sim.training.train import train

matplotlib.use("Agg")
DATA = DATA_PROFILES["dla_smoke"]


def _cross(size: int = 9) -> np.ndarray:
    occ = np.zeros((1, size, size), dtype=np.uint8)
    occ[0, size // 2, 2:-2] = 1
    occ[0, 2:-2, size // 2] = 1
    return occ


def test_counts_connectivity_and_start_kept():
    cross = _cross()
    assert metrics.particle_counts(cross)[0] == 9
    assert metrics.connected_fraction(cross)[0] == 1.0
    floating = cross.copy()
    floating[0, 0, 0] = 1
    assert metrics.connected_fraction(floating)[0] == pytest.approx(9 / 10)
    no_seed = cross.copy()
    no_seed[0, 4, 4] = 0
    assert metrics.connected_fraction(no_seed)[0] == 0.0
    start = np.zeros_like(cross[0])
    start[4, 3:6] = 1
    assert metrics.start_kept(cross, start)[0] == 1.0
    assert metrics.start_kept(no_seed, start)[0] == pytest.approx(2 / 3)


def test_shape_metrics_on_simulator_clusters():
    occ = {p: np.stack([occupancy(simulate(p, 600, run_seed("test", p, i)), 600) for i in range(40)])
           for p in (0.05, 1.0)}
    dense, sparse = (metrics.fractal_dimension(occ[p]) for p in (0.05, 1.0))
    assert np.nanmean(dense) > np.nanmean(sparse) + 0.1  # lower p packs clusters more densely
    assert 1.5 < np.nanmean(sparse) < 1.8
    assert metrics.gyration_radius(occ[1.0]).mean() > metrics.gyration_radius(occ[0.05]).mean()
    assert np.all(metrics.connected_fraction(occ[1.0]) == 1)


def test_wasserstein():
    a = np.arange(100.0)
    assert metrics.wasserstein(a, a) == 0
    assert metrics.wasserstein(a, a + 3) == pytest.approx(3)


def test_conditions_cover_the_test_split(dla_root):
    test = DLASplit(dla_root / "test")
    conds = conditions(test)
    assert [c.name for c in conds] == ["p=0.1", "p=1", "start=0", "start=1"]
    assert sum(len(c.runs) for c in conds) == len(test)
    assert all(c.t + c.x == DATA.particles for c in conds)


def test_results_round_trip(tmp_path):
    write_results(tmp_path / "r", {"a": 1}, {"x": np.ones(2)}, {}, {"suite": "test"})
    read = read_results(tmp_path / "r")
    assert read.scalars == {"a": 1} and read.manifest["suite"] == "test" and "time" in read.manifest


@pytest.mark.parametrize("objective", ["edm", "regression"])
def test_suite_writes_a_readable_results_folder(dla_root, tmp_path, objective):
    from dataclasses import replace

    cfg = replace(PROFILES["smoke_dla"], objective=objective)
    model = train(cfg, data_root=dla_root, out=tmp_path / "model")["folder"]
    out = evaluate_dla({"base": model}, tmp_path / "results", data_root=dla_root,
                       settings=Settings(steps=2, batch_size=4), device="cpu")
    results = read_results(out)
    assert results.manifest["models"]["base"]["folder"].endswith("/model")
    assert results.manifest["conditions"] == ["p=0.1", "p=1", "start=0", "start=1"]
    for cond in results.manifest["conditions"]:
        row = results.scalars["base"][cond]
        assert set(row) == set(metrics.PER_CLUSTER)
        assert row["particles"]["simulator_mean"] == DATA.particles
        assert row["start_kept"]["simulator_mean"] == 1.0
        assert results.examples[f"base/{cond}"].shape[1:] == (DATA.size, DATA.size)
    assert results.scalars["base"]["seconds_per_sample"] > 0
    json.dumps(results.scalars)  # plain JSON
    for metric in metrics.PER_CLUSTER:
        plots.plot_metric(results, metric, show=False)
    plots.plot_examples(results, show=False)
    plots.plot_radial_density(results, show=False)
    matplotlib.pyplot.close("all")


def test_step_sweep_scores_each_model_at_each_step_count(dla_root, tmp_path):
    from dataclasses import replace

    folders = {}
    for objective in ("edm", "flow_matching"):
        cfg = replace(PROFILES["smoke_dla"], objective=objective)
        folders[objective] = train(cfg, data_root=dla_root, out=tmp_path / objective)["folder"]
    out = evaluate_dla(folders, tmp_path / "results", data_root=dla_root,
                       settings=Settings(step_sweep=(1, 3), batch_size=4, only="p="), device="cpu")
    results = read_results(out)
    entries = results.manifest["entries"]
    assert list(entries) == ["edm@1", "edm@3", "flow_matching@1", "flow_matching@3"]
    assert [e["network_calls"] for e in entries.values()] == [1, 5, 1, 3]
    assert results.manifest["conditions"] == ["p=0.1", "p=1"]
    assert plots.models(results) == list(entries)
    plots.plot_steps(results, show=False)
    plots.table(results)
    matplotlib.pyplot.close("all")
