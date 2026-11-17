import numpy as np
import pytest

from erosion_sim.dla.config import PROFILES
from erosion_sim.dla.data import NO_START, DLAPairs, DLASplit, make_split, plan_split
from erosion_sim.dla.simulate import EMPTY, OutgrewGrid, occupancy, simulate
from erosion_sim.normalization import dla_conditioning

CFG = PROFILES["dla_smoke"]


def _connected(mask: np.ndarray) -> bool:
    """Whether the occupied cells form one 4-connected cluster, by flood fill from one of them."""
    mask = mask.astype(bool)
    reached = np.zeros_like(mask)
    reached[tuple(np.argwhere(mask)[0])] = True
    while True:
        grown = reached.copy()
        grown[1:] |= reached[:-1]
        grown[:-1] |= reached[1:]
        grown[:, 1:] |= reached[:, :-1]
        grown[:, :-1] |= reached[:, 1:]
        grown &= mask
        if np.array_equal(grown, reached):
            return bool(np.array_equal(reached, mask))
        reached = grown


def test_simulate_grows_one_connected_cluster_in_order():
    arrival = simulate(0.3, 80, seed=1, size=48)
    assert arrival.dtype == np.int16 and arrival.shape == (48, 48)
    assert arrival[24, 24] == 0
    np.testing.assert_array_equal(np.sort(arrival[arrival != EMPTY]), np.arange(80))
    for t in (1, 2, 10, 40, 80):
        assert occupancy(arrival, t).sum() == t
        assert _connected(occupancy(arrival, t))


def test_simulate_is_a_function_of_its_seed():
    np.testing.assert_array_equal(simulate(1.0, 60, seed=5, size=48), simulate(1.0, 60, seed=5, size=48))
    assert not np.array_equal(simulate(1.0, 60, seed=5, size=48), simulate(1.0, 60, seed=6, size=48))


def test_continuation_keeps_the_start():
    start = simulate(1.0, 30, seed=2, size=48)
    grown = simulate(1.0, 70, seed=3, size=48, start=start)
    np.testing.assert_array_equal(grown[start != EMPTY], start[start != EMPTY])
    assert (grown != EMPTY).sum() == 70 and _connected(grown != EMPTY)


def test_simulate_rejects_bad_settings():
    with pytest.raises(ValueError):
        simulate(0.0, 10, seed=0)
    with pytest.raises(OutgrewGrid):
        simulate(1.0, 400, seed=0, size=24)


def test_plan_split_follows_the_config():
    train = plan_split(CFG, "train")
    assert len(train["index"]) == CFG.train_runs and np.all(train["start_id"] == NO_START)
    assert np.all((train["p"] >= CFG.p_range[0]) & (train["p"] <= CFG.p_range[1]))
    assert len(set(train["seed"])) == CFG.train_runs
    test = plan_split(CFG, "test")
    fresh = test["start_id"] == NO_START
    assert fresh.sum() == len(CFG.test_p) * CFG.test_runs_per_p
    assert set(test["p"][fresh]) == set(CFG.test_p)
    for s, p in enumerate(CFG.start_p):
        assert np.all(test["p"][test["start_id"] == s] == p) and (test["start_id"] == s).sum() == CFG.runs_per_start


def test_make_split_round_trips_and_shares_starts(tmp_path):
    data = DLASplit(make_split(CFG, "test", tmp_path / "test"))
    assert data.arrivals.shape == (CFG.runs("test"), CFG.size, CFG.size)
    assert np.all((data.arrivals != EMPTY).sum(axis=(1, 2)) == CFG.particles)
    for s in range(len(CFG.start_p)):
        runs = data.select(start_id=s)
        firsts = [occupancy(data.arrivals[r], CFG.start_particles) for r in runs]
        assert all(np.array_equal(firsts[0], f) for f in firsts[1:])
        assert not np.array_equal(data.arrivals[runs[0]], data.arrivals[runs[1]])
    i = 3
    np.testing.assert_array_equal(
        data.arrivals[i], simulate(data.columns["p"][i], CFG.particles, int(data.columns["seed"][i]), CFG.size))


def test_pairs(tmp_path):
    data = DLASplit(make_split(CFG, "train", tmp_path / "train"))
    pairs = DLAPairs(data, runs=data.select(index=(0, 4)), repeats=50, seed=0)
    assert len(pairs) == 200
    seen = set()
    for i in range(0, len(pairs), 7):
        item = pairs[i]
        t, x = item["t"], item["steps"]
        assert 1 <= t <= int(0.9 * CFG.particles) and 1 <= x <= CFG.particles - t
        assert item["x"].shape == item["y"].shape == (1, CFG.size, CFG.size)
        assert item["x"].sum() == t and item["y"].sum() == t + x
        assert np.all(item["y"].numpy() >= item["x"].numpy())  # the start is kept
        assert item["run"] in (0, 1, 2, 3)
        seen.add((t, x))
    assert len(seen) > 10
    first = pairs[5]
    assert (first["t"], first["steps"]) == (pairs[5]["t"], pairs[5]["steps"])


def test_dla_conditioning_spans_unit_range():
    np.testing.assert_allclose(dla_conditioning(np.array([0.05, 1.0]), np.array([1, 1000]), (0.05, 1.0), 1000),
                               [[0, 0], [1, 1]], atol=1e-12)
