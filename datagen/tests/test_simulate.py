import numpy as np

from erosion_datagen.config import PROFILES
from erosion_datagen.simulate import (
    fill_pits,
    latin_hypercube,
    load_dataset,
    make_dataset,
    sample_params,
    simulate,
    starting_terrain,
)

CFG = PROFILES["smoke"]


def _smoke_run(**overrides):
    z0 = starting_terrain(CFG, terrain_seed=0, period=16.0, persistence=0.5)
    params = dict(K=3e-4, D=3e-4, m=0.4, U=0.0) | overrides
    return z0, simulate(z0, **params, dt=CFG.dt, steps=CFG.steps, snapshot_every=CFG.snapshot_every)


def test_latin_hypercube_fills_every_slice():
    u = latin_hypercube(np.random.default_rng(0), 10, 3)
    assert u.shape == (10, 3)
    for column in u.T:
        assert sorted(np.floor(column * 10).astype(int)) == list(range(10))


def test_sample_params_is_deterministic_and_in_range():
    a, b = sample_params(CFG, "train"), sample_params(CFG, "train")
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])
    assert not np.array_equal(a["K"], sample_params(CFG, "val")["K"][: len(a["K"])])
    assert np.all((a["K_ref"] >= CFG.K_ref[0]) & (a["K_ref"] <= CFG.K_ref[1]))
    # At the reference drainage area, the cutting rate K * A_ref^m does not depend on m.
    np.testing.assert_allclose(a["K"] * CFG.A_ref ** a["m"], a["K_ref"] * CFG.A_ref ** CFG.m_ref)
    ratio = a["D"] / a["K_ref"]
    assert np.all((ratio >= CFG.D_over_K[0]) & (ratio <= CFG.D_over_K[1]))
    assert np.all(a["U"] >= 0)


def test_fill_pits_removes_pits():
    z = np.full((5, 5), 10.0)
    z[2, 2] = 1.0  # a single pit in the middle
    filled = fill_pits(z)
    assert filled[2, 2] >= 10.0
    assert np.all(filled >= z)


def test_simulate_is_deterministic():
    _, a = _smoke_run()
    _, b = _smoke_run()
    np.testing.assert_array_equal(a, b)


def test_simulate_keeps_edges_fixed_and_erodes():
    z0, snaps = _smoke_run(U=1e-4)
    assert snaps.shape == (CFG.steps // CFG.snapshot_every + 1, *CFG.shape)
    np.testing.assert_array_equal(snaps[0], z0.astype(np.float32))
    for edge in (snaps[:, 0, :], snaps[:, -1, :], snaps[:, :, 0], snaps[:, :, -1]):
        np.testing.assert_allclose(edge, edge[:1].repeat(len(edge), axis=0))
    assert np.abs(snaps[-1] - snaps[0]).mean() > 0


def test_make_dataset_round_trips(tmp_path):
    out = make_dataset(CFG, "val", tmp_path)
    snapshots, params, meta = load_dataset(out)
    assert snapshots.shape == (CFG.val_runs, CFG.steps // CFG.snapshot_every + 1, *CFG.shape)
    assert params["times"][-1] == CFG.steps * CFG.dt
    assert meta["split"] == "val"
    # Run 0 regenerated on its own matches the stored copy.
    z0 = starting_terrain(CFG, params["terrain_seed"][0], params["period"][0], params["persistence"][0])
    alone = simulate(
        z0, params["K"][0], params["D"][0], params["m"][0], params["U"][0],
        dt=CFG.dt, steps=CFG.steps, snapshot_every=CFG.snapshot_every,
    )
    np.testing.assert_array_equal(snapshots[0], alone)
