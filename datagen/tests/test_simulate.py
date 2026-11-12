import json

import numpy as np

from erosion_datagen.config import PROFILES
from erosion_datagen.simulate import (
    fill_pits,
    latin_hypercube,
    load_dataset,
    make_dataset,
    run_terrain,
    sample_params,
    simulate,
    terrain_kind,
)
from erosion_terrain import TERRAIN_FIELDS, Terrain, fbm, fractal_noise

CFG = PROFILES["smoke"]


def _smoke_run(**overrides):
    z0 = Terrain(0, period=16.0).heights(0, 0, *CFG.shape)
    params = dict(K=3e-4, D=3e-4, m=0.4, U=0.0) | overrides
    return z0, simulate(z0, **params, dt=CFG.dt, steps=CFG.steps, snapshot_every=CFG.snapshot_every)


def test_latin_hypercube_fills_every_slice():
    u = latin_hypercube(np.random.default_rng(0), 10, 3)
    assert u.shape == (10, 3)
    for column in u.T:
        assert sorted(np.floor(column * 10).astype(int)) == list(range(10))


def test_sample_params_is_deterministic_and_in_range():
    a, b = sample_params(CFG, "train"), sample_params(CFG, "train")
    assert set(TERRAIN_FIELDS) <= set(a)
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
    z0 = run_terrain({k: v[0] for k, v in params.items()}).heights(0, 0, *CFG.shape)
    alone = simulate(
        z0, params["K"][0], params["D"][0], params["m"][0], params["U"][0],
        dt=CFG.dt, steps=CFG.steps, snapshot_every=CFG.snapshot_every,
    )
    np.testing.assert_array_equal(snapshots[0], alone)


def test_fbm_overlapping_regions_agree():
    big = fbm((40, 50), seed=7, period=16.0, origin=(-10, 100_000))
    small = fbm((12, 9), seed=7, period=16.0, origin=(5, 100_020))
    np.testing.assert_array_equal(small, big[15:27, 20:29])
    assert not np.array_equal(big, fbm((40, 50), seed=8, period=16.0, origin=(-10, 100_000)))


def test_terrain_kind_fixed_and_varied():
    fixed = terrain_kind(CFG, "train", 50)
    assert np.all(fixed["noise"] == "perlin") and np.all(fixed["fractal"] == "fbm") and np.all(fixed["warp"] == 0)
    assert np.all(fixed["octaves"] == CFG.octaves) and np.all(fixed["height"] == CFG.height)
    varied = terrain_kind(PROFILES["world_master"], "train", 2_000)
    master = PROFILES["world_master"]
    assert set(varied["noise"]) == set(master.noise_types)
    assert set(varied["fractal"]) == set(master.fractal_types)
    assert set(varied["octaves"]) == set(range(master.octave_range[0], master.octave_range[1] + 1))
    assert np.all((varied["height"] >= 32) & (varied["height"] <= 128))
    assert 0.4 < (varied["warp"] > 0).mean() < 0.6
    assert np.all((varied["warp"] == 0) | ((varied["warp"] >= 0.1) & (varied["warp"] <= 0.5)))
    np.testing.assert_array_equal(varied["noise"], terrain_kind(master, "train", 2_000)["noise"])


def test_run_terrain_uses_every_field():
    p = dict(seed=np.int64(5), period=16.0, persistence=0.5, octaves=np.int64(3), lacunarity=2.2, height=40.0,
             noise=np.str_("opensimplex2"), fractal=np.str_("ridged"), warp=0.3)
    terrain = run_terrain(p)
    assert terrain == Terrain(5, 16.0, 0.5, 3, 2.2, 40.0, "opensimplex2", "ridged", 0.3)
    expected = 40.0 * fractal_noise(CFG.shape, seed=5, period=16.0, octaves=3, persistence=0.5, lacunarity=2.2,
                                    noise="opensimplex2", fractal="ridged", warp=0.3)
    np.testing.assert_array_equal(terrain.heights(0, 0, *CFG.shape), expected)


def test_make_dataset_resumes_after_an_interruption(tmp_path):
    whole = make_dataset(CFG, "val", tmp_path / "whole")

    class Stop(Exception):
        pass

    def stop_after_two(done, total):
        if done == 2:
            raise Stop

    try:
        make_dataset(CFG, "val", tmp_path / "parts", progress=stop_after_two)
    except Stop:
        pass
    record = json.loads((tmp_path / "parts" / "progress.json").read_text())
    assert len(record["done"]) == 2 and not (tmp_path / "parts" / "meta.json").exists()
    seen = []
    make_dataset(CFG, "val", tmp_path / "parts", progress=lambda done, total: seen.append(done))
    assert seen[0] == 2 and seen[-1] == CFG.val_runs  # only the missing runs ran
    np.testing.assert_array_equal(load_dataset(whole)[0], load_dataset(tmp_path / "parts")[0])
    assert not (tmp_path / "parts" / "progress.json").exists()


def test_make_dataset_starts_over_with_new_settings(tmp_path):
    from dataclasses import replace

    def stop(done, total):
        if done == 1:
            raise RuntimeError

    try:
        make_dataset(CFG, "val", tmp_path, progress=stop)
    except RuntimeError:
        pass
    seen = []
    make_dataset(replace(CFG, steps=10), "val", tmp_path, progress=lambda done, total: seen.append(done))
    assert seen[0] == 0
    assert load_dataset(tmp_path)[0].shape[1] == 10 // CFG.snapshot_every + 1
