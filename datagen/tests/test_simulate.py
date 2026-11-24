import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from erosion_datagen.config import group_names, load_group
from erosion_datagen.simulate import (
    SettingsChanged,
    dataset_dir,
    draw_run,
    fill_pits,
    load_dataset,
    make_chunked,
    make_dataset,
    plan_split,
    run_seed,
    run_terrain,
    simulate,
)
from erosion_terrain import (
    ErodibilityMap,
    Terrain,
    fbm,
    fractal_noise,
    map_from_columns,
)

# The tests' own groups, in the main repo's tests/configs/data/.
TEST_GROUPS = Path(__file__).resolve().parents[2] / "tests" / "configs" / "data"
CFG = load_group("erosion_smoke/smoke", TEST_GROUPS)


def _smoke_run(**overrides):
    z0 = Terrain(0, period=16.0).heights(0, 0, CFG.size, CFG.size)
    params = dict(K=3e-4, D=3e-4, m=0.4, U=0.0) | overrides
    return z0, simulate(z0, **params, dt=CFG.dt, steps=CFG.steps, snapshot_every=CFG.snapshot_every)


def test_draws_are_per_run_and_deterministic():
    a = draw_run(CFG, "train", 3)
    assert a == draw_run(CFG, "train", 3)
    assert a != draw_run(CFG, "train", 4) and a != draw_run(CFG, "val", 3)
    # A run's settings do not depend on how many runs the split has.
    assert plan_split(replace(CFG, train_runs=10), "train")["K"][3] == a["K"]
    assert run_seed(CFG, "train", 3) != run_seed(replace(CFG, name="other"), "train", 3)


def test_main_group_draws_follow_the_plan():
    group = load_group("erosion_master/main_256")
    cols = plan_split(replace(group, train_runs=3_000), "train")
    n = len(cols["index"])
    assert set(cols["group"]) == {"main_256"} and set(cols["split"]) == {"train"}
    # 12 noise combinations, equally likely; maps 2/3 uniform, 1/6 patches, 1/6 bands.
    combos = {(a, b, w > 0) for a, b, w in zip(cols["noise"], cols["fractal"], cols["warp"])}
    assert len(combos) == 12
    shares = {s: (cols["map_shape"] == s).mean() for s in ("uniform", "patches", "bands")}
    assert abs(shares["uniform"] - 2 / 3) < 0.03 and abs(shares["patches"] - 1 / 6) < 0.03
    assert abs((cols["fractal"] == "ridged").mean() - 0.5) < 0.03
    pipeline_check = ((cols["noise"] == "perlin") & (cols["fractal"] == "fbm") & (cols["warp"] == 0)
                      & (cols["map_shape"] == "uniform")).mean()
    assert abs(pipeline_check - 1 / 18) < 0.015
    for name, (low, high) in (("period", group.period), ("height", group.height), ("K_ref", group.K_ref),
                              ("m", group.m), ("persistence", group.persistence), ("lacunarity", group.lacunarity)):
        assert cols[name].min() >= low and cols[name].max() <= high, name
    assert set(cols["octaves"]) == {4, 5, 6, 7}
    ratio = cols["D"] / cols["K_ref"]
    assert ratio.min() >= group.D_over_K[0] and ratio.max() <= group.D_over_K[1]
    np.testing.assert_allclose(cols["K"] * group.A_ref ** cols["m"], cols["K_ref"] * group.A_ref ** group.m_ref)
    assert np.all(cols["map_contrast"][cols["map_shape"] == "uniform"] == 1)
    assert np.all(cols["U"] == 0) and np.all(cols["ood_param"] == "") and len(set(cols["run_seed"])) == n


def test_scaled_bands_and_test_only_groups():
    assert load_group("erosion_master/main_2048").period == (128.0, 256.0) and load_group("erosion_master/main_2048").band_width == (32.0, 256.0)
    assert load_group("erosion_master/main_1024").snapshots == 11 and load_group("erosion_master/long_256").snapshots == 31
    assert load_group("erosion_master/ood_256").splits == ("test",)
    cols = plan_split(load_group("erosion_master/ood_256"), "test")
    assert set(cols["noise"]) == {"perlin"} and set(cols["fractal"]) == {"fbm"} and set(cols["warp"]) == {0.0}
    assert set(cols["map_shape"]) == {"uniform"}
    for i, (param, (low, high)) in enumerate(load_group("erosion_master/ood_256").ood_ranges):
        rows = cols["index"] % 3 == i
        assert set(cols["ood_param"][rows]) == {param}
        values = cols["D"][rows] / cols["K_ref"][rows] if param == "D_over_K" else cols[param][rows]
        assert values.min() >= low and values.max() <= high


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
    z0, snaps = _smoke_run()
    assert snaps.shape == (CFG.snapshots, CFG.size, CFG.size)
    np.testing.assert_array_equal(snaps[0], z0.astype(np.float32))
    for edge in (snaps[:, 0, :], snaps[:, -1, :], snaps[:, :, 0], snaps[:, :, -1]):
        np.testing.assert_allclose(edge, edge[:1].repeat(len(edge), axis=0))
    assert np.abs(snaps[-1] - snaps[0]).mean() > 0


def test_soft_rock_erodes_faster():
    z0 = Terrain(0, period=16.0).heights(0, 0, 32, 32)
    soft = np.ones((32, 32))
    soft[:, 16:] = 10.0
    snaps = simulate(z0, 3e-4, 1e-5, 0.4, 0.0, dt=100.0, steps=20, snapshot_every=20, K_map=soft)
    plain = simulate(z0, 3e-4, 1e-5, 0.4, 0.0, dt=100.0, steps=20, snapshot_every=20)
    drop, plain_drop = snaps[0] - snaps[-1], plain[0] - plain[-1]
    assert drop[2:-2, 18:-2].mean() > 2 * plain_drop[2:-2, 18:-2].mean()
    assert drop[2:-2, 18:-2].mean() > 2 * drop[2:-2, 2:14].mean()


def test_make_dataset_round_trips_with_maps(tmp_path):
    out = make_dataset(CFG, "train", tmp_path)
    snapshots, cols, meta = load_dataset(out)
    assert snapshots.shape == (CFG.train_runs, CFG.snapshots, CFG.size, CFG.size)
    assert meta["times"][-1] == CFG.steps * CFG.dt and meta["split"] == "train"
    np.testing.assert_array_equal(cols["index"], np.arange(CFG.train_runs))
    for i in range(CFG.train_runs):  # each run regenerated on its own matches the stored copy
        row = {k: v[i] for k, v in cols.items()}
        z0 = run_terrain(row).heights(0, 0, CFG.size, CFG.size)
        K_map = map_from_columns(row).factors(0, 0, CFG.size, CFG.size)
        alone = simulate(z0, row["K"], row["D"], row["m"], row["U"], dt=CFG.dt, steps=CFG.steps,
                         snapshot_every=CFG.snapshot_every, K_map=K_map)
        np.testing.assert_array_equal(snapshots[i], alone)
    assert dataset_dir(CFG, "train").parts[-3:] == ("erosion_smoke", "smoke", "train")


def test_map_from_columns():
    row = {"map_shape": np.str_("bands"), "map_contrast": 4.0, "map_period": 30.0, "band_width": 12.0,
           "band_angle": 45.0, "period": 20.0, "map_seed": np.int64(7)}
    assert map_from_columns(row) == ErodibilityMap("bands", 4.0, 30.0, 12.0, 45.0, 20.0, 7)
    with pytest.raises(ValueError):
        ErodibilityMap("stripes")


def test_fbm_overlapping_regions_agree():
    big = fbm((40, 50), seed=7, period=16.0, origin=(-10, 100_000))
    small = fbm((12, 9), seed=7, period=16.0, origin=(5, 100_020))
    np.testing.assert_array_equal(small, big[15:27, 20:29])
    assert not np.array_equal(big, fbm((40, 50), seed=8, period=16.0, origin=(-10, 100_000)))


def test_run_terrain_uses_every_field():
    p = dict(seed=np.int64(5), period=16.0, persistence=0.5, octaves=np.int64(3), lacunarity=2.2, height=40.0,
             noise=np.str_("opensimplex2"), fractal=np.str_("ridged"), warp=0.3)
    terrain = run_terrain(p)
    assert terrain == Terrain(5, 16.0, 0.5, 3, 2.2, 40.0, "opensimplex2", "ridged", 0.3)
    expected = 40.0 * fractal_noise((32, 32), seed=5, period=16.0, octaves=3, persistence=0.5, lacunarity=2.2,
                                    noise="opensimplex2", fractal="ridged", warp=0.3)
    np.testing.assert_array_equal(terrain.heights(0, 0, 32, 32), expected)


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


def test_make_dataset_refuses_new_settings(tmp_path):
    def stop(done, total):
        if done == 1:
            raise RuntimeError

    try:
        make_dataset(CFG, "val", tmp_path / "a", progress=stop)
    except RuntimeError:
        pass
    with pytest.raises(SettingsChanged, match="started"):  # an interrupted split
        make_dataset(replace(CFG, steps=10), "val", tmp_path / "a")
    make_dataset(CFG, "val", tmp_path / "b")
    seen = []
    make_dataset(CFG, "val", tmp_path / "b", progress=lambda done, total: seen.append(done))
    assert seen == []  # a finished split is left as it is
    with pytest.raises(SettingsChanged, match="made"):
        make_dataset(replace(CFG, steps=10), "val", tmp_path / "b")


def test_every_group_file_loads():
    assert "erosion_master/main_256" in group_names()
    assert group_names(TEST_GROUPS) == ["erosion_smoke/smoke", "erosion_smoke/smoke_ood"]
    for root in (None, TEST_GROUPS):
        for name in group_names(root):
            group = load_group(name, root)
            assert group.profile == name and group.snapshots > 1
    with pytest.raises(KeyError):
        load_group("main_256")  # the dataset is part of the name


def test_make_chunked_matches_chunks_simulated_alone(tmp_path):
    out = make_dataset(CFG, "test", tmp_path)
    path = make_chunked(CFG, "test", 16, tmp_path)
    chunked = np.load(path)
    whole, cols, _ = load_dataset(out)
    assert chunked.shape == whole.shape
    row = {k: v[1] for k, v in cols.items()}
    z0 = run_terrain(row).heights(16, 0, 16, 16)
    K_map = map_from_columns(row).factors(16, 0, 16, 16)
    alone = simulate(z0, row["K"], row["D"], row["m"], row["U"], dt=CFG.dt, steps=CFG.steps,
                     snapshot_every=CFG.snapshot_every, K_map=K_map)
    np.testing.assert_array_equal(chunked[1, :, 16:32, 0:16], alone)
    np.testing.assert_array_equal(chunked[:, 0], whole[:, 0])  # same start terrain

