from dataclasses import replace

import matplotlib
import numpy as np

from erosion_sim.evaluate import error_metrics
from erosion_sim.world.data import block_mean
from erosion_sim.world.evaluate import (by_elapsed, chunked_vs_whole, coarse_under, edge_profile, headline,
                                        level_pairs, order_invariance, query_times, seam_summary, show_examples,
                                        smooth_upsample, speed_comparison, window_disagreement)

from test_world import LEVELS, make_world

matplotlib.use("Agg")


def test_smooth_upsample():
    np.testing.assert_allclose(smooth_upsample(np.full((2, 3, 3), 5.0), 4), 5.0)
    z = smooth_upsample(np.arange(9.0).reshape(3, 3), 2)
    assert z.shape == (6, 6) and np.all(np.diff(z, axis=1) >= 0)


def test_level_pairs_and_plots(data_root):
    out = level_pairs(*LEVELS, count=6, steps=3, data_root=data_root, show=False)
    coarse, fine = out["coarse"], out["fine"]
    assert coarse["coarse model"].shape == coarse["z_a"].shape == (6, 8, 8)
    assert fine["coarse + fine models"].shape == fine["z_b"].shape == (6, 16, 16)
    # The coarse level's runs match the fine crops' runs: same pairs.
    np.testing.assert_array_equal(coarse["run"], fine["run"])
    assert set(error_metrics(fine, show=False)) == set(fine["methods"])
    assert fine["kind"]["coarse model only"] == "baseline"
    table = headline(fine, show=False)
    assert set(table) == {*fine["methods"], "simulator"}
    methods = ["coarse + fine models", "coarse model only"]
    show_examples(fine, methods, examples=(0, 1))
    result = by_elapsed(fine, methods, bins=2)
    assert len(result["dT"]) == 2 and len(result["coarse model only"]["rmse"]) == 2


def test_level_pairs_away_from_the_edge(data_root):
    (cm, cn, ccfg), (fm, fn, fcfg) = LEVELS
    coarse = (cm, cn, replace(ccfg, coarse_crop=4, margin=2))
    fine = (fm, fn, replace(fcfg, crop=8, margin=2))
    out = level_pairs(coarse, fine, count=6, steps=2, data_root=data_root, show=False)
    c, f = out["coarse"], out["fine"]
    assert c["z_a"].shape == (6, 6, 6)  # the 16² runs' interior, 2 cells in, at 2-cell blocks
    assert f["z_a"].shape == (6, 8, 8)
    np.testing.assert_array_equal(c["run"], f["run"])
    np.testing.assert_array_equal(c["start"], f["start"])
    # Each fine crop's block means are a piece of its coarse item, at the offset coarse_under uses.
    for i in range(6):
        piece = block_mean(f["z_a"][i], 2)
        hits = [(y, x) for y in range(3) for x in range(3)
                if np.allclose(c["z_a"][i][y:y + 4, x:x + 4], piece, atol=1e-4)]
        assert hits, i


def test_coarse_under():
    coarse = np.arange(2 * 6 * 6, dtype=float).reshape(2, 6, 6)
    items = [{"y0": 2, "x0": 6}, {"y0": 4, "x0": 2}]  # fine cells; the coarse maps start at fine cell 2
    out = coarse_under(coarse, items, offset=2, block=2, size=3)
    np.testing.assert_array_equal(out[0], coarse[0, 0:3, 2:5])
    np.testing.assert_array_equal(out[1], coarse[1, 1:4, 0:3])


def test_window_disagreement_and_seams():
    world = make_world()
    two = window_disagreement(world.fine, windows=3, show=False)["phases"]
    assert len(two) == 2 and all(r["rms"] > 0 for r in two)
    seams = {"T = 2": {level: window_disagreement(getattr(world, level), windows=2, show=False)
                       for level in ("coarse", "fine")}}
    assert set(seam_summary(seams, {"coarse": 1.0, "fine": 0.5})["T = 2"]) == {"coarse", "fine"}


def test_edge_profile():
    result = edge_profile(make_world().coarse, windows=2, show=False)
    assert len(result["distance"]) == 8 and np.all(result["mean_abs_change"] >= 0)


def test_order_invariance_and_timing():
    result = order_invariance(make_world, regions=4, size=12, spread=100, show=False)
    assert result["max_difference"] <= 1e-5 * result["max_change"]
    times = query_times(make_world, sizes=(8, 16), show=False)["sizes"]
    assert all(t["repeat"] < t["first"] for t in times)
    simulator = {"runs": [{"size": 8, "dT": 1e4, "seconds": 1.0}, {"size": 16, "dT": 1e4, "seconds": 2.0}]}
    matplotlib.use("Agg")
    rows = speed_comparison({"test": {"sizes": times}}, simulator, show=True)["rows"]
    assert [r["size"] for r in rows] == [8, 16] and rows[1]["simulator"][1e4] == 2.0


def test_chunked_vs_whole():
    result = chunked_vs_whole(make_world, 8, -16, 32, chunk=16, show=False)
    assert result["chunked"].shape == (32, 32) and result["grid"] == [16]
    # Float rounding only. The untrained test model makes huge changes, which magnify it.
    whole = make_world()
    assert result["max_difference"] <= 1e-4 * np.abs(whole.change(8, -16, 32, 32)).max()
