import numpy as np
import pytest
import torch

from erosion_sim.world.data import ErosionCoarse, ErosionCrops, block_mean, dihedral, upsample

from conftest import SIZE


def test_block_mean_upsample_and_dihedral():
    z = np.arange(16.0).reshape(4, 4)
    np.testing.assert_array_equal(block_mean(z, 2), [[2.5, 4.5], [10.5, 12.5]])
    np.testing.assert_array_equal(block_mean(upsample(z, 2), 2), z)
    views = [dihedral(z, k) for k in range(8)]
    assert len({v.tobytes() for v in views}) == 8


def test_crops_shapes_and_round_trip(dataset, scaling):
    data = ErosionCrops(dataset, scaling, crop=8, block=2)
    item = data[5]
    assert item["x"].shape == (2, 8, 8) and item["y"].shape == (1, 8, 8)
    # The change in m comes back from the target.
    dT = dataset.times[item["end"]] - dataset.times[item["start"]]
    dz = scaling.change.from_model(item["y"][0].double(), dataset.params["K_ref"][item["run"]], dT)
    torch.testing.assert_close(dz, (item["z_b"] - item["z_a"]).double(), atol=1e-4, rtol=0)
    # The coarse channel is the block mean of the target, repeated to 1 m cells.
    coarse = upsample(block_mean(item["y"][0].numpy(), 2), 2)
    np.testing.assert_allclose(item["x"][1].numpy(), coarse, atol=1e-6)


def test_crops_come_from_the_run(dataset, scaling):
    data = ErosionCrops(dataset, scaling, crop=8, block=2, augment=False)
    for index in range(0, len(data), 7):
        item = data[index]
        full = data.snapshots[item["run"], item["start"]]
        hits = [(y, x) for y in range(0, SIZE - 7, 2) for x in range(0, SIZE - 7, 2)
                if np.array_equal(full[y:y + 8, x:x + 8], item["z_a"].numpy())]
        assert hits, index


def test_items_are_reproducible_and_vary(dataset, scaling):
    data = ErosionCrops(dataset, scaling, crop=8, block=2)
    torch.testing.assert_close(data[3]["x"], data[np.int64(3)]["x"])
    other = ErosionCrops(dataset, scaling, crop=8, block=2, seed=1)
    assert any(not torch.equal(data[i]["z_a"], other[i]["z_a"]) for i in range(10))


def test_coarse_items(dataset, scaling):
    data = ErosionCoarse(dataset, scaling, block=2, augment=False)
    item = data[4]
    assert item["x"].shape == item["y"].shape == (1, SIZE // 2, SIZE // 2)
    run, a = item["run"], item["start"]
    np.testing.assert_allclose(item["z_a"].numpy(), block_mean(np.asarray(data.snapshots[run, a], np.float64), 2),
                               rtol=1e-6)
    # Fine crops of the same pair average to the coarse target, where they overlap.
    fine = ErosionCrops(dataset, scaling, crop=SIZE, block=2, augment=False)[4]
    np.testing.assert_allclose(block_mean(fine["y"][0].numpy(), 2), item["y"][0].numpy(), atol=1e-6)


def test_margin_keeps_crops_away_from_the_edge(dataset, scaling):
    fine = ErosionCrops(dataset, scaling, crop=8, block=2, margin=2, augment=False)
    coarse = ErosionCoarse(dataset, scaling, block=2, crop=4, margin=2, augment=False)
    for data, size in [(fine, 8), (coarse, 8)]:
        corners = {(data[i]["y0"], data[i]["x0"]) for i in range(len(data))}
        assert all(2 <= y <= SIZE - 2 - size and 2 <= x <= SIZE - 2 - size for y, x in corners)
        assert all(y % 2 == 0 and x % 2 == 0 for y, x in corners)
        assert len(corners) == 9  # y0 and x0 each 2, 4, or 6
    item = coarse[3]
    assert item["x"].shape == (1, 4, 4)
    y0, x0 = item["y0"], item["x0"]
    full = np.asarray(coarse.snapshots[item["run"], item["start"]], np.float64)
    np.testing.assert_allclose(item["z_a"].numpy(), block_mean(full[y0:y0 + 8, x0:x0 + 8], 2), rtol=1e-6)
    with pytest.raises(ValueError):
        ErosionCrops(dataset, scaling, crop=8, block=2, margin=SIZE)[0]
    with pytest.raises(ValueError):
        ErosionCoarse(dataset, scaling, block=2, margin=2)


def test_repeats_give_each_pair_several_crops(dataset, scaling):
    once = ErosionCrops(dataset, scaling, crop=8, block=2)
    many = ErosionCrops(dataset, scaling, crop=8, block=2, repeats=3)
    assert len(many) == 3 * len(once)
    for k in range(3):
        assert many.locate(6 + k) == once.locate(2)  # items 6, 7, 8 share pair 2
    assert len({(many[i]["y0"], many[i]["x0"]) for i in range(6, 9)} | {(once[2]["y0"], once[2]["x0"])}) > 1
    np.testing.assert_array_equal(many.elapsed(np.arange(6, 9)), once.elapsed(2))
