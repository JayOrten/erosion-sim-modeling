import numpy as np
import pytest
import torch

from erosion_sim.data import Normalizer
from erosion_sim.world.data import ErosionCoarse, ErosionCrops, block_mean, dihedral, upsample

from conftest import SIZE


@pytest.fixture
def normalizer(dataset_path):
    return Normalizer.fit(dataset_path, height=64.0)


def test_block_mean_upsample_and_dihedral():
    z = np.arange(16.0).reshape(4, 4)
    np.testing.assert_array_equal(block_mean(z, 2), [[2.5, 4.5], [10.5, 12.5]])
    np.testing.assert_array_equal(block_mean(upsample(z, 2), 2), z)
    views = [dihedral(z, k) for k in range(8)]
    assert len({v.tobytes() for v in views}) == 8


def test_crops_shapes_and_round_trip(dataset_path, normalizer):
    data = ErosionCrops(dataset_path, normalizer, crop=8, block=2)
    item = data[5]
    assert item["x"].shape == (2, 8, 8) and item["y"].shape == (1, 8, 8)
    # The change in m comes back from the target.
    dz = item["y"][0].double() * item["relief"] * item["scale"]
    torch.testing.assert_close(dz, (item["z_b"] - item["z_a"]).double(), atol=1e-4, rtol=0)
    # The coarse channel is the block mean of the target, repeated to 1 m cells.
    coarse = upsample(block_mean(item["y"][0].numpy(), 2), 2)
    np.testing.assert_allclose(item["x"][1].numpy(), coarse, atol=1e-6)


def test_crops_come_from_the_run(dataset_path, normalizer):
    data = ErosionCrops(dataset_path, normalizer, crop=8, block=2, augment=False)
    for index in range(0, len(data), 7):
        item = data[index]
        full = data.snapshots[item["run"], item["start"]]
        hits = [(y, x) for y in range(0, SIZE - 7, 2) for x in range(0, SIZE - 7, 2)
                if np.array_equal(full[y:y + 8, x:x + 8], item["z_a"].numpy())]
        assert hits, index


def test_items_are_reproducible_and_vary(dataset_path, normalizer):
    data = ErosionCrops(dataset_path, normalizer, crop=8, block=2)
    torch.testing.assert_close(data[3]["x"], data[np.int64(3)]["x"])
    other = ErosionCrops(dataset_path, normalizer, crop=8, block=2, seed=1)
    assert any(not torch.equal(data[i]["z_a"], other[i]["z_a"]) for i in range(10))


def test_coarse_items(dataset_path):
    normalizer = Normalizer.fit(dataset_path, height=64.0, block=2)
    data = ErosionCoarse(dataset_path, normalizer, block=2, augment=False)
    item = data[4]
    assert item["x"].shape == item["y"].shape == (1, SIZE // 2, SIZE // 2)
    run, a = item["run"], item["start"]
    np.testing.assert_allclose(item["z_a"].numpy(), block_mean(np.asarray(data.snapshots[run, a], np.float64), 2),
                               rtol=1e-6)
    # Fine crops of the same pair average to the coarse target, where they overlap.
    fine = ErosionCrops(dataset_path, Normalizer.fit(dataset_path, height=64.0), crop=SIZE, block=2,
                        augment=False)[4]
    np.testing.assert_allclose(block_mean(fine["y"][0].numpy() * fine["scale"], 2),
                               item["y"][0].numpy() * item["scale"], atol=1e-5)


def test_needs_a_fixed_height(dataset_path):
    with pytest.raises(ValueError):
        ErosionCrops(dataset_path, Normalizer.fit(dataset_path))
