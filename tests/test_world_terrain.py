import numpy as np
import pytest

from erosion_sim.data import Split
from erosion_sim.paths import dataset_dir
from erosion_sim.world.data import block_mean, coarse_heights
from erosion_terrain import Terrain


def test_overlapping_queries_agree():
    terrain = Terrain(seed=7)
    big = terrain.heights(-40, 99_992, 80, 60)
    np.testing.assert_array_equal(terrain.heights(-3, 100_000, 21, 30), big[37:58, 8:38])
    np.testing.assert_array_equal(coarse_heights(terrain, -10, 24_998, 20, 15), block_mean(big))
    assert not np.array_equal(big, Terrain(seed=8).heights(-40, 99_992, 80, 60))


def test_heights_match_the_training_terrain():
    terrain = Terrain(seed=0)
    relief = [np.ptp(terrain.heights(128 * i, 0, 128, 128)) for i in range(20)]
    assert 50 < np.mean(relief) < 90  # the world dataset averages about 68 m on 128² runs


WORLD = dataset_dir("world", "train")


@pytest.mark.skipif(not (WORLD / "params.npz").exists(), reason="needs the world dataset (datagen/README.md)")
def test_matches_the_world_dataset():
    data = Split(WORLD)
    for run in range(0, len(data), 997):
        z0 = data.terrain(run).heights(0, 0, *data.snapshots.shape[-2:]).astype(np.float32)
        np.testing.assert_array_equal(z0, data.snapshots[run, 0])
