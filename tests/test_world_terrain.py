import numpy as np
from erosion_terrain import Terrain

from erosion_sim.data.pairs import block_mean


def test_overlapping_queries_agree():
    terrain = Terrain(seed=7)
    big = terrain.heights(-40, 99_992, 80, 60)
    np.testing.assert_array_equal(terrain.heights(-3, 100_000, 21, 30), big[37:58, 8:38])
    np.testing.assert_array_equal(block_mean(terrain.heights(-40, 99_992, 80, 60), 4), block_mean(big, 4))
    assert not np.array_equal(big, Terrain(seed=8).heights(-40, 99_992, 80, 60))


def test_heights_match_the_training_terrain():
    terrain = Terrain(seed=0)
    relief = [np.ptp(terrain.heights(128 * i, 0, 128, 128)) for i in range(20)]
    assert 50 < np.mean(relief) < 90  # the world dataset averages about 68 m on 128² runs
