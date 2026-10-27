import numpy as np

from erosion_sim.hydrology import drainage_area, pits


def test_plane_drains_straight_downhill():
    rows, cols = 8, 10
    z = np.tile(np.arange(cols, dtype=float), (rows, 1))  # rises to the east
    area = drainage_area(z)
    # Each interior row collects every interior node east of the west edge, then the edge node adds itself.
    np.testing.assert_allclose(area[1:-1, 1], cols - 2)
    np.testing.assert_allclose(area[1:-1, 0], cols - 1)
    assert not pits(z).any()


def test_pit_stops_water():
    z = np.tile(np.arange(10, dtype=float), (8, 1))
    z[4, 5] = -1.0  # a hole
    assert pits(z)[4, 5]
    assert drainage_area(z)[4, 4] < drainage_area(np.tile(np.arange(10, dtype=float), (8, 1)))[4, 4]
