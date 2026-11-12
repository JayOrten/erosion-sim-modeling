import io

import matplotlib.pyplot as plt
import numpy as np
from erosion_terrain import fbm

from erosion_sim.world.viewer import TILE, Tiles, shade

from test_world import LEVELS


def tiles():
    return Tiles(LEVELS, steps=3, bf16=False, batch_size=16, keep=2)  # the untrained test levels


SETTINGS = {"seed": 0, "dT": 10_000.0, "K_ref": 2.5e-4, "D_over_K": 1.0, "m": 0.45, "persistence": 0.5}


def test_shade_has_no_per_tile_stretch():
    z = np.random.default_rng(0).normal(size=(10, 10))
    np.testing.assert_allclose(shade(z, 1.0)[2:5, 2:5], shade(z[1:7, 1:7], 1.0)[1:4, 1:4])
    flat = shade(np.zeros((5, 5)), 1.0)
    assert np.allclose(flat, flat[0, 0])


def test_fbm_step_samples_every_nth_cell():
    full = fbm((20, 20), seed=3, period=8.0, origin=(-5, 7))
    np.testing.assert_array_equal(fbm((5, 5), seed=3, period=8.0, origin=(-5, 7), step=4), full[::4, ::4])


def test_tiles_render_and_neighbors_line_up():
    t = tiles()
    for layer in ("eroded", "change"):
        png = t.model_tile(SETTINGS, layer, 0, 0, 0)
        assert plt.imread(io.BytesIO(png)).shape[:2] == (TILE, TILE)
    for zoom in (0, -4):
        assert plt.imread(io.BytesIO(t.start_tile(SETTINGS, zoom, 3, -2))).shape[:2] == (TILE, TILE)
    assert t.report()["done"] == 2 and t.report()["seconds_per_tile"] > 0
    # Heights of adjacent tiles at 1 m and at the coarse level come from one consistent grid.
    world = t.world(SETTINGS)
    for pixel in (1, 4):
        left, right, both = (sum(t.grid(world, pixel, 0, x0, n)) for x0, n in [(0, 8), (8 * pixel, 8), (0, 16)])
        np.testing.assert_allclose(np.hstack([left, right]), both[:8], atol=1e-4)
    assert len(t.worlds) == 1
