import numpy as np
import pytest
from world_helpers import THREE, TWO, level

from erosion_sim.world.checks import (
    chunked_vs_whole,
    order_invariance,
    query_times,
    window_disagreement,
)
from erosion_sim.world.world import World, check_stack


def make_world(levels=TWO, seed=0, dT=10_000.0, **kwargs):
    return World(seed, dT, levels, steps=4, **kwargs)


def tol(a):
    # Overlapping windows are summed and batched in whatever order the query needs, so float rounding
    # may differ, carried through every level.
    return 1e-5 * np.abs(a).max()


@pytest.mark.parametrize("levels", [TWO, THREE])
def test_regions_in_any_order_agree(levels):
    a = make_world(levels)
    first = a.get(0, 0, 24, 24)
    second = a.get(30, -10, 20, 20)
    b = make_world(levels)
    np.testing.assert_allclose(b.get(30, -10, 20, 20), second, rtol=0, atol=tol(second))
    np.testing.assert_allclose(b.get(0, 0, 24, 24), first, rtol=0, atol=tol(first))


def test_clearing_the_cache_changes_nothing():
    world = make_world(THREE)
    first = world.get(-8, 4, 24, 24)
    world.clear_cache()
    np.testing.assert_array_equal(world.get(-8, 4, 24, 24), first)


def test_small_query_equals_crop_of_large_query():
    big = make_world().get(-16, -16, 48, 48)
    small = make_world().get(-3, 5, 17, 9)
    np.testing.assert_allclose(small, big[13:30, 21:30], rtol=0, atol=tol(big))


def test_far_from_the_origin_and_heights():
    world = make_world()
    z = world.get(100_000, -100_000, 20, 20)
    np.testing.assert_allclose(z - world.terrain.heights(100_000, -100_000, 20, 20),
                               world.change(100_000, -100_000, 20, 20))
    assert np.abs(world.change(100_000, -100_000, 20, 20)).max() > 0


def test_seed_and_dT_change_the_world():
    z = make_world().get(0, 0, 16, 16)
    assert not np.allclose(make_world(seed=1).get(0, 0, 16, 16), z)
    assert not np.allclose(make_world(dT=20_000.0).get(0, 0, 16, 16), z)


def test_stacks_must_fit_together():
    with pytest.raises(ValueError, match="without context"):
        check_stack([TWO[1][2]])
    with pytest.raises(ValueError, match="divide"):
        check_stack([level(1, 8)[2], level(2, 8, context_cell=2)[2]])
    with pytest.raises(ValueError, match="context"):
        check_stack([THREE[0][2], THREE[2][2]])  # level trained under 2 m cells placed under 4 m cells
    with pytest.raises(ValueError, match="whole runs"):
        check_stack([level(1, None)[2]])


def test_checks():
    seams = window_disagreement(make_world().levels[1], windows=3)["phases"]
    assert len(seams) == 2 and all(p["rms"] >= 0 for p in seams)
    assert order_invariance(make_world, regions=3, size=12, spread=100)["max_difference"] < 1e-3
    assert chunked_vs_whole(make_world, 4, -8, 32, chunk=16)["max_difference"] < 1e-3
    assert query_times(make_world, sizes=(8,))["sizes"][0]["first"] > 0
