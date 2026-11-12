from dataclasses import replace

import numpy as np
import pytest
import torch

from erosion_sim.config import get_config
from erosion_sim.normalization import ChangeScaling, ParameterScaling, Scaling
from erosion_sim.train import build_model
from erosion_sim.world.world import World


def level(name, seed):
    cfg = replace(get_config("smoke", "diffusion"), fixed_edges=False, level=name, crop=16, block=2)
    scaling = Scaling(ParameterScaling(low=(0.0,) * 5, high=(1.0,) * 5), ChangeScaling(coef=1.0, power=0.0))
    torch.manual_seed(seed)
    model = build_model(cfg).eval()
    torch.nn.init.normal_(model.net.out.weight, std=0.1)  # so the output is not all zero
    return model, scaling, cfg


LEVELS = (level("coarse", 0), level("fine", 1))


def make_world(seed=0, **kwargs):
    return World(seed, 10_000.0, *LEVELS, coarse_window=16, steps=4, **kwargs)


def tol(a):
    # Overlapping windows are summed in whatever order they were computed, and batched with whichever
    # windows the query needs, so float rounding may differ. Two chained levels carry it through.
    return 1e-5 * np.abs(a).max()


def test_regions_in_any_order_agree():
    a = make_world()
    first = a.get(0, 0, 24, 24)
    second = a.get(30, -10, 20, 20)
    b = make_world()
    np.testing.assert_allclose(b.get(30, -10, 20, 20), second, rtol=0, atol=tol(second))
    np.testing.assert_allclose(b.get(0, 0, 24, 24), first, rtol=0, atol=tol(first))


def test_clearing_the_cache_changes_nothing():
    world = make_world()
    first = world.get(-8, 4, 24, 24)
    world.clear_cache()
    np.testing.assert_array_equal(world.get(-8, 4, 24, 24), first)


def test_small_query_equals_crop_of_large_query():
    big = make_world().get(-16, -16, 48, 48)
    small = make_world().get(-3, 5, 17, 9)
    np.testing.assert_allclose(small, big[13:30, 21:30], rtol=0, atol=tol(big))


def test_far_from_the_origin():
    world = make_world()
    z = world.get(100_000, -100_000, 20, 20)
    assert np.all(np.isfinite(z))
    np.testing.assert_allclose(z - world.terrain.heights(100_000, -100_000, 20, 20),
                               world.change(100_000, -100_000, 20, 20))
    assert np.abs(world.change(100_000, -100_000, 20, 20)).max() > 0


def test_seed_and_dT_change_the_world():
    z = make_world().get(0, 0, 16, 16)
    assert not np.allclose(make_world(seed=1).get(0, 0, 16, 16), z)
    other = World(0, 20_000.0, *LEVELS, coarse_window=16, steps=4)
    np.testing.assert_array_equal(other.terrain.heights(0, 0, 16, 16), make_world().terrain.heights(0, 0, 16, 16))
    assert not np.allclose(other.get(0, 0, 16, 16), z)


def test_levels_must_match():
    with pytest.raises(ValueError):
        World(0, 10_000.0, LEVELS[1], LEVELS[0])
