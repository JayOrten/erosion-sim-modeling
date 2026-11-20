import numpy as np
import pytest
import torch
from erosion_terrain import Terrain
from world_helpers import SCALING, level

from erosion_sim.objectives.sampling import generate
from erosion_sim.world.level import Level, parent_context, tent, world_params
from erosion_sim.world.noise import diffusion_noise

WINDOW = 16
PARAMS = world_params(K_ref=2e-4, D=2e-4, m=0.4)


def make_level(objective="edm", splits=(5.0,), seed=0, **changes):
    model, scaling, cfg = level(1, WINDOW, objective=objective, **changes)
    return Level(model, scaling, cfg, Terrain(seed=3).heights, seed, "fine", PARAMS, 10_000.0, splits, steps=4)


def test_tent():
    w = tent(WINDOW)
    assert w.shape == (WINDOW, WINDOW) and w.max() <= 1 and w.min() > 0
    torch.testing.assert_close(w, w.T)


@pytest.mark.parametrize("objective", ["edm", "flow_matching", "regression"])
def test_each_window_is_a_plain_sample(objective):
    lev = make_level(objective, splits=())
    ctx = (0, 2, -1)
    y0, x0 = lev.origin(ctx)
    out = lev._first_phase([ctx])[0]
    noise = torch.as_tensor(diffusion_noise(0, "fine", y0, x0, WINDOW, WINDOW))[None, None]
    expected = generate(lev.model, lev._inputs([ctx]), lev.cond, noise, 4)[0, 0]
    got = lev.to_meters((out[0] / out[1]).double().numpy())
    np.testing.assert_allclose(got, expected.double().numpy(), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("objective", ["edm", "flow_matching", "regression"])
@pytest.mark.parametrize("splits", [(), (5.0,)])
def test_small_query_equals_crop_of_large_query(objective, splits):
    big = make_level(objective, splits).change(-20, 5, 48, 40)
    small = make_level(objective, splits).change(-9, 17, 13, 21)
    # Overlapping windows are summed in a different order, so only float rounding may differ.
    np.testing.assert_allclose(small, big[11:24, 12:33], rtol=0, atol=1e-5 * np.abs(big).max())


def test_query_order_and_cache_do_not_matter():
    lev = make_level()
    a = lev.change(0, 0, 24, 24)
    lev.change(500, -300, 20, 20)
    lev.clear_cache()
    np.testing.assert_array_equal(lev.change(0, 0, 24, 24), a)


def test_heights_and_dependence_on_seed():
    lev = make_level()
    z = lev.heights(0, 0, 20, 20)
    np.testing.assert_allclose(z - Terrain(seed=3).heights(0, 0, 20, 20), lev.change(0, 0, 20, 20))
    assert np.abs(lev.change(0, 0, 20, 20)).max() > 0
    assert not np.allclose(make_level(seed=1).change(0, 0, 20, 20), lev.change(0, 0, 20, 20))


def test_map_input_changes_the_result():
    plain = make_level(inputs=("map",))
    soft = Level(plain.model, plain.scaling, plain.cfg, plain.terrain, 0, "fine", PARAMS, 10_000.0, steps=4,
                 log_map=lambda y0, x0, h, w: np.full((h, w), 0.5))
    assert not np.allclose(plain.change(0, 0, 16, 16), soft.change(0, 0, 16, 16))


def test_context_reads_the_parent_change():
    parent = make_level()
    model, scaling, cfg = level(1, WINDOW, context_cell=2, seed=1)
    context = parent_context(parent, SCALING, 2)
    child = Level(model, scaling, cfg, Terrain(seed=3).heights, 0, "child", PARAMS, 10_000.0, steps=4, context=context)
    c = context(-16, 8, 16, 16)
    np.testing.assert_allclose(c[::2, ::2], parent.change(-8, 4, 8, 8))  # change scaling is 1 m per unit
    with pytest.raises(ValueError):
        context(1, 0, 16, 16)
    assert np.isfinite(child.change(0, 0, 16, 16)).all()
    with pytest.raises(ValueError):  # trained with context, given none
        Level(model, scaling, cfg, Terrain(seed=3).heights, 0, "child", PARAMS, 10_000.0)
