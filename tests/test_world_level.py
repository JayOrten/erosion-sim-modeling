import numpy as np
import pytest
import torch
from erosion_terrain import Terrain

from erosion_sim.diffusion import Denoiser, sample
from erosion_sim.normalization import ChangeScaling, ParameterScaling, Scaling
from erosion_sim.world.data import coarse_heights
from erosion_sim.world.level import Level, coarse_context, tent, world_params
from erosion_sim.world.noise import diffusion_noise

WINDOW = 16


def make_level(splits=(5.0,), seed=0, name="fine"):
    torch.manual_seed(0)
    model = Denoiser(5, sigma_data=1.0, fixed_edges=False, channels=(8, 16), blocks=1, emb_dim=16, frequencies=4)
    torch.nn.init.normal_(model.net.out.weight, std=0.1)  # so the output is not all zero
    model.eval()
    scaling = Scaling(ParameterScaling(low=(0.0,) * 5, high=(1.0,) * 5), ChangeScaling(coef=1.0, power=0.0))
    return Level(model, scaling, Terrain(seed=3).heights, seed, name, world_params(K_ref=2e-4, D=2e-4, m=0.4),
                 dT=10_000.0, window=WINDOW, splits=splits, steps=4)


def test_tent():
    w = tent(WINDOW)
    assert w.shape == (WINDOW, WINDOW) and w.max() <= 1 and w.min() > 0
    torch.testing.assert_close(w, w.T)


def test_each_window_is_a_plain_sample():
    level = make_level(splits=())
    ctx = (0, 2, -1)
    y0, x0 = level.origin(ctx)
    out = level._first_phase([ctx])[0]
    noise = torch.as_tensor(diffusion_noise(0, "fine", y0, x0, WINDOW, WINDOW))[None, None]
    expected = sample(level.model, level._inputs([ctx]), level.cond, noise, steps=4)[0, 0]
    torch.testing.assert_close(out[0], expected * level.weight)
    torch.testing.assert_close(out[1], level.weight)


@pytest.mark.parametrize("splits", [(), (5.0,)])
def test_small_query_equals_crop_of_large_query(splits):
    big = make_level(splits).change(-20, 5, 48, 40)
    small = make_level(splits).change(-9, 17, 13, 21)
    # Overlapping windows are summed in a different order, so only float rounding may differ.
    np.testing.assert_allclose(small, big[11:24, 12:33], rtol=0, atol=1e-5 * np.abs(big).max())


def test_query_order_and_cache_do_not_matter():
    level = make_level()
    a = level.change(0, 0, 24, 24)
    level.change(500, -300, 20, 20)
    level.clear_cache()
    np.testing.assert_array_equal(level.change(0, 0, 24, 24), a)


def test_heights_and_dependence_on_seed():
    level = make_level()
    z = level.heights(0, 0, 20, 20)
    np.testing.assert_allclose(z - Terrain(seed=3).heights(0, 0, 20, 20), level.change(0, 0, 20, 20))
    assert np.abs(level.change(0, 0, 20, 20)).max() > 0
    assert not np.allclose(make_level(seed=1).change(0, 0, 20, 20), level.change(0, 0, 20, 20))


def make_two_levels(block=2):
    coarse = make_level(name="coarse")
    coarse.terrain = lambda y0, x0, h, w: coarse_heights(Terrain(seed=3), y0, x0, h, w, block=block)
    torch.manual_seed(1)
    model = Denoiser(5, sigma_data=1.0, fixed_edges=False, in_channels=2, channels=(8, 16), blocks=1, emb_dim=16,
                     frequencies=4)
    torch.nn.init.normal_(model.net.out.weight, std=0.1)
    model.eval()
    scaling = Scaling(ParameterScaling(low=(0.0,) * 5, high=(1.0,) * 5), ChangeScaling(coef=1.0, power=0.0))
    params = world_params(K_ref=2e-4, D=2e-4, m=0.4)
    context = coarse_context(coarse, scaling, block=block)
    fine = Level(model, scaling, Terrain(seed=3).heights, 0, "fine", params, 10_000.0, window=WINDOW, steps=4,
                 context=context)
    return coarse, fine, context


def test_fine_level_reads_the_coarse_change():
    coarse, fine, context = make_two_levels()
    c = context(-16, 8, 16, 16)
    assert c.shape == (1, 16, 16)
    np.testing.assert_allclose(c[0, ::2, ::2], coarse.change(-8, 4, 8, 8))  # change scaling is 1 m per unit
    with pytest.raises(ValueError):
        context(1, 0, 16, 16)
    # The coarse channel changes the fine result.
    no_context = Level(fine.model, fine.scaling, fine.terrain, 0, "fine", world_params(K_ref=2e-4, D=2e-4, m=0.4),
                       10_000.0, window=WINDOW, steps=4, context=lambda y0, x0, h, w: np.zeros((1, h, w)))
    assert not np.allclose(no_context.change(0, 0, 16, 16), fine.change(0, 0, 16, 16))


def test_two_levels_small_query_equals_crop_of_large_query():
    big = make_two_levels()[1].change(-24, 8, 40, 32)
    small = make_two_levels()[1].change(-10, 15, 12, 14)
    # Batching and summing order change float rounding, carried through both levels.
    np.testing.assert_allclose(small, big[14:26, 7:21], rtol=0, atol=1e-5 * np.abs(big).max())


def test_world_params_follow_datagen():
    from erosion_sim.data import Split
    from erosion_sim.paths import dataset_dir

    path = dataset_dir("world", "train")
    if not (path / "params.npz").exists():
        pytest.skip("needs the world dataset")
    params = Split(path).params
    for run in (0, 5, 77):
        p = world_params(params["K_ref"][run], params["D"][run], params["m"][run])
        np.testing.assert_allclose(p["K"][0], params["K"][run], rtol=1e-12)
