import numpy as np

from erosion_sim.world.noise import TILE, diffusion_noise


def test_overlapping_queries_agree():
    # A large region across tile boundaries and negative coordinates, and a small one inside it.
    big = diffusion_noise(0, "fine", -100, 2 * TILE - 7, 150, 130)
    small = diffusion_noise(0, "fine", -37, 2 * TILE + 20, 41, 66)
    np.testing.assert_array_equal(small, big[63:104, 27:93])


def test_query_order_does_not_matter():
    a = diffusion_noise(3, "coarse", 10, 10, 20, 20)
    diffusion_noise(3, "coarse", 1_000_000, -5, 64, 64)
    np.testing.assert_array_equal(a, diffusion_noise(3, "coarse", 10, 10, 20, 20))


def test_seed_and_level_give_different_fields():
    a = diffusion_noise(0, "fine", 0, 0, 32, 32)
    assert not np.allclose(a, diffusion_noise(1, "fine", 0, 0, 32, 32))
    assert not np.allclose(a, diffusion_noise(0, "coarse", 0, 0, 32, 32))


def test_unit_gaussian():
    z = diffusion_noise(0, "fine", -200, -200, 400, 400).astype(np.float64)
    assert abs(z.mean()) < 0.01
    assert abs(z.std() - 1) < 0.01
    # Neighbors are independent, including across tile edges.
    assert abs(np.corrcoef(z[:, :-1].ravel(), z[:, 1:].ravel())[0, 1]) < 0.01
