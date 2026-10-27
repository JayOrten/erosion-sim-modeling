import math

import torch

from erosion_sim.data import Normalizer
from erosion_sim.diffusion import SIGMA_MAX, SIGMA_MIN, Denoiser

COND = 5


def small_denoiser(sigma_data=0.7):
    return Denoiser(COND, sigma_data=sigma_data, channels=(8, 16), blocks=1, emb_dim=16, frequencies=4)


def inputs(batch=3, size=16):
    torch.manual_seed(0)
    return torch.rand(batch, 1, size, size), torch.rand(batch, COND), torch.randn(batch, 1, size, size)


def test_output_shape_and_zero_edges():
    model = small_denoiser()
    torch.nn.init.normal_(model.net.out.weight)
    x, cond, noisy = inputs()
    y = model(x, cond, noisy, torch.tensor([0.1, 1.0, 10.0]))
    assert y.shape == (3, 1, 16, 16)
    for edge in (y[..., 0, :], y[..., -1, :], y[..., :, 0], y[..., :, -1]):
        assert torch.all(edge == 0)
    assert y[..., 1:-1, 1:-1].abs().sum() > 0


def test_small_noise_returns_the_input():
    model = small_denoiser()
    torch.nn.init.normal_(model.net.out.weight)
    x, cond, noisy = inputs()
    y = model(x, cond, noisy, torch.full((3,), 1e-6))
    torch.testing.assert_close(y[..., 1:-1, 1:-1], noisy[..., 1:-1, 1:-1], atol=1e-4, rtol=0)


def test_large_noise_returns_the_scaled_network_output():
    model = small_denoiser(sigma_data=0.7)
    torch.nn.init.normal_(model.net.out.weight)
    x, cond, noisy = inputs()
    sigma = torch.full((3,), 1e4)
    y = model(x, cond, noisy, sigma)
    # c_skip -> 0 and c_out -> sigma_data, so D -> sigma_data * F(c_in * noisy).
    c_in = 1 / math.sqrt(1e4**2 + 0.7**2)
    noisy = noisy.clone()
    noisy[..., [0, -1], :] = 0  # the denoiser zeroes edge nodes before the U-Net sees them
    noisy[..., :, [0, -1]] = 0
    net_cond = torch.cat([cond, model.noise_feature(sigma)[:, None]], dim=1)
    f = model.net(torch.cat([x, c_in * noisy], dim=1), net_cond)
    torch.testing.assert_close(y, 0.7 * f, atol=1e-4, rtol=1e-3)


def test_untrained_denoiser_only_shrinks_the_input():
    model = small_denoiser(sigma_data=0.7)  # the U-Net's output layer starts at zero
    x, cond, noisy = inputs()
    sigma = torch.tensor([0.1, 1.0, 10.0])
    y = model(x, cond, noisy, sigma)
    c_skip = (0.7**2 / (sigma**2 + 0.7**2)).reshape(-1, 1, 1, 1)
    torch.testing.assert_close(y[..., 1:-1, 1:-1], (c_skip * noisy)[..., 1:-1, 1:-1])


def test_noise_feature_maps_the_sampler_range_to_unit_interval():
    model = small_denoiser(sigma_data=0.7)
    ends = model.noise_feature(torch.tensor([SIGMA_MIN * 0.7, SIGMA_MAX * 0.7]))
    torch.testing.assert_close(ends, torch.tensor([0.0, 1.0]))


def test_noise_level_changes_the_output():
    model = small_denoiser()
    torch.nn.init.normal_(model.net.out.weight)
    x, cond, noisy = inputs(batch=1)
    a = model(x, cond, noisy, torch.tensor([1.0]))
    b = model(x, cond, noisy * 1.0, torch.tensor([1.0 + 1e-3]))
    assert not torch.allclose(a, b)


def test_normalizer_measures_sigma_data(dataset_path):
    fitted = Normalizer.fit(dataset_path, scale="fitted")
    relief = Normalizer.fit(dataset_path, scale="relief")
    # The fitted scale divides out the typical size of the change, so its RMS is near 1.
    assert 0.5 < fitted.sigma_data < 2.0
    assert 0 < relief.sigma_data < fitted.sigma_data
