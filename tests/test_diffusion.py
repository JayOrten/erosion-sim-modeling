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


def test_loss_weight_gives_unit_loss_for_an_untrained_denoiser():
    # The U-Net's effective target has unit variance at every sigma. An untrained U-Net outputs zero,
    # so its weighted loss is about 1 whatever the noise level.
    from erosion_sim.diffusion import denoising_loss

    model = small_denoiser(sigma_data=0.7)
    torch.manual_seed(0)
    y = 0.7 * torch.randn(8, 1, 64, 64)
    y[..., [0, -1], :] = 0
    y[..., :, [0, -1]] = 0
    interior = (62 / 64) ** 2  # edges are exact, so they add zero error
    for sigma in (0.01, 0.7, 50.0):
        loss, _ = denoising_loss(model, torch.rand(8, 1, 64, 64), torch.rand(8, COND), y,
                                 torch.full((8,), sigma), torch.randn(8, 1, 64, 64))
        assert abs(loss.mean().item() / interior - 1) < 0.05


def test_smoke_diffusion_training(data_root, tmp_path):
    from erosion_sim.config import get_config
    from erosion_sim.train import load_checkpoint, train

    out = tmp_path / "diffusion-smoke.pt"
    history = train(get_config("smoke", "diffusion"), data_root=data_root, show=False, out=out)
    assert len(history["val_loss"]) == 2
    assert all(math.isfinite(v) for v in history["train_loss"] + history["val_loss"])

    model, normalizer, cfg, _ = load_checkpoint(out, device="cpu")
    assert isinstance(model, Denoiser) and cfg.objective == "diffusion"
    assert model.sigma_data == normalizer.sigma_data != 1.0


def test_validation_noise_is_fixed(data_root, tmp_path):
    from erosion_sim.config import get_config
    from erosion_sim.train import train

    # With a learning rate of zero the model never changes, so fixed noise gives identical losses.
    history = train(get_config("smoke", "diffusion"), data_root=data_root, show=False, out=tmp_path / "d.pt",
                    lr=0.0, warmup_steps=0, steps=3, eval_every=1)
    assert history["val_loss"][0] == history["val_loss"][1] == history["val_loss"][2]


def test_diffusion_resume_matches_an_uninterrupted_run(data_root, tmp_path, monkeypatch):
    import numpy as np
    import pytest

    import erosion_sim.train as train_module
    from erosion_sim.config import get_config

    args = dict(data_root=data_root, show=False, out=tmp_path / "diffusion-smoke.pt", steps=8, eval_every=4,
                log_every=4)
    full = train_module.train(get_config("smoke", "diffusion"), **args)

    # Interrupt at the second evaluation (step 8), so the last checkpoint is from step 4.
    evaluate, calls = train_module.evaluate_loss, []

    def interrupted(*a, **k):
        calls.append(1)
        if len(calls) == 2:
            raise KeyboardInterrupt
        return evaluate(*a, **k)

    monkeypatch.setattr(train_module, "evaluate_loss", interrupted)
    with pytest.raises(KeyboardInterrupt):
        train_module.train(get_config("smoke", "diffusion"), **args)
    monkeypatch.setattr(train_module, "evaluate_loss", evaluate)

    # Training noise depends only on (seed, step), so the resumed run matches exactly.
    resumed = train_module.train(get_config("smoke", "diffusion"), **args, resume=True)
    assert resumed["eval_step"] == full["eval_step"] == [4, 8]
    np.testing.assert_allclose(resumed["train_loss"], full["train_loss"], rtol=1e-5)
    np.testing.assert_allclose(resumed["val_loss"], full["val_loss"], rtol=1e-5)


class GaussianDenoiser(Denoiser):
    """The ideal denoiser when the data is Gaussian with standard deviation sigma_data. It counts calls."""

    def __init__(self, sigma_data):
        super().__init__(COND, sigma_data=sigma_data, channels=(8,), emb_dim=8, frequencies=2)
        self.calls = 0

    def forward(self, x, cond, noisy, sigma):
        self.calls += 1
        s = sigma.reshape(-1, 1, 1, 1)
        return self.sigma_data**2 / (s**2 + self.sigma_data**2) * noisy


def test_sigma_steps():
    from erosion_sim.diffusion import sigma_steps

    s = sigma_steps(10, 0.7)
    assert len(s) == 11 and s[-1] == 0
    torch.testing.assert_close(s[[0, 9]], torch.tensor([SIGMA_MAX * 0.7, SIGMA_MIN * 0.7], dtype=torch.float64))
    assert torch.all(s[1:] < s[:-1])


def test_heun_sampler_solves_the_gaussian_case():
    from erosion_sim.diffusion import sample

    # For Gaussian data the ODE has the exact solution y ∝ sqrt(sigma^2 + sigma_data^2).
    sd = 0.7
    noise = torch.ones(1, 1, 16, 16)
    exact = sd * SIGMA_MAX * sd / math.sqrt((SIGMA_MAX * sd) ** 2 + sd**2)
    errors = []
    for steps in (32, 64):
        model = GaussianDenoiser(sd)
        y = sample(model, torch.zeros(1, 1, 16, 16), torch.zeros(1, COND), noise, steps)
        errors.append(abs(y[0, 0, 5, 5].item() / exact - 1))
        assert model.calls == 2 * steps - 1
    # Heun's method is second order: doubling the steps cuts the error about 4 times.
    assert errors[1] < 0.005
    assert 3 < errors[0] / errors[1] < 5.5


def test_initial_noise_is_keyed():
    from erosion_sim.diffusion import initial_noise

    a = initial_noise([(0, 0), (0, 1), (1, 0)], (16, 16))
    b = initial_noise([(0, 1)], (16, 16))
    assert a.shape == (3, 1, 16, 16)
    torch.testing.assert_close(a[1], b[0])
    assert not torch.allclose(a[0], a[1]) and not torch.allclose(a[0], a[2])
    assert torch.all(a[..., 0, :] == 0) and torch.all(a[..., :, -1] == 0)


def test_predict_and_rollout_with_a_diffusion_model(data_root, tmp_path):
    import numpy as np

    from erosion_sim.config import get_config
    from erosion_sim.data import load_params
    from erosion_sim.generate import predict, rollout
    from erosion_sim.train import load_checkpoint, train

    out = tmp_path / "diffusion-smoke.pt"
    train(get_config("smoke", "diffusion"), data_root=data_root, show=False, out=out)
    model, normalizer, _, _ = load_checkpoint(out, device="cpu")
    params, _ = load_params(data_root / "val")
    z0 = np.load(data_root / "val" / "snapshots.npy")[0, 0].astype(np.float64)

    three = predict(model, normalizer, z0, params, 0, 1000.0, samples=3, seed=5, steps=4)
    assert three.shape == (3, *z0.shape) and np.isfinite(three).all()
    # Sample i depends only on its key, not on how many samples were drawn.
    one = predict(model, normalizer, z0, params, 0, 1000.0, samples=1, seed=5, steps=4)
    np.testing.assert_allclose(one[0], three[0], rtol=1e-5, atol=1e-5)
    assert not np.allclose(three[0], three[1])
    other = predict(model, normalizer, z0, params, 0, 1000.0, samples=1, seed=6, steps=4)
    assert not np.allclose(other[0], three[0])
    # Edges stay at the start heightmap, as in the simulator.
    np.testing.assert_allclose(three[:, 0, :], np.broadcast_to(z0[0], (3, z0.shape[1])))

    frames = rollout(model, normalizer, z0, params, 0, 1000.0, 3)
    assert frames.shape == (4, *z0.shape) and np.isfinite(frames).all()

    # Batched over runs, each run gets the frames it would get alone.
    from erosion_sim.generate import batched_rollout

    starts = np.load(data_root / "val" / "snapshots.npy")[:3, 0].astype(np.float64)
    for mode in ("rollout", "direct"):
        batch = batched_rollout(model, normalizer, starts, params, [0, 1, 2], 1000.0, 3, mode, seed=2)
        for r in range(3):
            alone = rollout(model, normalizer, starts[r], params, r, 1000.0, 3, mode, seed=2)
            np.testing.assert_allclose(batch[r], alone, rtol=1e-5, atol=1e-5)
