from dataclasses import replace

import pytest
import torch

from erosion_sim import flow
from erosion_sim.config import PROFILES
from erosion_sim.diffusion import Denoiser, initial_noise
from erosion_sim.dit import DiT, apply_rope, rope_angles
from erosion_sim.model import UNet, build_backbone
from erosion_sim.sampling import generate, is_generative, network_calls
from erosion_sim.train import load_checkpoint, train

SMALL_DIT = dict(backbone="dit", patch=4, dit_depth=2, dit_width=16, dit_heads=4)


def test_rope_scores_depend_only_on_the_offset():
    cos, sin = rope_angles(5, 5, 8)
    q, k = torch.randn(8), torch.randn(8)

    def score(a, b):  # tokens a and b on the 5 × 5 grid, as (row, col)
        qa = apply_rope(q[None, None, None], cos[None, None, [a[0] * 5 + a[1]]], sin[None, None, [a[0] * 5 + a[1]]])
        kb = apply_rope(k[None, None, None], cos[None, None, [b[0] * 5 + b[1]]], sin[None, None, [b[0] * 5 + b[1]]])
        return (qa * kb).sum()

    torch.testing.assert_close(score((0, 0), (1, 2)), score((3, 2), (4, 4)))
    assert not torch.isclose(score((0, 0), (1, 2)), score((0, 0), (2, 1)))


def test_dit_shapes_zero_start_and_fixed_edges():
    model = DiT(3, patch=4, depth=2, width=16, heads=4, in_channels=2)
    x, cond = torch.rand(2, 2, 16, 24), torch.rand(2, 3)
    assert torch.equal(model(x, cond), torch.zeros(2, 1, 16, 24))  # output layer starts at zero
    torch.nn.init.normal_(model.out.weight)
    out = model(x, cond)
    assert out.shape == (2, 1, 16, 24) and out[..., 1:-1, 1:-1].abs().sum() > 0
    assert out[..., 0, :].abs().sum() == out[..., :, -1].abs().sum() == 0
    with pytest.raises(ValueError):
        model(torch.rand(1, 2, 18, 16), torch.rand(1, 3))
    with pytest.raises(ValueError):
        build_backbone("mlp", 3, 1, True)
    assert isinstance(build_backbone("unet", 3, 1, True, channels=(8, 16), emb_dim=16), UNet)


class GaussianFlow(flow.FlowModel):
    """The exact velocity when the target is Gaussian with standard deviation sigma_data."""

    def __init__(self, sigma_data):
        super().__init__(1, sigma_data=sigma_data, fixed_edges=False, channels=(8,), emb_dim=8)

    def forward(self, x, cond, z_s, s):
        t = s.reshape(-1, 1, 1, 1)
        return (2 * t - 1) / ((1 - t) ** 2 + t**2) * z_s


def test_euler_sampler_solves_the_gaussian_case():
    model = GaussianFlow(sigma_data=2.5)
    noise = torch.randn(64, 1, 16, 16, generator=torch.Generator().manual_seed(0))
    y = flow.sample(model, torch.zeros(64, 1, 16, 16), torch.zeros(64, 1), noise, steps=200)
    assert abs(y.std().item() / 2.5 - 1) < 0.05
    torch.testing.assert_close(y, flow.sample(model, torch.zeros(64, 1, 16, 16), torch.zeros(64, 1), noise, steps=200))


def test_flow_loss_is_zero_for_the_exact_velocity():
    model = flow.FlowModel(1, sigma_data=2.0, fixed_edges=False, channels=(8,), emb_dim=8)
    y, noise, s = torch.randn(4, 1, 8, 8) * 2.0, torch.randn(4, 1, 8, 8), torch.rand(4)
    exact = noise - y / 2.0
    model.forward = lambda x, cond, z_s, t: exact
    mse, estimate = flow.flow_loss(model, None, None, y, s, noise)
    torch.testing.assert_close(mse, torch.zeros(4))
    torch.testing.assert_close(estimate, y)


def test_generate_and_network_calls_follow_the_objective():
    x, cond = torch.rand(2, 1, 16, 16), torch.rand(2, 2)
    noise = initial_noise([(0,), (1,)], (16, 16), fixed_edges=False)
    models = {
        "regression": UNet(2, channels=(8, 16), emb_dim=16, fixed_edges=False),
        "edm": Denoiser(2, fixed_edges=False, channels=(8, 16), emb_dim=16),
        "flow_matching": flow.FlowModel(2, fixed_edges=False, channels=(8, 16), emb_dim=16),
    }
    assert [network_calls(m, 8) for m in models.values()] == [1, 15, 8]
    assert [is_generative(m) for m in models.values()] == [False, True, True]
    for model in models.values():
        assert generate(model, x, cond, noise, 2).shape == (2, 1, 16, 16)


@pytest.mark.parametrize("backbone", ["unet", "dit"])
@pytest.mark.parametrize("objective", ["regression", "edm", "flow_matching"])
@pytest.mark.parametrize("task", ["erosion", "dla"])
def test_every_task_backbone_and_objective_trains_and_loads(task, backbone, objective, data_root, dla_root, tmp_path):
    base = PROFILES["smoke_dla" if task == "dla" else "smoke"]
    cfg = replace(base, objective=objective, **(SMALL_DIT if backbone == "dit" else {}))
    history = train(cfg, data_root=dla_root if task == "dla" else data_root, out=tmp_path / "m")
    assert all(torch.isfinite(torch.tensor(history["train_loss"] + history["val_loss"])))
    model, _, loaded, _ = load_checkpoint(tmp_path / "m", device="cpu")
    assert loaded == cfg
    net = model.net if is_generative(model) else model
    assert isinstance(net, DiT if backbone == "dit" else UNet)
    size = 48 if task == "dla" else 16
    x = torch.rand(2, cfg.in_channels, size, size)
    noise = torch.randn(2, 1, size, size)
    with torch.no_grad():
        assert generate(model, x, torch.rand(2, cfg.cond_dim), noise, 2).shape == (2, 1, size, size)
