from dataclasses import replace

import pytest
import torch

from erosion_sim.config import PROFILES
from erosion_sim.models import build_backbone
from erosion_sim.models.dit import DiT, apply_rope, rope_angles
from erosion_sim.models.fno import FNO
from erosion_sim.models.unet import UNet
from erosion_sim.objectives import flow
from erosion_sim.objectives.edm import Denoiser, initial_noise
from erosion_sim.objectives.sampling import generate, is_generative, network_calls
from erosion_sim.training.checkpoints import load_checkpoint
from erosion_sim.training.train import train

SMALL = {"unet": {}, "dit": dict(backbone="dit", patch=4, dit_depth=2, dit_width=16, dit_heads=4),
         "fno": dict(backbone="fno", fno_modes=4, fno_width=8, fno_depth=2)}


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


@pytest.mark.parametrize("backbone", ["unet", "dit", "fno"])
@pytest.mark.parametrize("objective", ["regression", "edm", "flow_matching"])
@pytest.mark.parametrize("task", ["erosion", "dla"])
def test_every_task_backbone_and_objective_trains_and_loads(task, backbone, objective, data_root, dla_root, tmp_path):
    base = PROFILES["smoke_dla" if task == "dla" else "smoke"]
    cfg = replace(base, objective=objective, **SMALL[backbone])
    history = train(cfg, data_root=dla_root if task == "dla" else data_root, out=tmp_path / "m")
    assert all(torch.isfinite(torch.tensor(history["train_loss"] + history["val_loss"])))
    model, _, loaded, _ = load_checkpoint(tmp_path / "m", device="cpu")
    assert loaded == cfg
    net = model.net if is_generative(model) else model
    assert isinstance(net, {"unet": UNet, "dit": DiT, "fno": FNO}[backbone])
    size = 48 if task == "dla" else 16
    x = torch.rand(2, cfg.in_channels, size, size)
    noise = torch.randn(2, 1, size, size)
    with torch.no_grad():
        assert generate(model, x, torch.rand(2, cfg.cond_dim), noise, 2).shape == (2, 1, size, size)


def test_fno_is_shift_equivariant():
    model = FNO(2, modes=4, width=8, depth=2, fixed_edges=False)
    torch.nn.init.normal_(model.out.weight)
    x, cond = torch.rand(1, 1, 16, 16), torch.rand(1, 2)
    shifted = model(torch.roll(x, (3, 5), (-2, -1)), cond)
    torch.testing.assert_close(shifted, torch.roll(model(x, cond), (3, 5), (-2, -1)), atol=1e-5, rtol=1e-4)


def test_residual_model_adds_a_sample_to_the_regression(data_root, tmp_path):
    from erosion_sim.objectives.residual import ResidualModel

    base = train(PROFILES["smoke"], data_root=data_root, out=tmp_path / "base")["folder"]
    cfg = replace(PROFILES["smoke"], objective="edm", residual_of=str(base))
    train(cfg, data_root=data_root, out=tmp_path / "residual")
    model, scaling, loaded, ckpt = load_checkpoint(tmp_path / "residual", device="cpu")
    regression, regression_scaling, _, _ = load_checkpoint(base, device="cpu")
    assert isinstance(model, ResidualModel) and loaded.residual_of == str(base)
    assert scaling == regression_scaling  # the residual model works in the regression model's units
    assert is_generative(model) and network_calls(model, 4) == 1 + 7
    x, cond, noise = torch.rand(2, 1, 16, 16), torch.rand(2, 4), torch.randn(2, 1, 16, 16)
    expected = regression(x, cond) + generate(model.inner, x, cond, noise, 2)
    torch.testing.assert_close(generate(model, x, cond, noise, 2), expected.detach())
    with pytest.raises(ValueError):
        replace(PROFILES["smoke"], residual_of=str(base))  # a regression residual model


def test_consistency_student_starts_from_the_teacher_and_samples_in_few_calls(data_root, tmp_path):
    from erosion_sim.objectives import consistency

    teacher = train(replace(PROFILES["smoke"], objective="edm"), data_root=data_root, out=tmp_path / "teacher")["folder"]
    cfg = replace(PROFILES["smoke"], objective="consistency", teacher=str(teacher), steps=0)
    with pytest.raises(ValueError):
        replace(PROFILES["smoke"], objective="consistency")
    train(cfg, data_root=data_root, out=tmp_path / "student0", steps=1, eval_every=1, lr=0.0, warmup_steps=0)
    student, scaling, _, ckpt = load_checkpoint(tmp_path / "student0", device="cpu")
    edm, edm_scaling, _, _ = load_checkpoint(teacher, device="cpu")
    assert isinstance(student, consistency.ConsistencyModel) and scaling == edm_scaling
    assert student.sigma_data == edm.sigma_data
    for k, v in edm.state_dict().items():  # no learning rate, so still the teacher's weights
        torch.testing.assert_close(student.state_dict()[k], v)
    train(cfg, data_root=data_root, out=tmp_path / "student", steps=6)
    student = load_checkpoint(tmp_path / "student", device="cpu")[0]
    x, cond, noise = torch.rand(2, 1, 16, 16), torch.rand(2, 4), torch.randn(2, 1, 16, 16)
    assert network_calls(student, 2) == 2
    one, two = generate(student, x, cond, noise, 1), generate(student, x, cond, noise, 2)
    assert one.shape == two.shape == (2, 1, 16, 16) and not torch.equal(one, two)
    # Run step by step as world phases, the sample is the same.
    from erosion_sim.objectives.sampling import (
        phase_schedules,
        run_phase,
        start_state,
        to_target,
    )

    schedules = phase_schedules(student, 2, (5.0,))
    state = start_state(student, noise, schedules)
    for schedule in schedules:
        state = run_phase(student, x, cond, state, schedule)
    torch.testing.assert_close(to_target(student, state), two, atol=1e-5, rtol=1e-5)

