import torch

from erosion_sim.model import UNet

COND = 5


def small_model(**kwargs):
    return UNet(COND, channels=(8, 16), blocks=1, emb_dim=16, frequencies=4, **kwargs)


def test_output_shape_and_fixed_edges():
    model = small_model()
    torch.nn.init.normal_(model.out.weight)  # so the output is not all zero
    y = model(torch.rand(3, 1, 16, 16), torch.rand(3, COND))
    assert y.shape == (3, 1, 16, 16)
    for edge in (y[..., 0, :], y[..., -1, :], y[..., :, 0], y[..., :, -1]):
        assert torch.all(edge == 0)
    assert y[..., 1:-1, 1:-1].abs().sum() > 0


def test_untrained_model_predicts_no_change():
    y = small_model()(torch.rand(2, 1, 16, 16), torch.rand(2, COND))
    assert torch.all(y == 0)


def test_zero_conditioning_and_no_attention_run():
    for attention in (True, False):
        y = small_model(attention=attention)(torch.rand(2, 1, 16, 16), torch.zeros(2, COND))
        assert torch.isfinite(y).all()


def test_conditioning_changes_the_output():
    torch.manual_seed(0)
    model = small_model()
    torch.nn.init.normal_(model.out.weight)
    x = torch.rand(1, 1, 16, 16)
    a = model(x, torch.zeros(1, COND))
    b = model(x, torch.ones(1, COND))
    assert not torch.allclose(a, b)


def test_gradients_reach_every_parameter():
    model = small_model()
    torch.nn.init.normal_(model.out.weight)
    model(torch.rand(2, 1, 16, 16), torch.rand(2, COND)).pow(2).mean().backward()
    missing = [name for name, p in model.named_parameters() if p.grad is None or p.grad.abs().sum() == 0]
    assert missing == []
