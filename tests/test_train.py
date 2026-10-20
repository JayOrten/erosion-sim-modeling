import torch

from erosion_sim.config import PROFILES
from erosion_sim.train import load_checkpoint, train


def test_smoke_training_saves_a_loadable_checkpoint(data_root, tmp_path):
    out = tmp_path / "unet-smoke.pt"
    history = train(PROFILES["smoke"], data_root=data_root, show=False, out=out)
    assert out.exists()
    assert len(history["val_loss"]) == 2
    assert all(torch.isfinite(torch.tensor(history["train_loss"])))

    model, normalizer, cfg, ckpt = load_checkpoint(out, device="cpu")
    assert cfg == PROFILES["smoke"]
    x, cond = torch.rand(2, 1, 16, 16), torch.rand(2, 5)
    with torch.no_grad():
        assert model(x, cond).shape == (2, 1, 16, 16)
