import numpy as np

from erosion_sim.config import PROFILES
from erosion_sim.data import Split
from erosion_sim.evaluate import comparison_gif
from erosion_sim.predict import rollout
from erosion_sim.train import load_checkpoint, train


def test_rollout_and_gif(data_root, tmp_path):
    ckpt = tmp_path / "regression-smoke.pt"
    train(PROFILES["smoke"], data_root=data_root, show=False, out=ckpt)
    model, scaling, _, _ = load_checkpoint(ckpt, device="cpu")
    val = Split(data_root / "val")
    params, z0 = val.params, np.array(val.snapshots[0, 0])
    for mode in ("rollout", "direct"):
        frames = rollout(model, scaling, z0, params, 0, 1000.0, 3, mode)
        assert frames.shape == (4, *z0.shape)
        assert np.isfinite(frames).all()
        np.testing.assert_array_equal(frames[0], z0)
    gif = comparison_gif(ckpt, run=0, lag=2, out=tmp_path / "a.gif", data_root=data_root)
    assert gif.exists() and gif.stat().st_size > 0


def test_calls_needed():
    from erosion_sim.evaluate import calls_needed

    assert calls_needed([10], 50) == 5
    assert calls_needed([10], 15) is None
    assert calls_needed(range(1, 51), 25) == 1
    assert calls_needed([1, 10], 23) == 5  # 10 + 10 + 1 + 1 + 1
