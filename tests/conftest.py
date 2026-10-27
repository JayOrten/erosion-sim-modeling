import json

import numpy as np
import pytest

RUNS, SNAPS, SIZE = 4, 6, 16


def write_fake_dataset(path, seed=0):
    """A tiny dataset in the format datagen writes: terrain that lowers at a rate set by K_ref."""
    path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    K_ref = np.array([1e-4, 2e-4, 3e-4, 4e-4])
    z0 = rng.uniform(0, 64, (RUNS, 1, SIZE, SIZE))
    t = np.arange(SNAPS)[None, :, None, None] * 1000.0
    snapshots = (z0 * np.exp(-0.05 * K_ref[:, None, None, None] * t)).astype(np.float32)
    snapshots[:, :, 0, :] = snapshots[:, :1, 0, :]  # fixed edges, as in the simulator
    np.save(path / "snapshots.npy", snapshots)
    np.savez(
        path / "params.npz",
        times=np.arange(SNAPS) * 1000.0,
        K=K_ref * 1.2, K_ref=K_ref, D=K_ref * np.array([0.5, 1, 2, 3]), m=np.array([0.3, 0.4, 0.5, 0.6]),
        U=np.array([0.0, 0.0, 1e-5, 2e-5]), period=np.full(RUNS, 64.0), persistence=np.full(RUNS, 0.5),
        terrain_seed=np.arange(RUNS),
    )
    (path / "meta.json").write_text(json.dumps({"config": {}, "split": path.name}))
    return path


@pytest.fixture
def dataset_path(tmp_path):
    return write_fake_dataset(tmp_path / "train")


@pytest.fixture
def data_root(tmp_path):
    write_fake_dataset(tmp_path / "data" / "train", seed=0)
    write_fake_dataset(tmp_path / "data" / "val", seed=1)
    return tmp_path / "data"
