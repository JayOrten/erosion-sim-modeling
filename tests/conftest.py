import json

import numpy as np
import pytest
import torch

from erosion_sim.data import Split
from erosion_sim.normalization import Scaling

# The test models are tiny. Many CPU threads only contend with each other (and with anything else running),
# which made the suite over 20 times slower.
torch.set_num_threads(1)

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
        U=np.array([0.0, 0.0, 1e-5, 2e-5]),
        # Terrain fields, as datagen stores them. The fake snapshots do not follow them.
        seed=np.arange(RUNS), period=np.full(RUNS, 64.0), persistence=np.full(RUNS, 0.5), octaves=np.full(RUNS, 5),
        lacunarity=np.full(RUNS, 2.0), height=np.full(RUNS, 64.0), noise=np.full(RUNS, "perlin"),
        fractal=np.full(RUNS, "fbm"), warp=np.zeros(RUNS),
    )
    (path / "meta.json").write_text(json.dumps({"config": {}, "split": path.name}))
    return path


@pytest.fixture
def dataset(tmp_path):
    return Split(write_fake_dataset(tmp_path / "train"))


@pytest.fixture
def scaling(dataset):
    return Scaling.fit(dataset.snapshots, dataset.params, np.arange(len(dataset)), dataset.times)


@pytest.fixture
def data_root(tmp_path):
    write_fake_dataset(tmp_path / "data" / "train", seed=0)
    write_fake_dataset(tmp_path / "data" / "val", seed=1)
    return tmp_path / "data"
