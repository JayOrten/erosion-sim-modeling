import json
from pathlib import Path

import numpy as np
import pytest
import torch

from erosion_sim.data.normalization import Scaling
from erosion_sim.data.splits import Split

# The test models are tiny. Many CPU threads only contend with each other (and with anything else running),
# which made the suite over 20 times slower.
torch.set_num_threads(1)

RUNS, SNAPS, SIZE = 4, 6, 16
SMOKE = "erosion_smoke/smoke"  # the smoke models' dataset
# The tests' own configs (smoke models, datasets, evaluations, predictors), laid out like the repo's configs/.
CONFIGS = Path(__file__).parent / "configs"


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
        path / "runs.npz",
        index=np.arange(RUNS),
        K=K_ref * 1.2, K_ref=K_ref, D=K_ref * np.array([0.5, 1, 2, 3]), m=np.array([0.3, 0.4, 0.5, 0.6]),
        U=np.array([0.0, 0.0, 1e-5, 2e-5]),
        # Terrain fields, as datagen stores them. The fake snapshots do not follow them.
        seed=np.arange(RUNS), period=np.full(RUNS, 64.0), persistence=np.full(RUNS, 0.5), octaves=np.full(RUNS, 5),
        lacunarity=np.full(RUNS, 2.0), height=np.full(RUNS, 64.0), noise=np.full(RUNS, "perlin"),
        fractal=np.full(RUNS, "fbm"), warp=np.zeros(RUNS),
    )
    (path / "meta.json").write_text(json.dumps({"config": {}, "split": path.name, "times": (np.arange(SNAPS) * 1000.0).tolist()}))
    return path


@pytest.fixture
def dataset(tmp_path):
    return Split(write_fake_dataset(tmp_path / "train"))


@pytest.fixture
def scaling(dataset):
    return Scaling.fit(dataset.snapshots, dataset.columns, np.arange(len(dataset)), dataset.times)


@pytest.fixture
def data_root(tmp_path):
    """A data folder holding a fake erosion_smoke/smoke dataset (the smoke model's) with train and val."""
    write_fake_dataset(tmp_path / "data" / SMOKE / "train", seed=0)
    write_fake_dataset(tmp_path / "data" / SMOKE / "val", seed=1)
    return tmp_path / "data"


@pytest.fixture(scope="session")
def dla_root(tmp_path_factory):
    """A data folder holding the dla_smoke dataset, every split, generated once per test session."""
    from erosion_sim.dla.config import data_config
    from erosion_sim.dla.data import make_split

    root = tmp_path_factory.mktemp("dla")
    for split in ("train", "val", "test"):
        make_split(data_config("dla_smoke", CONFIGS), split, root / "dla_smoke" / split, processes=1)
    return root
