import json
from dataclasses import asdict, dataclass

import matplotlib
import numpy as np

from erosion_terrain import TERRAIN_FIELDS, Terrain

from erosion_sim.world.compare import Version, compare_versions, example_index
from erosion_sim.data import Split
from erosion_sim.world.evaluate import run_world

from test_world import LEVELS

matplotlib.use("Agg")


@dataclass(frozen=True)
class FakeVersion(Version):
    """The untrained test models instead of checkpoint files."""

    def load(self, device=None):
        return list(LEVELS)

    def key(self):  # no checkpoint files to read
        return asdict(self)


# Two runs with different terrain kinds, as world_master draws them.
TERRAINS = [Terrain(3, persistence=0.5), Terrain(4, persistence=0.4, noise="opensimplex2", warp=0.2)]


def write_world_runs(root, size=48):
    """A fake dataset split in the format datagen writes, on real world terrain. The "simulator" lowers
    everything by 1 m per 10,000 yr. Returns the profile folder (``root`` for ``Split.open``)."""
    path = root / "test"
    path.mkdir(parents=True)
    times = np.array([0.0, 10_000.0, 20_000.0])
    z0 = np.stack([t.heights(0, 0, size, size) for t in TERRAINS])
    np.save(path / "snapshots.npy", np.stack([z0 - t / 1e4 for t in times], axis=1).astype(np.float32))
    n = len(TERRAINS)
    terrain = {name: np.array([getattr(t, name) for t in TERRAINS]) for name in TERRAIN_FIELDS}
    np.savez(path / "params.npz", times=times, K=np.full(n, 3e-4), K_ref=np.full(n, 2.5e-4), D=np.full(n, 2.5e-4),
             m=np.full(n, 0.4), U=np.zeros(n), **terrain)
    (path / "meta.json").write_text(json.dumps({"config": {}, "split": "test"}))
    return root


def test_split_rebuilds_each_terrain(tmp_path):
    data = Split.open("world_master", "test", write_world_runs(tmp_path))
    for i, terrain in enumerate(TERRAINS):
        assert data.terrain(i) == terrain
        np.testing.assert_allclose(data.snapshots[i, 0], terrain.heights(0, 0, 48, 48), atol=1e-4)


def test_compare_versions(tmp_path, monkeypatch):
    import erosion_sim.world.compare as compare

    monkeypatch.setattr(compare, "CACHE_DIR", tmp_path / "cache")
    root = write_world_runs(tmp_path / "data")
    versions = {"a": FakeVersion("x", "y", steps=2), "b": FakeVersion("x", "y", steps=3, splits=())}
    result = compare_versions(versions, runs=range(2), dTs=(10_000.0, 20_000.0), margin=8, coarse_only=True,
                              root=root, show=True)
    pairs = result["pairs"]
    assert pairs["methods"] == ["no change", "a", "a, coarse level only", "b", "b, coarse level only"]
    assert pairs["z_a"].shape == (4, 32, 32)
    # The fake simulator lowers by 1 m per 10,000 yr.
    assert result["by_dT"]["dT"] == [10_000.0, 20_000.0]
    assert np.isclose(result["by_dT"]["no change"]["rmse"][1], 2.0, atol=1e-4)
    assert np.isclose(result["table"]["no change"]["rmse"], np.sqrt((1 + 4) / 2), atol=1e-4)
    i = example_index(result, run=1, dT=10_000.0)
    assert pairs["run"][i] == 1 and pairs["dT"][i] == 10_000.0
    # The world on run 1 is the test world's own output on that run's terrain.
    world = run_world(Split.open("world_master", "test", root), 1, 10_000.0, *LEVELS, steps=2)
    assert world.terrain == TERRAINS[1]
    np.testing.assert_allclose(pairs["a"][i], world.get(8, 8, 32, 32), atol=1e-4)


def test_predictions_are_cached_by_checkpoint(tmp_path, monkeypatch):
    import erosion_sim.world.compare as compare

    root = write_world_runs(tmp_path / "data")
    monkeypatch.setattr(compare, "CACHE_DIR", tmp_path / "cache")
    files = [tmp_path / "coarse.pt", tmp_path / "fine.pt"]
    for f in files:
        f.write_bytes(b"1")

    @dataclass(frozen=True)
    class Files(FakeVersion):
        key = Version.key  # keyed by the checkpoint files, as real versions are

        def paths(self):
            return tuple(files)

    calls = []
    real = compare._predict
    monkeypatch.setattr(compare, "_predict", lambda *a: calls.append(1) or real(*a))
    run = dict(runs=range(2), dTs=(10_000.0,), margin=8, root=root, show=False)
    first = compare_versions({"a": Files("x", "y", steps=2)}, **run)
    second = compare_versions({"a": Files("x", "y", steps=2)}, **run)
    assert len(calls) == 1
    np.testing.assert_array_equal(first["pairs"]["a"], second["pairs"]["a"])
    files[1].write_bytes(b"22")  # a retrained checkpoint: computed again
    compare_versions({"a": Files("x", "y", steps=2)}, **run)
    compare_versions({"a": Files("x", "y", steps=3)}, **run)  # other sampler settings: computed again
    assert len(calls) == 3
