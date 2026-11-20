from dataclasses import replace

import pytest
from conftest import write_fake_dataset

from erosion_sim.config import get_config
from erosion_sim.experiments.results import read_results
from erosion_sim.training.train import train
from erosion_sim.world.suite import Settings, evaluate_world


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("world")
    for split, seed in (("train", 0), ("val", 1), ("test", 2)):
        write_fake_dataset(root / "data" / split, seed=seed)
    base = replace(get_config("smoke", "edm"), fixed_edges=False, augment=True)
    configs = {
        "coarse": replace(base, cell=2, window=4),
        "fine": replace(base, cell=1, window=8, context_cell=2),
        "whole": get_config("smoke", "regression"),
    }
    return root, {label: train(cfg, data_root=root / "data", out=root / label)["folder"] for label, cfg in configs.items()}


def test_world_suite_scores_worlds_levels_and_baselines(trained, tmp_path):
    root, folders = trained
    settings = Settings(worlds=(("world", ("coarse", "fine")),), runs=2, ends_yr=(2000.0, 4000.0), margin=2,
                        steps=2, chunk=8, chunk_edge=2, chunk_model="whole", level_pairs_per_run=2, seam_windows=2,
                        baselines=("analytical", "analytical_chunks", "analytical_blended", "model_chunks"))
    out = evaluate_world(folders, tmp_path / "r", settings, data_root=root / "data", device="cpu")
    results = read_results(out)
    rows = results.scalars["entries"]
    assert set(rows) == {"no change", "world", "analytical", "analytical_chunks", "analytical_blended", "model_chunks",
                         "level:coarse", "level:fine"}
    assert rows["world"]["pairs"] == 4 and rows["level:fine"]["pairs"] == 4
    assert rows["no change"]["skill"] == 0 and "chunk_edge_rmse" in rows["analytical_chunks"]
    assert rows["world"]["seconds_per_query"] > 0
    assert results.manifest["test"]["interior"] == 12
    assert set(results.scalars["seams"]["world"]) == {"level0-2m", "level1-1m"}
    assert set(results.scalars["by"]["dT"]) == {"2000", "4000"}
    assert results.examples["world"].shape == (2, 12, 12)
    assert results.manifest["entries"]["world"]["kind"] == "world"


def test_world_suite_needs_a_world_and_chunk_files(trained, tmp_path):
    root, folders = trained
    with pytest.raises(ValueError):
        evaluate_world(folders, tmp_path / "r", Settings(), data_root=root / "data")
    with pytest.raises(FileNotFoundError, match="chunked"):
        evaluate_world(folders, tmp_path / "r", Settings(worlds=(("w", ("coarse", "fine")),), chunk=8, ends_yr=(2000.0,),
                                                          baselines=("simulator_chunks",)),
                       data_root=root / "data")
