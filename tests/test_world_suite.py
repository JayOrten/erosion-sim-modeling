from dataclasses import replace

import numpy as np
import pytest
from conftest import CONFIGS, write_fake_dataset

from erosion_sim.config import model_config
from erosion_sim.evaluation.results import read_results
from erosion_sim.training.train import train
from erosion_sim.world.suite import Settings, evaluate_world


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("world")
    for split, seed in (("train", 0), ("val", 1), ("test", 2)):
        write_fake_dataset(root / "data" / "erosion_smoke" / "smoke" / split, seed=seed)
    base = replace(replace(model_config("smoke", CONFIGS), objective="edm"), fixed_edges=False, augment=True)
    configs = {
        "coarse": replace(base, cell=2, window=4),
        "fine": replace(base, cell=1, window=8, context_cell=2),
        "whole": replace(model_config("smoke", CONFIGS), objective="regression"),
    }
    return root, {label: train(cfg, data_root=root / "data", out=root / label)["folder"] for label, cfg in configs.items()}


SAMPLER = (("steps", 2), ("splits", (5.0,)))
BASELINES = {"no_change": {"method": "no_change"}, "analytical": {"method": "analytical"},
             "analytical_chunks": {"method": "analytical_chunks", "chunk": 8},
             "analytical_blended": {"method": "analytical_blended", "chunk": 8},
             "model_chunks": {"method": "model_chunks", "model": "whole", "chunk": 8, "steps": 2}}


def test_world_suite_scores_cascades_levels_and_baselines(trained, tmp_path):
    root, folders = trained
    settings = Settings(cascades=(("world", ("coarse", "fine"), SAMPLER),), runs=2, ends_yr=(2000.0, 4000.0),
                        margin=2, steps=2, chunk=8, chunk_edge=2, level_pairs_per_run=2, seam_windows=2)
    out = evaluate_world(folders, tmp_path / "r", settings, data_root=root / "data", device="cpu", baselines=BASELINES)
    results = read_results(out)
    rows = results.scalars["entries"]
    assert set(rows) == {"no_change", "world", "analytical", "analytical_chunks", "analytical_blended", "model_chunks",
                         "level:coarse", "level:fine"}
    assert rows["world"]["pairs"] == 4 and rows["level:fine"]["pairs"] == 4
    assert rows["no_change"]["skill"] == 0 and "chunk_edge_rmse" in rows["analytical_chunks"]
    assert rows["world"]["seconds_per_query"] > 0
    assert results.manifest["test"]["interior"] == 12
    assert set(results.scalars["seams"]["world"]) == {"level0-2m", "level1-1m"}
    assert set(results.scalars["by"]["dT"]) == {"2000", "4000"}
    assert results.examples["world"].shape == (2, 12, 12)
    assert results.manifest["entries"]["world"]["kind"] == "cascade"
    assert results.manifest["entries"]["model_chunks"] == {"kind": "baseline", "method": "model_chunks"}

    # Baselines scored on their own share the interior and values with the run above.
    baseline_settings = replace(settings, cascades=(), data_profile="erosion_smoke/smoke")
    alone = read_results(evaluate_world({"whole": folders["whole"]}, tmp_path / "b", baseline_settings,
                                        data_root=root / "data", device="cpu", baselines=BASELINES))
    assert alone.manifest["test"]["interior"] == 12 and "world" not in alone.scalars["entries"]
    for name in BASELINES:
        np.testing.assert_array_equal(alone.arrays[f"{name}/sq_error"], results.arrays[f"{name}/sq_error"])


def test_world_suite_needs_something_to_score_and_chunk_files(trained, tmp_path):
    root, folders = trained
    with pytest.raises(ValueError, match="nothing to score"):
        evaluate_world(folders, tmp_path / "r", Settings(), data_root=root / "data")
    with pytest.raises(FileNotFoundError, match="chunked"):
        evaluate_world(folders, tmp_path / "r", Settings(data_profile="erosion_smoke/smoke", chunk=8, ends_yr=(2000.0,)),
                       data_root=root / "data", baselines={"s": {"method": "simulator_chunks", "chunk": 8}})
    with pytest.raises(ValueError, match="margin"):
        evaluate_world(folders, tmp_path / "r", Settings(cascades=(("w", ("coarse", "fine"), SAMPLER),), margin=3,
                                                          ends_yr=(2000.0,)), data_root=root / "data")
