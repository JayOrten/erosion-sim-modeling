from dataclasses import replace

from erosion_sim.config import PROFILES
from erosion_sim.identity import config_from_json, config_hash, model_dir, to_json

CFG = PROFILES["smoke"]


def test_hash_ignores_labels_and_settings_that_do_not_change_the_weights():
    same = replace(CFG, profile="another_label", num_workers=3, log_every=7, compile=True)
    assert config_hash(same) == config_hash(CFG)
    for change in (dict(lr=1e-3), dict(seed=1), dict(objective="edm"), dict(data=(("fractal", "fbm"),))):
        assert config_hash(replace(CFG, **change)) != config_hash(CFG)


def test_set_and_range_filters_hash_differently():
    one_of = replace(CFG, data=(("index", frozenset({0, 10})),))
    within = replace(CFG, data=(("index", (0, 10)),))
    assert config_hash(one_of) != config_hash(within)
    reordered = replace(CFG, data=(("noise", frozenset({"perlin", "value_cubic"})),))
    assert config_hash(reordered) == config_hash(replace(CFG, data=(("noise", frozenset({"value_cubic", "perlin"})),)))


def test_config_round_trips_through_json():
    cfg = replace(CFG, data=(("noise", frozenset({"perlin", "ridged"})), ("index", (0, 2667)), ("U", 0.0)),
                  lags=(1, 2))
    assert config_from_json(to_json(cfg.__dict__)) == cfg


def test_model_dir_reuses_a_folder_trained_under_another_label(tmp_path):
    first = model_dir(CFG, tmp_path)
    assert first == tmp_path / f"smoke-{config_hash(CFG)}"
    first.mkdir()
    assert model_dir(replace(CFG, profile="B1_base"), tmp_path) == first
    assert model_dir(replace(CFG, lr=1e-3), tmp_path).name.startswith("smoke-")
    assert model_dir(replace(CFG, lr=1e-3), tmp_path) != first
