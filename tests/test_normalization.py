import numpy as np

from erosion_sim.normalization import (
    HEIGHT_UNIT, PARAMETER_NAMES, ChangeScaling, ParameterScaling, Scaling, change_rms, terrain_to_model,
)

from conftest import SIZE, SNAPS


def test_terrain_is_centered_and_in_height_units():
    z = np.random.default_rng(0).normal(size=(2, SIZE, SIZE)) * 10
    x = terrain_to_model(z)
    np.testing.assert_allclose(x.mean(axis=(1, 2)), 0, atol=1e-12)
    np.testing.assert_allclose(x, (z - z.mean(axis=(1, 2), keepdims=True)) / HEIGHT_UNIT)
    # Raising a heightmap changes nothing; doubling its relief doubles the input.
    np.testing.assert_allclose(terrain_to_model(z + 100), x)
    np.testing.assert_allclose(terrain_to_model(2 * z), 2 * x)


def test_parameter_scaling_maps_the_training_range_to_unit_interval(dataset):
    runs = np.arange(len(dataset))
    scaling = ParameterScaling.fit(dataset.params, runs, dataset.times)
    shortest, longest = dataset.times[1] - dataset.times[0], dataset.times[-1] - dataset.times[0]
    low = scaling.scale(dataset.params, runs, shortest)
    high = scaling.scale(dataset.params, runs, longest)
    assert low.shape == (len(runs), len(PARAMETER_NAMES))
    np.testing.assert_allclose(low[:, 0], 0)
    np.testing.assert_allclose(high[:, 0], 1)
    np.testing.assert_allclose(low[:, 1:].min(axis=0), 0)
    np.testing.assert_allclose(low[:, 1:].max(axis=0), 1)


def test_change_scaling_round_trips():
    change = ChangeScaling(coef=0.5, power=0.8)
    dz = np.random.default_rng(1).normal(size=(SIZE, SIZE))
    np.testing.assert_allclose(change.to_model(dz, 2e-4, 5000.0), dz / (0.5 * (2e-4 * 5000.0) ** 0.8))
    np.testing.assert_allclose(change.from_model(change.to_model(dz, 2e-4, 5000.0), 2e-4, 5000.0), dz)


def test_change_scaling_evens_out_target_size(dataset):
    runs = np.arange(len(dataset))
    s, times, K_ref = dataset.snapshots, dataset.times, dataset.params["K_ref"]
    change = ChangeScaling.fit(s, dataset.params, runs, times)

    def target_rms(lag):
        dz = s[:, lag] - s[:, 0]
        return np.sqrt((change.to_model(dz, K_ref[:, None, None], times[lag] - times[0]) ** 2).mean())

    raw = np.sqrt(((s[:, SNAPS - 1] - s[:, 0]) ** 2).mean()) / np.sqrt(((s[:, 1] - s[:, 0]) ** 2).mean())
    assert raw > 3  # without scaling, long elapsed times give much larger changes
    assert 0.5 < target_rms(SNAPS - 1) / target_rms(1) < 2.0
    # The targets are about size 1, so sigma_data is near 1.
    assert 0.5 < change_rms(s, dataset.params, runs, times, change) < 2.0


def test_scaling_round_trips_through_a_dict(dataset):
    from dataclasses import asdict

    scaling = Scaling.fit(dataset.snapshots, dataset.params, np.arange(len(dataset)), dataset.times)
    assert Scaling.from_dict(asdict(scaling)) == scaling
