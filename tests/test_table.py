import numpy as np
import pytest

from erosion_sim import table

COLUMNS = {
    "index": np.arange(6),
    "fractal": np.array(["fbm", "ridged", "fbm", "fbm", "ridged", "fbm"]),
    "noise": np.array(["perlin", "perlin", "opensimplex2", "value_cubic", "perlin", "perlin"]),
    "K_ref": np.array([1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 1e-4]),
    "D": np.array([1e-4, 2e-4, 6e-4, 4e-4, 5e-4, 3e-4]),
    "U": np.zeros(6),
}


def test_value_filter():
    np.testing.assert_array_equal(table.select(COLUMNS, fractal="fbm"), [0, 2, 3, 5])
    np.testing.assert_array_equal(table.select(COLUMNS, U=0), np.arange(6))


def test_set_filter():
    np.testing.assert_array_equal(table.select(COLUMNS, noise={"opensimplex2", "value_cubic"}), [2, 3])
    np.testing.assert_array_equal(table.select(COLUMNS, noise=frozenset({"perlin"})), [0, 1, 4, 5])
    np.testing.assert_array_equal(table.select(COLUMNS, noise=["perlin"]), [0, 1, 4, 5])


def test_range_filter_is_half_open():
    np.testing.assert_array_equal(table.select(COLUMNS, index=(0, 3)), [0, 1, 2])
    np.testing.assert_array_equal(table.select(COLUMNS, index=(4, None)), [4, 5])
    np.testing.assert_array_equal(table.select(COLUMNS, K_ref=(None, 2e-4)), [0, 5])


def test_filters_combine_and_derived_columns_work():
    np.testing.assert_array_equal(table.select(COLUMNS, fractal="fbm", index=(0, 4)), [0, 2, 3])
    np.testing.assert_array_equal(table.select(COLUMNS, D_over_K=(1.5, None)), [2, 5])
    np.testing.assert_array_equal(table.select(COLUMNS), np.arange(6))


def test_bad_filters_fail():
    with pytest.raises(KeyError):
        table.select(COLUMNS, nope=1)
    with pytest.raises(ValueError):
        table.select(COLUMNS, index=(0, 1, 2))


def test_load_reads_runs_npz(tmp_path):
    np.savez(tmp_path / "runs.npz", **COLUMNS)
    loaded = table.load(tmp_path)
    assert set(loaded) == set(COLUMNS)
    np.testing.assert_array_equal(loaded["fractal"], COLUMNS["fractal"])
