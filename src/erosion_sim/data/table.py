"""Run tables: one row per simulation run, stored as ``runs.npz`` with one array per column.

``datagen/`` writes the table next to the snapshots. ``select`` picks runs by column, and is how
experiments and configs name the data they use.
"""

from pathlib import Path

import numpy as np

# Columns computed from stored ones, usable in filters like any other column.
DERIVED = {
    "D_over_K": lambda c: c["D"] / c["K_ref"],
}


def load(path: Path) -> dict[str, np.ndarray]:
    """The columns of the run table in folder ``path``."""
    with np.load(Path(path) / "runs.npz") as f:
        return dict(f)


def column(columns: dict[str, np.ndarray], name: str) -> np.ndarray:
    """One column by name, stored or derived."""
    if name in columns:
        return columns[name]
    if name in DERIVED:
        return DERIVED[name](columns)
    raise KeyError(f"no column {name!r}; stored: {sorted(columns)}, derived: {sorted(DERIVED)}")


def _matches(values: np.ndarray, condition) -> np.ndarray:
    if isinstance(condition, tuple):
        if len(condition) != 2:
            raise ValueError(f"a range is (low, high), got {condition!r}")
        low, high = condition
        keep = np.ones(len(values), dtype=bool)
        if low is not None:
            keep &= values >= low
        if high is not None:
            keep &= values < high
        return keep
    if isinstance(condition, (set, frozenset, list)):
        return np.isin(values, list(condition))
    return values == condition


def select(columns: dict[str, np.ndarray], **filters) -> np.ndarray:
    """Row indices of the runs that match every filter, in table order.

    Each filter names a column (stored or in ``DERIVED``) and gives one of:

    - a value: the column equals it. ``fractal="fbm"``, ``U=0``.
    - a set, frozenset, or list: the column is one of its values. ``noise={"perlin", "value_cubic"}``.
    - a tuple ``(low, high)``: ``low <= column < high``. Either end may be None. ``index=(0, 2667)`` is
      the first 2,667 runs.

    No filters selects every run.
    """
    count = len(next(iter(columns.values())))
    keep = np.ones(count, dtype=bool)
    for name, condition in filters.items():
        keep &= _matches(column(columns, name), condition)
    return np.flatnonzero(keep)
