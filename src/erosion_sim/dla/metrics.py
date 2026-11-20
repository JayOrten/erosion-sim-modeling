"""DLA metrics on clusters stored as occupancy images: (N, H, W), 1 occupied, 0 empty.

A model sample cannot match one simulator run cell for cell, since DLA is random. So each metric gives
one value per cluster, and ``compare`` summarizes how the model's values are distributed against the
simulator's under the same conditions.

The seed particle sits at the grid center, ``size // 2`` in both axes.
"""

import numpy as np
from numba import njit

OFFSETS = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=np.int64)


def particle_counts(occupied: np.ndarray) -> np.ndarray:
    """Occupied cells per cluster."""
    return occupied.reshape(len(occupied), -1).sum(axis=1).astype(np.int64)


@njit(cache=True)
def _reachable(mask: np.ndarray, row: int, col: int) -> int:
    """Occupied cells 4-connected to (row, col), by breadth-first search. 0 if (row, col) is empty."""
    if not mask[row, col]:
        return 0
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=np.bool_)
    queue = np.empty((h * w, 2), dtype=np.int64)
    queue[0, 0], queue[0, 1] = row, col
    seen[row, col] = True
    head, tail = 0, 1
    while head < tail:
        r, c = queue[head, 0], queue[head, 1]
        head += 1
        for k in range(4):
            nr, nc = r + OFFSETS[k, 0], c + OFFSETS[k, 1]
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                queue[tail, 0], queue[tail, 1] = nr, nc
                tail += 1
    return tail


def connected_fraction(occupied: np.ndarray) -> np.ndarray:
    """Per cluster, the fraction of occupied cells joined to the seed particle at the center. 1 is one
    piece; less means floating particles or a missing seed."""
    center = occupied.shape[-1] // 2
    counts = particle_counts(occupied)
    joined = np.array([_reachable(o.astype(np.bool_), center, center) for o in occupied])
    return np.where(counts > 0, joined / np.maximum(counts, 1), 0.0)


def start_kept(occupied: np.ndarray, start: np.ndarray) -> np.ndarray:
    """Per cluster, the fraction of the start cluster's cells still occupied. A grown cluster keeps them all."""
    start = np.broadcast_to(start, occupied.shape).astype(bool)
    kept = (occupied.astype(bool) & start).reshape(len(occupied), -1).sum(axis=1)
    return kept / np.maximum(start.reshape(len(occupied), -1).sum(axis=1), 1)


def _coordinates(size: int) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = np.mgrid[:size, :size]
    return rows.astype(np.float64), cols.astype(np.float64)


def gyration_radius(occupied: np.ndarray) -> np.ndarray:
    """Per cluster, the root mean square distance of its cells from their center of mass."""
    rows, cols = _coordinates(occupied.shape[-1])
    mass = occupied.reshape(len(occupied), -1).astype(np.float64)
    n = np.maximum(mass.sum(axis=1), 1)
    r, c = rows.ravel(), cols.ravel()
    mean_r, mean_c = mass @ r / n, mass @ c / n
    square = mass @ (r**2 + c**2) / n
    return np.sqrt(np.clip(square - mean_r**2 - mean_c**2, 0, None))


def fractal_dimension(occupied: np.ndarray, low: float = 2.0, high_factor: float = 1.0, radii: int = 12) -> np.ndarray:
    """Per cluster, the mass dimension: the slope of log(cells within radius r of the seed) against log r.

    The fit uses ``radii`` radii spaced evenly in log from ``low`` to ``high_factor`` times the cluster's
    radius of gyration, inside the part that has stopped growing. Large DLA clusters at ``p`` = 1 reach
    about 1.71; at 600 particles this estimate gives about 1.63, rising to about 1.84 at ``p`` = 0.05.
    Noisy for a single cluster; compare distributions.
    """
    size = occupied.shape[-1]
    rows, cols = _coordinates(size)
    distance = np.hypot(rows - size // 2, cols - size // 2).ravel()
    order = np.argsort(distance)
    sorted_distance = distance[order]
    flat = occupied.reshape(len(occupied), -1)[:, order].astype(np.int64)
    cumulative = np.cumsum(flat, axis=1)
    out = np.full(len(occupied), np.nan)
    for i, high in enumerate(high_factor * gyration_radius(occupied)):
        if high <= low * 1.5:
            continue
        r = np.geomspace(low, high, radii)
        mass = cumulative[i, np.searchsorted(sorted_distance, r, side="right") - 1]
        out[i] = np.polyfit(np.log(r), np.log(np.maximum(mass, 1)), 1)[0]
    return out


def radial_density(occupied: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Mean occupancy in rings around the seed, between consecutive ``edges`` (cells), averaged over the
    clusters. Shape (len(edges) - 1,)."""
    size = occupied.shape[-1]
    rows, cols = _coordinates(size)
    ring = np.digitize(np.hypot(rows - size // 2, cols - size // 2).ravel(), edges) - 1
    mean = occupied.reshape(len(occupied), -1).mean(axis=0)
    inside = (ring >= 0) & (ring < len(edges) - 1)
    total = np.bincount(ring[inside], weights=mean[inside], minlength=len(edges) - 1)
    cells = np.bincount(ring[inside], minlength=len(edges) - 1)
    return total / np.maximum(cells, 1)


def wasserstein(a: np.ndarray, b: np.ndarray, quantiles: int = 101) -> float:
    """The 1-Wasserstein distance between two samples of a scalar: the mean gap between their quantiles.
    In the metric's own units; 0 when the distributions match. NaNs are ignored."""
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if not len(a) or not len(b):
        return float("nan")
    q = np.linspace(0, 1, quantiles)
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


def _mean_std(values: np.ndarray) -> tuple[float, float]:
    """Mean and standard deviation, ignoring NaNs (a metric undefined for a cluster). NaN if none are left."""
    values = values[np.isfinite(values)]
    return (float(values.mean()), float(values.std())) if len(values) else (float("nan"), float("nan"))


def compare(model: np.ndarray, simulator: np.ndarray) -> dict[str, float]:
    """Mean and standard deviation of a metric for the model and the simulator, and the distance between
    the two distributions. NaN values (undefined for that cluster, such as the dimension of an empty
    one) are left out."""
    model_mean, model_std = _mean_std(model)
    simulator_mean, simulator_std = _mean_std(simulator)
    return {"model_mean": model_mean, "model_std": model_std, "simulator_mean": simulator_mean,
            "simulator_std": simulator_std, "wasserstein": wasserstein(model, simulator)}


# Every per-cluster metric the suite reports, by name.
PER_CLUSTER = {
    "particles": lambda occ, start: particle_counts(occ).astype(np.float64),
    "connected_fraction": lambda occ, start: connected_fraction(occ),
    "start_kept": start_kept,
    "gyration_radius": lambda occ, start: gyration_radius(occ),
    "fractal_dimension": lambda occ, start: fractal_dimension(occ),
}
