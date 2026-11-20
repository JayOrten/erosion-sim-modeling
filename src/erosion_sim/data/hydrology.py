"""Water routing on a heightmap, in the simulator's way, compiled with Numba.

The simulator routes water with D8 (each cell drains to its steepest of 8 neighbors) over a pit-filled
copy of the terrain. Here:

- ``receivers``, ``drainage_area``: D8 routing on the surface as given. Water that reaches a pit stops,
  so pits in a model's output show up as broken rivers.
- ``fill_pits``: priority-flood filling with a tiny slope on filled areas, so every cell drains to an
  edge. Used as post-processing for model output (experiment B7) and before routing a start terrain.
- ``pits``: interior cells with no lower neighbor.
- ``slope_area_exponent``: the slope-area fit (Flint's law) on channel cells.

Edge cells are outlets: they drain nowhere, as the simulator's fixed edges do.
"""

import heapq

import numpy as np
from numba import njit

_DY = np.array([-1, -1, -1, 0, 0, 1, 1, 1])
_DX = np.array([-1, 0, 1, -1, 1, -1, 0, 1])
_DIST = np.sqrt(_DY.astype(np.float64) ** 2 + _DX.astype(np.float64) ** 2)


@njit(cache=True)
def _receivers(z: np.ndarray, cell_size: float) -> np.ndarray:
    rows, cols = z.shape
    out = np.arange(rows * cols)
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            best = 0.0
            for k in range(8):
                ni, nj = i + _DY[k], j + _DX[k]
                slope = (z[i, j] - z[ni, nj]) / (cell_size * _DIST[k])
                if slope > best:
                    best = slope
                    out[i * cols + j] = ni * cols + nj
    return out


def receivers(z: np.ndarray, cell_size: float = 1.0) -> np.ndarray:
    """Flat index of the cell each cell drains to: its steepest downhill neighbor (D8). Edge cells and
    pits drain to themselves."""
    return _receivers(np.ascontiguousarray(z, dtype=np.float64), float(cell_size))


@njit(cache=True)
def _accumulate(z_flat: np.ndarray, target: np.ndarray, cell_area: float) -> np.ndarray:
    area = np.full(z_flat.size, cell_area)
    for node in np.argsort(-z_flat):  # highest first, so a cell is complete before it passes on
        if target[node] != node:
            area[target[node]] += area[node]
    return area


def drainage_area(z: np.ndarray, cell_size: float = 1.0) -> np.ndarray:
    """Area (m²) of land that drains through each cell, including the cell itself."""
    z = np.ascontiguousarray(z, dtype=np.float64)
    return _accumulate(z.ravel(), _receivers(z, float(cell_size)), cell_size * cell_size).reshape(z.shape)


def pits(z: np.ndarray) -> np.ndarray:
    """Boolean mask of interior cells with no lower neighbor."""
    own = np.arange(z.size).reshape(z.shape)
    mask = receivers(z).reshape(z.shape) == own
    mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = False
    return mask


def fill_pits(z: np.ndarray, epsilon: float = 1e-4) -> np.ndarray:
    """A copy of ``z`` with every pit filled to its spill point, plus ``epsilon`` (m) per cell along each
    filled path, so water always has a way down to an edge (priority flood, Barnes et al. 2014)."""
    return _fill(np.ascontiguousarray(z, dtype=np.float64), float(epsilon))


@njit(cache=True)
def _fill(z: np.ndarray, epsilon: float) -> np.ndarray:
    rows, cols = z.shape
    out = z.copy()
    done = np.zeros((rows, cols), dtype=np.bool_)
    heap = [(0.0, 0, 0)]
    heap.pop()
    for i in range(rows):
        for j in range(cols):
            if i == 0 or j == 0 or i == rows - 1 or j == cols - 1:
                done[i, j] = True
                heapq.heappush(heap, (out[i, j], i, j))
    while heap:
        h, i, j = heapq.heappop(heap)
        for k in range(8):
            ni, nj = i + _DY[k], j + _DX[k]
            if 0 <= ni < rows and 0 <= nj < cols and not done[ni, nj]:
                done[ni, nj] = True
                if out[ni, nj] <= h + epsilon:
                    out[ni, nj] = h + epsilon
                heapq.heappush(heap, (out[ni, nj], ni, nj))
    return out


def slope_area_exponent(z: np.ndarray, threshold: float, cell_size: float = 1.0, min_cells: int = 20) -> float:
    """The exponent ``theta`` in slope ∝ area^(-theta), fitted on channel cells of ``z``.

    The stream power law with n = 1 predicts ``theta = m`` in steady channels (Flint's law). Channel cells
    are those with drainage area at least ``threshold`` (m²) and a downhill receiver; slope is the drop
    to the receiver over the distance. ``z`` is pit-filled first, as the simulator routes water. Returns
    NaN with fewer than ``min_cells`` channel cells.
    """
    filled = fill_pits(z)
    target = receivers(filled, cell_size)
    area = _accumulate(filled.ravel(), target, cell_size * cell_size)
    flat = filled.ravel()
    cols = z.shape[1]
    dist = cell_size * np.hypot(target // cols - np.arange(z.size) // cols, target % cols - np.arange(z.size) % cols)
    slope = np.where(dist > 0, (flat - flat[target]) / np.where(dist > 0, dist, 1), 0)
    keep = (area >= threshold) & (slope > 1e-6)
    if keep.sum() < min_cells:
        return float("nan")
    return float(-np.polyfit(np.log(area[keep]), np.log(slope[keep]), 1)[0])
