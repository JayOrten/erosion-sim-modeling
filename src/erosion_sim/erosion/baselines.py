"""Non-learned baselines for the erosion tracks.

- ``analytical_stream_power``: the stream power law solved exactly, as in Tzathas, Gailleton, Steer &
  Cordonnier (Eurographics 2024). With n = 1 the law ``dz/dt = -K A^m |∇z|`` moves heights upstream along
  the drainage network at speed ``K A^m``: after a time ``dT``, a cell has the height its flow path had
  ``dT`` of travel time downstream. Travel time is ``τ(i) = τ(receiver) + distance / (K_i A_i^m)``,
  counted from the outlets at the fixed edges, which keep their heights.

  Simplifications against the simulator: the drainage network is fixed at the start (routed over the
  pit-filled start terrain), and there is no hillslope diffusion (``D``). So the baseline is exact for
  channel incision alone and misses hillslope smoothing and river capture.
- ``chunked``: any bounded method run on separate closed chunks of a region, stitched (track C).
- ``blended``: a bounded method on chunks at half-chunk stride, averaged with tent weights (track C).
"""

from typing import Callable

import numpy as np
from numba import njit

from ..data.hydrology import _accumulate, _receivers, fill_pits


@njit(cache=True)
def _travel_heights(z: np.ndarray, target: np.ndarray, tau: np.ndarray, dT: float) -> np.ndarray:
    """For each node, the height at travel time ``tau - dT`` along its flow path, linearly interpolated
    between nodes, or the outlet's height if the path reaches the outlet first."""
    out = np.empty_like(z)
    for i in range(z.size):
        goal = tau[i] - dT
        j = i
        while target[j] != j and tau[target[j]] > goal:
            j = target[j]
        if target[j] == j:
            out[i] = z[j]
        else:
            k = target[j]
            span = tau[j] - tau[k]
            frac = (tau[j] - goal) / span if span > 0 else 1.0
            out[i] = z[j] + frac * (z[k] - z[j])
    return out


@njit(cache=True)
def _travel_time(z: np.ndarray, target: np.ndarray, rate: np.ndarray, cols: int, cell: float) -> np.ndarray:
    tau = np.zeros(z.size)
    for node in np.argsort(z):  # lowest first, so each receiver's time is known before its donors'
        r = target[node]
        if r != node:
            dy, dx = node // cols - r // cols, node % cols - r % cols
            tau[node] = tau[r] + cell * np.sqrt(dy * dy + dx * dx) / rate[node]
    return tau


def analytical_stream_power(z0: np.ndarray, K: float, m: float, dT: float, K_map: np.ndarray | None = None,
                            cell_size: float = 1.0) -> np.ndarray:
    """Heights (m) after ``dT`` years of stream power erosion (n = 1, no diffusion) from ``z0``, with
    erodibility ``K`` times ``K_map`` per cell. Edges stay fixed, as in the simulator."""
    filled = fill_pits(z0).ravel()
    target = _receivers(filled.reshape(z0.shape), float(cell_size))
    area = _accumulate(filled, target, cell_size * cell_size)
    rate = K * (1.0 if K_map is None else np.asarray(K_map, dtype=np.float64).ravel()) * area**m
    tau = _travel_time(filled, target, rate, z0.shape[1], float(cell_size))
    return _travel_heights(filled, target, tau, float(dT)).reshape(z0.shape)


def tent_weights(size: int) -> np.ndarray:
    """1 at the chunk center, falling linearly to 0.001 at its edge, in each axis (as ``world.level.tent``)."""
    mid = (size - 1) / 2
    w = 1 - 0.999 * np.abs(np.arange(size) - mid) / mid
    return np.outer(w, w)


def chunked(method: Callable[[np.ndarray, tuple[int, int]], np.ndarray], z0: np.ndarray, chunk: int) -> np.ndarray:
    """Run ``method(z0_chunk, (y0, x0))`` on separate ``chunk`` × ``chunk`` pieces of ``z0`` and stitch the
    results. Each chunk is closed: its edges are outlets that the region does not have. ``z0``'s sides
    must be multiples of ``chunk``."""
    h, w = z0.shape
    if h % chunk or w % chunk:
        raise ValueError(f"a {h}×{w} region does not split into {chunk}² chunks")
    out = np.empty_like(z0, dtype=np.float64)
    for y in range(0, h, chunk):
        for x in range(0, w, chunk):
            out[y:y + chunk, x:x + chunk] = method(z0[y:y + chunk, x:x + chunk], (y, x))
    return out


def blended(method: Callable[[np.ndarray, tuple[int, int]], np.ndarray], z0: np.ndarray, chunk: int) -> np.ndarray:
    """Run ``method`` on ``chunk``² pieces at a stride of half a chunk, and average the overlapping results
    with tent weights. The region's border half-chunk is covered by fewer chunks. ``z0``'s sides must be
    multiples of half a chunk."""
    h, w = z0.shape
    stride = chunk // 2
    if h % stride or w % stride or h < chunk or w < chunk:
        raise ValueError(f"a {h}×{w} region does not split into {chunk}² chunks at stride {stride}")
    total, weight = np.zeros((h, w)), np.zeros((h, w))
    tent = tent_weights(chunk)
    for y in range(0, h - chunk + 1, stride):
        for x in range(0, w - chunk + 1, stride):
            total[y:y + chunk, x:x + chunk] += tent * method(z0[y:y + chunk, x:x + chunk], (y, x))
            weight[y:y + chunk, x:x + chunk] += tent
    return total / weight


def chunk_edge_mask(shape: tuple[int, int], chunk: int, distance: int) -> np.ndarray:
    """Cells within ``distance`` cells of an internal chunk boundary. The region's own edges are not
    chunk boundaries."""
    near = np.zeros(shape, dtype=bool)
    for axis, size in enumerate(shape):
        coord = np.arange(size)
        line = np.zeros(size, dtype=bool)
        for boundary in range(chunk, size, chunk):  # the boundary lies between cells boundary - 1 and boundary
            line |= (coord >= boundary - distance) & (coord < boundary + distance)
        near |= line[:, None] if axis == 0 else line[None, :]
    return near
