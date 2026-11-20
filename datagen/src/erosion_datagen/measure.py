"""Measurements that set data and window sizes (implementation plan, step 8). Results go in
docs/notes/datagen-measurements.md.

- ``feature_size``: the dominant wavelength of a terrain, as the peak of its variance per log wavenumber.
- ``edge_influence``: how far the simulator's fixed edges change the result, by running a region alone
  and as the middle of a larger run.
- ``cell_steps_per_second``: simulator speed at a grid size.
"""

import time

import numpy as np
from erosion_terrain import Terrain

from .simulate import simulate


def radial_spectrum(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Radially averaged power spectrum of a square field (mean removed, Hann window). Returns
    (wavenumber in cycles per cell, power), for wavenumbers 1 to n/2 cycles across the field."""
    n = z.shape[0]
    window = np.outer(np.hanning(n), np.hanning(n))
    power = np.abs(np.fft.fft2((z - z.mean()) * window)) ** 2
    k = np.hypot(*np.meshgrid(np.fft.fftfreq(n), np.fft.fftfreq(n), indexing="ij")) * n
    rings = np.rint(k).astype(int)
    total = np.bincount(rings.ravel(), power.ravel())
    count = np.bincount(rings.ravel())
    keep = np.arange(1, n // 2)
    return keep / n, total[keep] / count[keep]


def feature_size(z: np.ndarray) -> float:
    """The dominant wavelength (cells): where wavenumber × power peaks, which is where the most height
    variance sits per octave of scale."""
    k, power = radial_spectrum(z)
    return float(1 / k[np.argmax(k * power)])


def feature_sizes(period: float = 48.0, size: int = 512, seeds: int = 12) -> dict[str, dict[str, float]]:
    """Median feature size, and its ratio to ``period``, for each of the 12 noise combinations, over
    ``seeds`` terrains of ``size``² with the other settings at the middle of the main groups' ranges."""
    out = {}
    for noise in ("perlin", "opensimplex2", "value_cubic"):
        for fractal in ("fbm", "ridged"):
            for warp in (0.0, 0.3):
                sizes = [feature_size(Terrain(s, period=period, octaves=5, persistence=0.5, lacunarity=2.1, noise=noise,
                                              fractal=fractal, warp=warp).heights(0, 0, size, size)) for s in range(seeds)]
                name = f"{noise}/{fractal}/{'warp' if warp else 'no warp'}"
                out[name] = {"cells": float(np.median(sizes)), "per_period": float(np.median(sizes) / period),
                             "spread": float(np.percentile(sizes, 90) / np.percentile(sizes, 10))}
    return out


def edge_influence(terrain: Terrain, K: float, D: float, m: float, inner: int = 256, outer: int = 512,
                   steps: int = 500, dt: float = 100.0) -> dict[str, np.ndarray]:
    """Run the middle ``inner``² of a terrain alone (its edges fixed) and as part of an ``outer``² run.

    Returns, for each distance from the inner region's edge (cells), the RMS difference between the two
    final height changes, and the RMS height change of the larger run there. Where the difference is
    small next to the change, the fixed edges no longer matter.
    """
    offset = (outer - inner) // 2
    z_outer = terrain.heights(-offset, -offset, outer, outer)
    big = simulate(z_outer, K, D, m, 0.0, dt=dt, steps=steps, snapshot_every=steps)
    alone = simulate(z_outer[offset:offset + inner, offset:offset + inner], K, D, m, 0.0, dt=dt, steps=steps,
                     snapshot_every=steps)
    change_big = (big[-1] - big[0])[offset:offset + inner, offset:offset + inner]
    change_alone = alone[-1] - alone[0]
    rows, cols = np.mgrid[:inner, :inner]
    distance = np.minimum.reduce([rows, cols, inner - 1 - rows, inner - 1 - cols])
    gap, change = [], []
    for d in range(inner // 2):
        ring = distance == d
        gap.append(np.sqrt(np.mean((change_alone[ring] - change_big[ring]) ** 2)))
        change.append(np.sqrt(np.mean(change_big[ring] ** 2)))
    return {"distance": np.arange(inner // 2), "difference": np.array(gap), "change": np.array(change)}


def cell_steps_per_second(size: int, steps: int = 20, period: float = 64.0) -> float:
    """Simulator speed on one process at ``size``²: grid cells times steps, per second."""
    z0 = Terrain(0, period=period).heights(0, 0, size, size)
    simulate(z0, 3e-4, 3e-4, 0.45, 0.0, dt=100.0, steps=2, snapshot_every=2)  # warm up
    start = time.perf_counter()
    simulate(z0, 3e-4, 3e-4, 0.45, 0.0, dt=100.0, steps=steps, snapshot_every=steps)
    return size * size * steps / (time.perf_counter() - start)
