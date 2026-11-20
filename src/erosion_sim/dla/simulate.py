"""Diffusion-limited aggregation (DLA) with a sticking probability.

Particles arrive one at a time. Each one wanders randomly until it touches the cluster, then sticks with
probability ``p``; otherwise it keeps wandering. One time step is one particle. Below ``p`` = 1, walkers
reach deeper before sticking, so clusters are denser and their fractal dimension rises from about 1.71
toward 2.

A run is stored as an arrival image: each occupied cell holds its arrival step (0 for the seed particle
at the center), and empty cells hold -1. The cluster after ``t`` particles is every cell with arrival
step below ``t``.

The walker is copied from ``../seeded-pa-vae/src/dla/simulate.py``: walkers start on a circle just
outside the cluster, jump across circles known to be empty, and restart if they wander too far.
"""

import hashlib
import json
import math

import numpy as np
from numba import njit

EMPTY = -1
LAUNCH_MARGIN = 5  # walkers start on a circle of radius r_max + LAUNCH_MARGIN
KILL_FACTOR = 20  # restart walkers that get farther than KILL_FACTOR * launch radius
OFFSETS = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=np.int64)


class OutgrewGrid(ValueError):
    """The cluster came within reach of the grid's edge."""


def run_seed(*key) -> int:
    """A 63-bit seed from a public key such as ``(dataset, split, index)``, by hashing its canonical JSON."""
    payload = json.dumps(["erosion-sim-dla-v1", *key], separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") >> 1


def _radius(arrival: np.ndarray, center: int) -> float:
    """Distance from the center to the farthest occupied cell."""
    rows, cols = np.nonzero(arrival != EMPTY)
    return float(np.hypot(rows - center, cols - center).max())


@njit(cache=True)
def _grow(arrival, first, last, p, r_max, rng):
    """Add particles ``first`` to ``last - 1`` to ``arrival`` in place. Returns the new r_max, or -1 if
    the cluster came within reach of the edge (``arrival`` is then partly grown)."""
    size = arrival.shape[0]
    center = size // 2
    for step in range(first, last):
        launch = r_max + LAUNCH_MARGIN
        kill = KILL_FACTOR * launch
        angle = 2 * math.pi * rng.random()
        x = center + round(launch * math.cos(angle))
        y = center + round(launch * math.sin(angle))
        while True:
            dist = math.hypot(x - center, y - center)
            if dist > kill:
                angle = 2 * math.pi * rng.random()
                x = center + round(launch * math.cos(angle))
                y = center + round(launch * math.sin(angle))
                continue
            # A circle around the walker that is known to be empty: jump to its edge.
            rho = dist - r_max - 2
            if rho > 1:
                angle = 2 * math.pi * rng.random()
                x = round(x + rho * math.cos(angle))
                y = round(y + rho * math.sin(angle))
                continue
            # Close to the cluster, so (x, y) and its neighbors are inside the grid.
            touching = False
            for k in range(4):
                if arrival[x + OFFSETS[k, 0], y + OFFSETS[k, 1]] != EMPTY:
                    touching = True
                    break
            if touching and (p >= 1.0 or rng.random() < p):
                break
            # Step to a random neighbor. A step into the cluster is refused, and the walker stays.
            k = rng.integers(0, 4)
            nx, ny = x + OFFSETS[k, 0], y + OFFSETS[k, 1]
            if arrival[nx, ny] == EMPTY:
                x, y = nx, ny
        arrival[x, y] = step
        r_max = max(r_max, math.hypot(x - center, y - center))
        # Walkers within r_max + 3 of the center index the grid, and need one more cell for neighbors.
        if r_max + 4 >= center:
            return -1.0
    return r_max


def simulate(p: float, particles: int, seed: int, size: int = 128, start: np.ndarray | None = None) -> np.ndarray:
    """Grow one cluster to ``particles`` particles and return its arrival image: int16, shape (size, size).

    ``start`` continues a stored arrival image instead of starting from the seed particle; its particles
    keep their arrival steps, and new ones continue the count. ``seed`` seeds the run's PCG64 stream.
    Raises ``OutgrewGrid`` if the cluster comes within reach of the edge.
    """
    if not 0 < p <= 1:
        raise ValueError(f"p must be in (0, 1], got {p}")
    if particles >= np.iinfo(np.int16).max:
        raise ValueError("arrival steps are stored as int16")
    center = size // 2
    if start is None:
        arrival = np.full((size, size), EMPTY, dtype=np.int16)
        arrival[center, center] = 0
    else:
        if start.shape != (size, size):
            raise ValueError(f"start has shape {start.shape}, expected {(size, size)}")
        arrival = start.astype(np.int16, copy=True)
    first = int((arrival != EMPTY).sum())
    if first > particles:
        raise ValueError(f"start already has {first} particles, more than {particles}")
    rng = np.random.Generator(np.random.PCG64(seed))
    r_max = _grow(arrival, first, particles, float(p), _radius(arrival, center), rng)
    if r_max < 0:
        raise OutgrewGrid(f"a cluster with p = {p} reached the edge of a {size}² grid before {particles} particles")
    return arrival


def occupancy(arrival: np.ndarray, t) -> np.ndarray:
    """The cluster after ``t`` particles: 1 where occupied, 0 where empty. Works on stacked images, with
    ``t`` broadcast against the leading dimensions."""
    t = np.asarray(t)[(...,) + (None,) * 2]
    return ((arrival != EMPTY) & (arrival < t)).astype(np.uint8)
