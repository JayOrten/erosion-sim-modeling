"""Procedural starting terrain: Perlin noise summed over octaves (fBm)."""

import numpy as np


def _fade(t: np.ndarray) -> np.ndarray:
    """Perlin's smoothstep 6t^5 - 15t^4 + 10t^3, flat at 0 and 1."""
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def perlin(shape: tuple[int, int], period: float, rng: np.random.Generator) -> np.ndarray:
    """One layer of 2D Perlin noise with lattice cells ``period`` pixels wide. Values are in about [-0.7, 0.7]."""
    rows, cols = shape
    lattice = (int(rows / period) + 2, int(cols / period) + 2)
    angles = rng.uniform(0.0, 2.0 * np.pi, lattice)
    grad_y, grad_x = np.sin(angles), np.cos(angles)

    y, x = np.meshgrid(np.arange(rows) / period, np.arange(cols) / period, indexing="ij")
    i0, j0 = np.floor(y).astype(int), np.floor(x).astype(int)
    fy, fx = y - i0, x - j0

    def corner(di: int, dj: int) -> np.ndarray:
        # Dot product of the corner's gradient with the offset from that corner.
        return grad_y[i0 + di, j0 + dj] * (fy - di) + grad_x[i0 + di, j0 + dj] * (fx - dj)

    u, v = _fade(fx), _fade(fy)
    top = corner(0, 0) + u * (corner(0, 1) - corner(0, 0))
    bottom = corner(1, 0) + u * (corner(1, 1) - corner(1, 0))
    return top + v * (bottom - top)


def fbm(
    shape: tuple[int, int],
    seed: int = 0,
    period: float = 128.0,
    octaves: int = 5,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
) -> np.ndarray:
    """Sum ``octaves`` Perlin layers. Each layer is ``lacunarity`` times finer and ``persistence`` times weaker.

    Returns values rescaled to [0, 1].
    """
    rng = np.random.default_rng(seed)
    total = np.zeros(shape)
    amplitude = 1.0
    for _ in range(octaves):
        total += amplitude * perlin(shape, period, rng)
        period /= lacunarity
        amplitude *= persistence
    return (total - total.min()) / (total.max() - total.min())
