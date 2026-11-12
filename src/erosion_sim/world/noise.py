"""Diffusion noise for the infinite world: unit Gaussian noise as a function of (seed, level, y, x).

A diffusion sample starts from Gaussian noise. In the world, overlapping windows must start from the
same noise where they overlap, so the noise belongs to world coordinates, not to windows
(docs/plans/infinite-plan.md, step 2).

Gaussian noise has no formula per cell, so the plane is split into fixed square tiles. Each tile has
its own random stream, seeded from SHA-256 of (seed, level, tile row, tile column) into NumPy's PCG64.
A query reads whichever tiles it touches. This follows ``gaussian_noise_patch`` in terrain-diffusion's
``world_pipeline.py``.

The noise never depends on the elapsed time ``dT``, so every ``dT`` starts from the same noise.
"""

import numpy as np

from ..diffusion import noise_seed

TILE = 64


def diffusion_noise(seed: int, level: str, y0: int, x0: int, h: int, w: int) -> np.ndarray:
    """The noise on cells ``y0 .. y0 + h - 1`` by ``x0 .. x0 + w - 1``: float32, shape (h, w).

    ``level`` names the hierarchy level (``"coarse"`` or ``"fine"``), so each level has its own field.
    Coordinates are integer cells of that level and may be negative.
    """
    tile = TILE
    out = np.empty((h, w), dtype=np.float32)
    for row in range(y0 // tile, (y0 + h - 1) // tile + 1):
        for col in range(x0 // tile, (x0 + w - 1) // tile + 1):
            # The part of this tile inside the query, in world cells.
            top, bottom = max(y0, row * tile), min(y0 + h, (row + 1) * tile)
            left, right = max(x0, col * tile), min(x0 + w, (col + 1) * tile)
            # Tile (row, col) of the field, from its own random stream.
            rng = np.random.Generator(np.random.PCG64(noise_seed("world", seed, level, row, col, tile)))
            piece = rng.standard_normal((tile, tile), dtype=np.float32)
            out[top - y0:bottom - y0, left - x0:right - x0] = piece[top - row * tile:bottom - row * tile,
                                                                     left - col * tile:right - col * tile]
    return out
