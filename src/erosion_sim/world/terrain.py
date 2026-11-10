"""Starting terrain for the infinite world, at any region (infinite-plan step 1).

The noise itself is ``erosion_terrain.fbm`` (``terrain/`` at the repo root), the same function datagen
uses for the training runs. ``tests/test_world_terrain.py`` checks the world against snapshot 0 of the
``world`` dataset, which is the raw starting terrain.
"""

from dataclasses import dataclass

import numpy as np
from erosion_terrain import fbm

from .data import BLOCK, block_mean


@dataclass(frozen=True)
class Terrain:
    """The uneroded world: heights in m at 1 m cells, ``height`` times Perlin fBm.

    Defaults match the ``world`` data profile in ``datagen``, except ``persistence``, which that profile
    draws per run from 0.35 to 0.65.
    """

    seed: int
    period: float = 32.0
    persistence: float = 0.5
    octaves: int = 5
    lacunarity: float = 2.0
    height: float = 64.0

    def fine(self, y0: int, x0: int, h: int, w: int) -> np.ndarray:
        """Heights (m) on fine cells ``y0 .. y0 + h - 1`` by ``x0 .. x0 + w - 1``."""
        return self.height * fbm((h, w), self.seed, self.period, self.octaves, self.persistence, self.lacunarity,
                                 origin=(y0, x0))

    def coarse(self, y0: int, x0: int, h: int, w: int, block: int = BLOCK) -> np.ndarray:
        """Heights (m) on coarse cells: each the mean of its ``block`` × ``block`` fine cells.

        Coarse cell (i, j) covers fine cells ``block * i .. block * i + block - 1`` (and the same for j).
        """
        return block_mean(self.fine(block * y0, block * x0, block * h, block * w), block)
