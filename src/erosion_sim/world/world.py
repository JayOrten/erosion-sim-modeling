"""The infinite eroded world: give a seed, parameters, and an elapsed time, query any region (infinite-plan step 8).

``World`` builds the two levels once (``world.level.Level``): the coarse level on 4 m cells, then the
fine level on 1 m cells, which reads the coarse change as context. A query only runs the windows that
touch it, and their results are cached. Everything is a fixed function of the seed, the region, the
parameters, and ``dT``, so query order and the cache do not change any result.
"""

from pathlib import Path

import numpy as np
from erosion_terrain import Terrain

from ..paths import checkpoint_path
from ..train import load_checkpoint
from .data import coarse_heights
from .level import SPLIT, Level, coarse_context, world_params


def load_levels(coarse: Path | None = None, fine: Path | None = None, device: str | None = None) -> list[tuple]:
    """``[(model, scaling, config)]`` for the coarse and fine level, from checkpoint files; by default
    ``diffusion-world_coarse.pt`` and ``diffusion-world_fine.pt``."""
    coarse = coarse or checkpoint_path("world_coarse", "diffusion")
    fine = fine or checkpoint_path("world_fine", "diffusion")
    return [load_checkpoint(path, device=device)[:3] for path in (coarse, fine)]


class World:
    """Eroded terrain for any region of an unbounded world.

    - ``seed``: picks the diffusion noise of both levels, and the starting terrain unless ``terrain`` is given.
    - ``dT``: elapsed time in yr. ``K_ref``, ``D``, ``m``: erosion parameters, the same everywhere (see
      ``world_params``, and datagen's config for the ranges the models were trained on). No uplift.
    - ``coarse``, ``fine``: ``(model, scaling, config)`` for each level, as from ``load_checkpoint``.
      Use ``World.load`` to read them from checkpoint files.
    - ``terrain``: the starting terrain. None uses Perlin fBm with the world's defaults (``Terrain``), with
      ``persistence`` and the fine model's height scale. Any ``Terrain`` works; the models take whatever
      terrain they are given.
    - ``coarse_window``: coarse window in coarse cells (32: a whole training run). The fine window is
      the fine model's training crop.
    - ``splits``, ``steps``, ``batch_size``, ``bf16``: the sampler phases, step count, windows per model
      call, and 16-bit autocast of both levels (see ``Level``).

    Cells are 1 m. Row ``y`` and column ``x`` are integer cells and may be negative. FastNoiseLite's
    32-bit coordinates keep the terrain smooth up to about 10^5 cells from the origin.
    """

    def __init__(self, seed: int, dT: float, coarse: tuple, fine: tuple, K_ref: float = 2.5e-4, D: float = 2.5e-4,
                 m: float = 0.45, persistence: float = 0.5, coarse_window: int = 32,
                 splits: tuple[float, ...] = (SPLIT,), steps: int | None = None, batch_size: int = 64,
                 bf16: bool = False, terrain: Terrain | None = None):
        (coarse_model, coarse_scaling, coarse_cfg), (fine_model, fine_scaling, fine_cfg) = coarse, fine
        if coarse_cfg.level != "coarse" or fine_cfg.level != "fine":
            raise ValueError("coarse and fine must be models trained with level='coarse' and level='fine'")
        if coarse_cfg.block != fine_cfg.block:
            raise ValueError("the two levels must use the same block size")
        self.seed, self.dT, self.block = seed, float(dT), fine_cfg.block
        self.params = world_params(K_ref=K_ref, D=D, m=m)
        self.terrain = terrain or Terrain(seed, persistence=persistence)
        sampler = {"batch_size": batch_size, "bf16": bf16} | ({} if steps is None else {"steps": steps})
        self.coarse = Level(coarse_model, coarse_scaling, self._coarse_terrain, seed, "coarse", self.params, self.dT,
                            coarse_window, splits, **sampler)
        self.fine = Level(fine_model, fine_scaling, self.terrain.heights, seed, "fine", self.params, self.dT,
                          fine_cfg.crop, splits, context=coarse_context(self.coarse, fine_scaling, self.block),
                          **sampler)

    @classmethod
    def load(cls, seed: int, dT: float, coarse: Path | None = None, fine: Path | None = None,
             device: str | None = None, **kwargs) -> "World":
        """A world from checkpoint files; by default ``diffusion-world_coarse.pt`` and ``diffusion-world_fine.pt``."""
        return cls(seed, dT, *load_levels(coarse, fine, device), **kwargs)

    def _coarse_terrain(self, y0: int, x0: int, h: int, w: int) -> np.ndarray:
        return coarse_heights(self.terrain, y0, x0, h, w, block=self.block)

    def get(self, y0: int, x0: int, h: int, w: int) -> np.ndarray:
        """Eroded heights (m) on cells ``y0 .. y0 + h - 1`` by ``x0 .. x0 + w - 1``."""
        return self.fine.heights(y0, x0, h, w)

    def change(self, y0: int, x0: int, h: int, w: int) -> np.ndarray:
        """``Δz`` (m) on the same cells: ``get`` minus the starting terrain."""
        return self.fine.change(y0, x0, h, w)

    def coarse_change(self, y0: int, x0: int, h: int, w: int) -> np.ndarray:
        """The coarse level's ``Δz`` (m) on coarse cells (``block`` fine cells wide)."""
        return self.coarse.change(y0, x0, h, w)

    def clear_cache(self) -> None:
        """Drop every cached window of both levels. Later queries recompute them, with the same values."""
        self.coarse.clear_cache()
        self.fine.clear_cache()
