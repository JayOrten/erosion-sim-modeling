"""The infinite eroded world: give a seed, parameters, and an elapsed time, query any region.

``World`` stacks trained levels from coarsest to finest (``world.level.Level``). Each level's cells are a
whole number of times smaller than its parent's, and each level below the top reads its parent's change
as context. A query only runs the windows that touch it, and their results are cached. Everything is a
fixed function of the seed, the region, the parameters, and ``dT``, so query order and the cache do not
change any result.
"""

from pathlib import Path

import numpy as np
from erosion_terrain import ErodibilityMap, Terrain

from ..data.pairs import block_mean
from ..training.checkpoints import load_checkpoint
from .level import SPLIT, Level, parent_context, world_params


def load_levels(folders: list[Path], device: str | None = None) -> list[tuple]:
    """``[(model, scaling, config)]`` for each level, coarsest first, from model folders."""
    return [load_checkpoint(folder, device=device)[:3] for folder in folders]


def check_stack(configs: list) -> None:
    """Raise unless the configs form a stack: cells shrink by whole factors, each level below the top was
    trained with its parent's cell as ``context_cell``, the top without context, and every window's
    half (the stride) is a whole number of parent cells."""
    for k, cfg in enumerate(configs):
        if cfg.window is None:
            raise ValueError(f"level {k} was trained on whole runs; world levels are trained on windows")
        if k == 0:
            if cfg.context_cell is not None:
                raise ValueError("the top level must be trained without context")
            continue
        parent = configs[k - 1]
        if parent.cell <= cfg.cell or parent.cell % cfg.cell:
            raise ValueError(f"level {k}'s cells ({cfg.cell} m) must divide its parent's ({parent.cell} m)")
        if cfg.context_cell != parent.cell:
            raise ValueError(f"level {k} was trained with context at {cfg.context_cell} m, not its parent's "
                             f"{parent.cell} m")
        if (cfg.window // 2) % (parent.cell // cfg.cell):
            raise ValueError(f"level {k}'s stride must be a whole number of parent cells")


class World:
    """Eroded terrain for any region of an unbounded world.

    - ``seed``: picks the diffusion noise of every level, and the starting terrain unless ``terrain`` is given.
    - ``dT``: elapsed time in yr. ``K_ref``, ``D``, ``m``: erosion parameters, the same everywhere (see
      ``world_params``, and datagen's groups for the ranges the models were trained on).
    - ``levels``: ``(model, scaling, config)`` per level, coarsest first, as from ``load_checkpoint``. Use
      ``World.load`` to read them from model folders.
    - ``terrain``: the starting terrain. None uses Perlin fBm with ``Terrain``'s defaults. ``erodibility``:
      the erodibility map; None is uniform.
    - ``splits``, ``steps``, ``batch_size``, ``bf16``: the sampler phases, step count, windows per model call,
      and 16-bit autocast of every level (see ``Level``).

    ``get`` and ``change`` work on the finest level's cells (1 m for a stack ending at 1 m). Row ``y`` and
    column ``x`` are integer cells and may be negative. FastNoiseLite's 32-bit coordinates keep the terrain
    smooth up to about 10^5 cells from the origin.
    """

    def __init__(self, seed: int, dT: float, levels: list[tuple], K_ref: float = 2.5e-4, D: float = 2.5e-4,
                 m: float = 0.45, terrain: Terrain | None = None, erodibility: ErodibilityMap | None = None,
                 splits: tuple[float, ...] = (SPLIT,), steps: int | None = None, batch_size: int = 64,
                 bf16: bool = False):
        configs = [cfg for _, _, cfg in levels]
        check_stack(configs)
        self.seed, self.dT = seed, float(dT)
        self.params = world_params(K_ref=K_ref, D=D, m=m)
        self.terrain = terrain or Terrain(seed)
        self.erodibility = erodibility or ErodibilityMap()
        sampler = {"batch_size": batch_size, "bf16": bf16} | ({} if steps is None else {"steps": steps})
        self.levels: list[Level] = []
        for k, (model, scaling, cfg) in enumerate(levels):
            context = None
            if k:
                parent = self.levels[-1]
                context = parent_context(parent, scaling, parent.cfg.cell // cfg.cell)
            self.levels.append(Level(model, scaling, cfg, self._terrain(cfg.cell), seed, f"level{k}-{cfg.cell}m",
                                     self.params, self.dT, splits, context=context, log_map=self._log_map(cfg.cell),
                                     **sampler))

    @classmethod
    def load(cls, seed: int, dT: float, folders: list[Path], device: str | None = None, **kwargs) -> "World":
        """A world from model folders, coarsest level first."""
        return cls(seed, dT, load_levels(folders, device), **kwargs)

    def _terrain(self, cell: int):
        def heights(y0: int, x0: int, h: int, w: int) -> np.ndarray:
            return block_mean(self.terrain.heights(cell * y0, cell * x0, cell * h, cell * w), cell)
        return heights

    def _log_map(self, cell: int):
        def log_map(y0: int, x0: int, h: int, w: int) -> np.ndarray:
            return block_mean(np.log10(self.erodibility.factors(cell * y0, cell * x0, cell * h, cell * w)), cell)
        return log_map

    @property
    def finest(self) -> Level:
        return self.levels[-1]

    def get(self, y0: int, x0: int, h: int, w: int) -> np.ndarray:
        """Eroded heights (m) on the finest level's cells ``y0 .. y0 + h - 1`` by ``x0 .. x0 + w - 1``."""
        return self.finest.heights(y0, x0, h, w)

    def change(self, y0: int, x0: int, h: int, w: int) -> np.ndarray:
        """``Δz`` (m) on the same cells: ``get`` minus the starting terrain."""
        return self.finest.change(y0, x0, h, w)

    def clear_cache(self) -> None:
        """Drop every cached window of every level. Later queries recompute them, with the same values."""
        for level in self.levels:
            level.clear_cache()
