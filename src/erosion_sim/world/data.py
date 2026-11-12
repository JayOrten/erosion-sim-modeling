"""Training items for the two world levels, cut from simulator runs (infinite-plan steps 4 and 9a).

- ``ErosionCrops`` (fine level): random crops at 1 m cells. Each has a second input channel with the
  run's true coarse change over the crop, so the fine model learns to read coarse context.
- ``ErosionCoarse`` (coarse level): whole runs, or random crops, as 4×4 block means.

``margin`` keeps crops that many cells away from the run's edge, where the simulator holds heights fixed
and drains water; the world has no such edges. ``repeats`` gives each (run, start, end) pair that many
items, each with its own crop and flip, for large runs that hold many crops.

Both apply a random flip and 90° rotation, since the world has no preferred direction. The crop and
the flip are a fixed function of (seed, item index), so items are reproducible and a resumed training
run sees the same items.
"""

import numpy as np
import torch
from erosion_terrain import Terrain

from ..data import ErosionPairs, Split
from ..diffusion import noise_seed
from ..normalization import Scaling, terrain_to_model

BLOCK = 4  # fine cells per coarse cell, along each side


def block_mean(z: np.ndarray, block: int = BLOCK) -> np.ndarray:
    """Mean of each ``block`` × ``block`` square of ``z`` (..., H, W). H and W must be multiples of it."""
    *lead, h, w = z.shape
    return z.reshape(*lead, h // block, block, w // block, block).mean(axis=(-3, -1))


def coarse_heights(terrain: Terrain, y0: int, x0: int, h: int, w: int, block: int = BLOCK) -> np.ndarray:
    """``terrain``'s heights (m) on coarse cells: each the mean of its ``block`` × ``block`` fine cells.

    Coarse cell (i, j) covers fine cells ``block * i .. block * i + block - 1`` (and the same for j).
    """
    return block_mean(terrain.heights(block * y0, block * x0, block * h, block * w), block)


def upsample(z: np.ndarray, block: int = BLOCK) -> np.ndarray:
    """Repeat each cell of ``z`` (..., H, W) into a ``block`` × ``block`` square."""
    return z.repeat(block, axis=-2).repeat(block, axis=-1)


def dihedral(z: np.ndarray, k: int) -> np.ndarray:
    """One of the 8 flips and 90° rotations of ``z`` (..., H, W), numbered 0 to 7; 0 is no change."""
    z = np.rot90(z, k % 4, axes=(-2, -1))
    return np.flip(z, axis=-1) if k >= 4 else z


class _LevelItems(ErosionPairs):
    """Shared parts: the random choice per item and the scaling into model units."""

    def __init__(self, data: Split, scaling: Scaling, runs=None, lags=None, seed: int = 0,
                 augment: bool = True, block: int = BLOCK, margin: int = 0, repeats: int = 1):
        if margin % block:
            raise ValueError("margin must be a multiple of block")
        super().__init__(data, scaling, runs=runs, lags=lags)
        self.seed, self.augment = seed, augment
        self.block, self.margin, self.repeats = block, margin, repeats

    def __len__(self) -> int:
        return super().__len__() * self.repeats

    def locate(self, index):
        """(run, start, end) snapshot indices of item ``index``; items ``k * repeats ..`` share a pair."""
        return super().locate(np.asarray(index) // self.repeats)

    def _corner(self, rng: np.random.Generator, size: int) -> np.ndarray:
        """Random (y0, x0) in fine cells for a ``size``-cell crop: multiples of block, ``margin`` from the edge."""
        room = self.snapshots.shape[-1] - 2 * self.margin - size
        if room < 0:
            raise ValueError(f"runs of {self.snapshots.shape[-1]}² cannot hold a {size}² crop {self.margin} cells "
                             "from the edge")
        return self.margin + self.block * rng.integers(0, room // self.block + 1, 2)

    def _rng(self, index: int) -> np.random.Generator:
        return np.random.Generator(np.random.PCG64(noise_seed("world-items", type(self).__name__, self.seed, int(index))))

    def _item(self, index: int, z_a: np.ndarray, z_b: np.ndarray, extra: np.ndarray | None, run, a, b,
              **more) -> dict:
        """The item for start ``z_a`` and end ``z_b`` (m). ``extra`` is a change in m, added as an input channel."""
        dT, K_ref = self.times[b] - self.times[a], self.params["K_ref"][run]
        cond = self.scaling.parameters.scale(self.params, run, dT)
        channels = [terrain_to_model(z_a)]
        if extra is not None:
            channels.append(self.scaling.change.to_model(extra, K_ref, dT))
        return {
            "x": torch.from_numpy(np.stack(channels).astype(np.float32)),
            "y": torch.from_numpy(self.scaling.change.to_model(z_b - z_a, K_ref, dT).astype(np.float32))[None],
            "cond": torch.from_numpy(cond.astype(np.float32)),
            "z_a": torch.from_numpy(z_a.astype(np.float32)),
            "z_b": torch.from_numpy(z_b.astype(np.float32)),
            "run": run,
            "start": a,
            "end": b,
            **more,
        }


class ErosionCrops(_LevelItems):
    """Fine-level items: a ``crop`` × ``crop`` piece of a run at 1 m cells.

    - ``x``: (2, crop, crop). Channel 0 is the start terrain, channel 1 the run's true coarse change
      (``block`` × ``block`` means of ``z_b - z_a``), repeated back to 1 m cells. Both are in model units.
    - ``y``: (1, crop, crop), the change in model units. The other keys are as in ``ErosionPairs``,
      for the crop, plus ``y0`` and ``x0``: the crop's top-left cell in the run (before any flip).

    Crop corners are multiples of ``block``, so coarse cells line up with the crop.
    """

    def __init__(self, data: Split, scaling, runs=None, lags=None, crop: int = 64, block: int = BLOCK, seed: int = 0,
                 augment: bool = True, margin: int = 0, repeats: int = 1):
        super().__init__(data, scaling, runs=runs, lags=lags, seed=seed, augment=augment, block=block,
                         margin=margin, repeats=repeats)
        if crop % block:
            raise ValueError("crop must be a multiple of block")
        self.crop = crop

    def __getitem__(self, index: int) -> dict:
        run, a, b = (int(v) for v in self.locate(index))
        rng = self._rng(index)
        y0, x0 = self._corner(rng, self.crop)
        k = int(rng.integers(8)) if self.augment else 0
        z_a, z_b = (dihedral(np.asarray(self.snapshots[run, t, y0:y0 + self.crop, x0:x0 + self.crop], np.float64), k)
                    for t in (a, b))
        coarse = upsample(block_mean(z_b - z_a, self.block), self.block)
        return self._item(index, z_a, z_b, coarse, run, a, b, y0=int(y0), x0=int(x0))


class ErosionCoarse(_LevelItems):
    """Coarse-level items as ``block`` × ``block`` means: whole runs (128² runs become 32²), or with
    ``crop``, a random ``crop`` × ``crop`` piece in coarse cells.

    ``x`` is (1, h, w) at coarse cells; the other keys are as in ``ErosionPairs``, at coarse cells. With
    ``crop``, also ``y0`` and ``x0``: the crop's top-left fine cell in the run (before any flip).
    """

    def __init__(self, data: Split, scaling, runs=None, lags=None, block: int = BLOCK, seed: int = 0,
                 augment: bool = True, crop: int | None = None, margin: int = 0, repeats: int = 1):
        super().__init__(data, scaling, runs=runs, lags=lags, seed=seed, augment=augment, block=block,
                         margin=margin, repeats=repeats)
        if crop is None and margin:
            raise ValueError("a margin needs a crop")
        self.crop = crop

    def __getitem__(self, index: int) -> dict:
        run, a, b = (int(v) for v in self.locate(index))
        rng = self._rng(index)
        more = {}
        if self.crop is None:
            region = (slice(None), slice(None))
        else:
            size = self.crop * self.block
            y0, x0 = self._corner(rng, size)
            region = (slice(y0, y0 + size), slice(x0, x0 + size))
            more = {"y0": int(y0), "x0": int(x0)}
        k = int(rng.integers(8)) if self.augment else 0
        z_a, z_b = (dihedral(block_mean(np.asarray(self.snapshots[run, t][region], np.float64), self.block), k)
                    for t in (a, b))
        return self._item(index, z_a, z_b, None, run, a, b, **more)
