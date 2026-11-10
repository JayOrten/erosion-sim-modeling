"""Training items for the two world levels, cut from the 128² simulator runs (infinite-plan step 4).

- ``ErosionCrops`` (fine level): random crops at 1 m cells. Each has a second input channel with the
  run's true coarse change over the crop, so the fine model learns to read coarse context.
- ``ErosionCoarse`` (coarse level): whole runs as 4×4 block means.

Both apply a random flip and 90° rotation, since the world has no preferred direction. The crop and
the flip are a fixed function of (seed, item index), so items are reproducible and a resumed training
run sees the same items. Both expect a ``Normalizer`` with a fixed ``height`` (step 3).
"""

import numpy as np
import torch

from ..data import ErosionPairs, Normalizer
from ..diffusion import noise_seed

BLOCK = 4  # fine cells per coarse cell, along each side


def block_mean(z: np.ndarray, block: int = BLOCK) -> np.ndarray:
    """Mean of each ``block`` × ``block`` square of ``z`` (..., H, W). H and W must be multiples of it."""
    *lead, h, w = z.shape
    return z.reshape(*lead, h // block, block, w // block, block).mean(axis=(-3, -1))


def upsample(z: np.ndarray, block: int = BLOCK) -> np.ndarray:
    """Repeat each cell of ``z`` (..., H, W) into a ``block`` × ``block`` square."""
    return z.repeat(block, axis=-2).repeat(block, axis=-1)


def dihedral(z: np.ndarray, k: int) -> np.ndarray:
    """One of the 8 flips and 90° rotations of ``z`` (..., H, W), numbered 0 to 7; 0 is no change."""
    z = np.rot90(z, k % 4, axes=(-2, -1))
    return np.flip(z, axis=-1) if k >= 4 else z


class _LevelItems(ErosionPairs):
    """Shared parts: the random choice per item and the scaling into model units."""

    def __init__(self, path, normalizer: Normalizer, runs=None, lags=None, seed: int = 0, augment: bool = True):
        if normalizer.height is None:
            raise ValueError("world levels need a Normalizer with a fixed height")
        super().__init__(path, normalizer, runs=runs, lags=lags)
        self.seed = seed
        self.augment = augment

    def _rng(self, index: int) -> np.random.Generator:
        return np.random.Generator(np.random.PCG64(noise_seed("world-items", type(self).__name__, self.seed, int(index))))

    def _item(self, index: int, z_a: np.ndarray, z_b: np.ndarray, extra: np.ndarray | None, run, a, b) -> dict:
        """The item for start ``z_a`` and end ``z_b`` (m). ``extra`` is a change in m, added as an input channel."""
        x, relief = self.normalizer.input(z_a)
        relief = float(relief)
        dT = float(self.times[b] - self.times[a])
        scale = float(self.normalizer.target_scale(self.params["K_ref"][run], dT))
        cond = self.normalizer.conditioning(self.params, np.array([run]), np.array([dT]), relief)[0]
        channels = [x] if extra is None else [x, extra / (relief * scale)]
        return {
            "x": torch.from_numpy(np.stack(channels).astype(np.float32)),
            "y": torch.from_numpy(((z_b - z_a) / (relief * scale)).astype(np.float32))[None],
            "cond": torch.from_numpy(cond.astype(np.float32)),
            "z_a": torch.from_numpy(z_a.astype(np.float32)),
            "z_b": torch.from_numpy(z_b.astype(np.float32)),
            "run": run,
            "start": a,
            "end": b,
            "relief": relief,
            "scale": scale,
        }


class ErosionCrops(_LevelItems):
    """Fine-level items: a ``crop`` × ``crop`` piece of a run at 1 m cells.

    - ``x``: (2, crop, crop). Channel 0 is the start terrain, channel 1 the run's true coarse change
      (``block`` × ``block`` means of ``z_b - z_a``), repeated back to 1 m cells. Both are in model units.
    - ``y``: (1, crop, crop), the change in model units. The other keys are as in ``ErosionPairs``,
      for the crop.

    Crop corners are multiples of ``block``, so coarse cells line up with the crop.
    """

    def __init__(self, path, normalizer, runs=None, lags=None, crop: int = 64, block: int = BLOCK, seed: int = 0,
                 augment: bool = True):
        super().__init__(path, normalizer, runs=runs, lags=lags, seed=seed, augment=augment)
        if crop % block:
            raise ValueError("crop must be a multiple of block")
        self.crop, self.block = crop, block

    def __getitem__(self, index: int) -> dict:
        run, a, b = (int(v) for v in self.locate(index))
        rng = self._rng(index)
        size = self.snapshots.shape[-1]
        y0, x0 = self.block * rng.integers(0, (size - self.crop) // self.block + 1, 2)
        k = int(rng.integers(8)) if self.augment else 0
        z_a, z_b = (dihedral(np.asarray(self.snapshots[run, t, y0:y0 + self.crop, x0:x0 + self.crop], np.float64), k)
                    for t in (a, b))
        coarse = upsample(block_mean(z_b - z_a, self.block), self.block)
        return self._item(index, z_a, z_b, coarse, run, a, b)


class ErosionCoarse(_LevelItems):
    """Coarse-level items: a whole run as ``block`` × ``block`` means, so 128² runs become 32².

    ``x`` is (1, H / block, W / block); the other keys are as in ``ErosionPairs``, at coarse cells.
    """

    def __init__(self, path, normalizer, runs=None, lags=None, block: int = BLOCK, seed: int = 0,
                 augment: bool = True):
        super().__init__(path, normalizer, runs=runs, lags=lags, seed=seed, augment=augment)
        self.block = block

    def __getitem__(self, index: int) -> dict:
        run, a, b = (int(v) for v in self.locate(index))
        k = int(self._rng(index).integers(8)) if self.augment else 0
        z_a, z_b = (dihedral(block_mean(np.asarray(self.snapshots[run, t], np.float64), self.block), k)
                    for t in (a, b))
        return self._item(index, z_a, z_b, None, run, a, b)
