"""The training items drawn from an erosion dataset (``splits.Split``).

A training item is a pair of snapshots of one run: a start ``z_a``, a later ``z_b``, and the elapsed time
between them. ``ErosionPairs`` serves both tracks:

- Track B: whole runs at 1 m cells (``window=None``).
- Track C: windows of a run at any cell size (``cell``, ``window``), with the change of the level above
  as a context channel (``context_cell``).

Optional extra input channels (``inputs``): the erodibility map and the start terrain's drainage area.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..config import Config
from ..seeding import noise_seed
from .hydrology import drainage_area, fill_pits
from .normalization import Scaling, change_rms, terrain_to_model
from .splits import Split

INPUTS = ("map", "drainage")  # extra input channels a model can ask for (Config.inputs)


# ---- Grids ----

def block_mean(z: np.ndarray, block: int) -> np.ndarray:
    """Mean of each ``block`` × ``block`` square of ``z`` (..., H, W). H and W must be multiples of it."""
    if block == 1:
        return z
    *lead, h, w = z.shape
    return z.reshape(*lead, h // block, block, w // block, block).mean(axis=(-3, -1))


def upsample(z: np.ndarray, block: int) -> np.ndarray:
    """Repeat each cell of ``z`` (..., H, W) into a ``block`` × ``block`` square."""
    return z if block == 1 else z.repeat(block, axis=-2).repeat(block, axis=-1)


def dihedral(z: np.ndarray, k: int) -> np.ndarray:
    """One of the 8 flips and 90° rotations of ``z`` (..., H, W), numbered 0 to 7; 0 is no change."""
    z = np.rot90(z, k % 4, axes=(-2, -1))
    return np.flip(z, axis=-1) if k >= 4 else z


# ---- Model inputs ----

def drainage_channel(z_a: np.ndarray, cell: int) -> tuple[np.ndarray, np.ndarray]:
    """The start terrain's drainage area (m²), routed over its pit-filled surface as the simulator does,
    and the input channel made from it: ``log10`` of the area in cells over ``log10`` of the window's cell
    count, so about [0, 1]. Computed on the window alone; it cannot see land upstream of the window."""
    area = drainage_area(fill_pits(z_a), cell)
    cells = area / (cell * cell)
    return area, np.log10(cells) / np.log10(cells.size)


def drainage_weights(area: np.ndarray, beta: float) -> np.ndarray:
    """Per-cell loss weights ``(A / mean A) ** beta``, scaled to mean 1 over the window."""
    w = (area / area.mean()) ** beta
    return w / w.mean()


def input_channels(z_a: np.ndarray, cell: int, inputs: tuple[str, ...], log_map: np.ndarray | None = None,
                   context: np.ndarray | None = None) -> np.ndarray:
    """A model's input channels for start heights ``z_a`` (m, at ``cell``-m cells), in a fixed order: the
    terrain, then each of ``inputs`` in ``INPUTS`` order, then the context channel if given.

    The map channel is ``log10`` of the erodibility factor on ``K`` (``log_map``; zeros if None, a uniform
    map), so at contrast 10 it spans [-0.5, 0.5]. ``context`` is already in model units. Used by
    ``ErosionPairs`` and by the world's levels.
    """
    unknown = set(inputs) - set(INPUTS)
    if unknown:
        raise ValueError(f"unknown inputs {sorted(unknown)}; choose from {INPUTS}")
    channels = [terrain_to_model(z_a)]
    if "map" in inputs:
        channels.append(np.zeros_like(z_a) if log_map is None else log_map)
    if "drainage" in inputs:
        channels.append(drainage_channel(z_a, cell)[1])
    if context is not None:
        channels.append(context)
    return np.stack(channels)


class ErosionPairs(Dataset):
    """Training items from every (run, start, end) snapshot pair of the selected runs.

    - ``cell``: model cells are ``cell`` × ``cell`` block means of the 1 m runs.
    - ``window``: None uses whole runs. Otherwise each item is a random ``window`` × ``window`` crop (in
      model cells), at least ``margin`` 1 m cells from the run's edge, with corners on the coarsest grid in
      use, so cells line up.
    - ``context_cell``: the cell size of the level above. Its true change over the crop (block means at
      ``context_cell``, repeated to ``cell``) is an extra input channel, in this model's units, as the
      world gives a level its parent's change. ``context_noise`` adds Gaussian noise to it in training,
      with that times the channel's RMS as standard deviation (experiment C4).
    - ``inputs``: extra channels from ``INPUTS``. ``drainage_weight`` > 0 adds a per-cell loss weight
      ``weight``, ``(A / mean A) ** drainage_weight`` (experiment B7).
    - ``repeats``: items per pair, each with its own crop and flip. ``augment``: a random flip and 90°
      rotation per item, since the world has no preferred direction.

    Crops, flips, and noise are a fixed function of (``seed``, item index), so items are reproducible and a
    resumed run sees the same items. ``lags`` limits the elapsed times, in snapshot steps.

    Each item is a dict: ``x`` (channels, h, w) inputs (``input_channels``); ``y`` (1, h, w) the change in
    model units, ``scaling.change.to_model(z_b - z_a, K_ref, dT)``; ``cond`` the scaled parameters; ``z_a``,
    ``z_b`` (h, w) heights in m at model cells; ``run``, ``start``, ``end``; ``y0``, ``x0`` the crop's top-left
    1 m cell; ``flip`` the flip number; and ``weight`` (1, h, w) when ``drainage_weight`` is set.
    """

    def __init__(self, data: Split, scaling: Scaling, runs: np.ndarray | None = None, lags=None, cell: int = 1,
                 window: int | None = None, context_cell: int | None = None, margin: int = 0, repeats: int = 1,
                 augment: bool = False, inputs: tuple[str, ...] = (), drainage_weight: float = 0.0,
                 context_noise: float = 0.0, seed: int = 0):
        if context_cell is not None and context_cell % cell:
            raise ValueError("context_cell must be a multiple of cell")
        self.data, self.scaling = data, scaling
        self.columns, self.times = data.columns, data.times
        self.runs = np.arange(len(data)) if runs is None else np.asarray(runs)
        count = len(self.times)
        pairs = [(a, b) for a in range(count) for b in range(a + 1, count)]
        if lags is not None:
            lags = set(int(k) for k in lags)
            pairs = [(a, b) for a, b in pairs if b - a in lags]
        self.pairs = np.array(pairs, dtype=np.int64).reshape(-1, 2)
        self.cell, self.window, self.context_cell = cell, window, context_cell
        self.margin, self.repeats, self.augment, self.seed = margin, repeats, augment, seed
        self.inputs, self.drainage_weight, self.context_noise = tuple(inputs), drainage_weight, context_noise
        size = data.snapshots.shape[-1]
        self.align = context_cell or cell
        if window is None:
            if size % self.align:
                raise ValueError(f"runs of {size}² do not divide into {self.align}-cell blocks")
        elif size - 2 * margin < window * cell:
            raise ValueError(f"runs of {size}² cannot hold a {window * cell}-cell window {margin} cells from the edge")
        elif (window * cell) % self.align:
            raise ValueError("the window must be a whole number of context cells")

    @property
    def snapshots(self) -> np.ndarray:
        return self.data.snapshots

    def __len__(self) -> int:
        return len(self.runs) * len(self.pairs) * self.repeats

    def locate(self, index):
        """(run, start, end) snapshot indices of item ``index``. Works on an array of indices too."""
        index = np.asarray(index) // self.repeats
        a, b = self.pairs[index % len(self.pairs)].T
        return self.runs[index // len(self.pairs)], a, b

    def elapsed(self, index):
        """Elapsed time (yr) of item ``index``, or of an array of indices."""
        _, a, b = self.locate(index)
        return self.times[b] - self.times[a]

    def _rng(self, index: int, role: str) -> np.random.Generator:
        """Item ``index``'s stream for one ``role`` (crop, flip, or context noise), each independent."""
        return np.random.Generator(np.random.PCG64(noise_seed("erosion-items", role, self.seed, int(index))))

    def region(self, index: int) -> tuple[int, int, int]:
        """(y0, x0, side) of item ``index``'s region of the run, in 1 m cells. ``index`` is any integer key."""
        size = self.snapshots.shape[-1]
        if self.window is None:
            return 0, 0, size
        side = self.window * self.cell
        room = (size - 2 * self.margin - side) // self.align
        y, x = self._rng(index, "crop").integers(0, room + 1, 2)
        return self.margin + self.align * int(y), self.margin + self.align * int(x), side

    def __getitem__(self, index: int) -> dict:
        run, a, b = (int(v) for v in self.locate(index))
        return self.item(run, a, b, index)

    def item(self, run: int, a: int, b: int, index: int = 0, z_a: np.ndarray | None = None) -> dict:
        """The item for run ``run`` from snapshot ``a`` to ``b``, with crop, flip, and noise drawn from key
        ``index``. ``z_a`` (1 m heights over the run) replaces the start snapshot, for chained calls whose
        start is a model's own output; the target still comes from the snapshots."""
        y0, x0, side = self.region(index)
        flip = int(self._rng(index, "flip").integers(8)) if self.augment else 0
        crop = (slice(y0, y0 + side), slice(x0, x0 + side))
        fine_a, fine_b = (np.asarray(self.snapshots[run, t][crop], dtype=np.float64) for t in (a, b))
        if z_a is not None:
            fine_a = np.asarray(z_a, dtype=np.float64)[crop]
        z_a, z_b = block_mean(fine_a, self.cell), block_mean(fine_b, self.cell)
        dT, K_ref = self.times[b] - self.times[a], self.columns["K_ref"][run]
        log_map = None
        if "map" in self.inputs:
            log_map = block_mean(np.log10(self.data.erodibility_map(run).factors(y0, x0, side, side)), self.cell)
        context = None
        if self.context_cell is not None:
            coarse = block_mean(fine_b - fine_a, self.context_cell)
            context = self.scaling.change.to_model(upsample(coarse, self.context_cell // self.cell), K_ref, dT)
            if self.context_noise:
                rms = np.sqrt(np.mean(context**2))
                context = context + self.context_noise * rms * self._rng(index, "noise").standard_normal(context.shape)
        x = input_channels(z_a, self.cell, self.inputs, log_map, context)
        y = self.scaling.change.to_model(z_b - z_a, K_ref, dT)[None]
        fields = {"x": x, "y": y, "z_a": z_a, "z_b": z_b}
        if self.drainage_weight:
            fields["weight"] = drainage_weights(drainage_channel(z_a, self.cell)[0], self.drainage_weight)[None]
        item = {k: torch.from_numpy(np.ascontiguousarray(dihedral(v, flip)).astype(np.float32)) for k, v in fields.items()}
        cond = self.scaling.parameters.scale(self.columns, run, dT)
        return item | {"cond": torch.from_numpy(cond.astype(np.float32)), "run": run, "start": a, "end": b,
                       "y0": y0, "x0": x0, "flip": flip}


def make_datasets(cfg: Config, data_root: Path | None = None) -> tuple[ErosionPairs, ErosionPairs, Scaling, float]:
    """Erosion training and validation items for ``cfg``, the ``Scaling`` fitted on the training runs (on
    ``cfg.cell`` block means), and ``sigma_data``, the RMS of the training target."""
    train_data, val_data = (Split.open(cfg.data_profile, split, data_root) for split in ("train", "val"))
    train_runs, val_runs = (data.select(**dict(cfg.data)) for data in (train_data, val_data))
    train_runs = train_runs[: cfg.max_runs]
    snapshots, columns, times = train_data.snapshots, train_data.columns, train_data.times
    scaling = Scaling.fit(snapshots, columns, train_runs, times, block=cfg.cell)
    sigma_data = change_rms(snapshots, columns, train_runs, times, scaling.change, block=cfg.cell)
    common = dict(lags=cfg.lags, cell=cfg.cell, window=cfg.window, context_cell=cfg.context_cell, margin=cfg.margin,
                  inputs=cfg.inputs, seed=cfg.seed)
    train = ErosionPairs(train_data, scaling, train_runs, repeats=cfg.repeats, augment=cfg.augment,
                         drainage_weight=cfg.drainage_weight, context_noise=cfg.context_noise, **common)
    val = ErosionPairs(val_data, scaling, val_runs, **common)
    return train, val, scaling, sigma_data
