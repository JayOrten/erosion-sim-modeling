"""Generated erosion datasets, and training pairs drawn from them.

A dataset comes from ``datagen/`` (see ``datagen/README.md``): one folder per split,
``artifacts/data/<profile>/<split>/``, holding the run table (``runs.npz``) and the snapshots. ``Split`` opens one. Each sample in it is one simulation: a
starting terrain (a ``Terrain``), erosion parameters, and its snapshots over time. A training item is a
pair of snapshots from one sample: a start ``z_a``, a later ``z_b``, and the elapsed time between them.
"""

import json
from functools import partial
from pathlib import Path

import numpy as np
import torch
from erosion_terrain import TERRAIN_FIELDS, Terrain
from torch.utils.data import Dataset

from . import table
from .config import Config
from .normalization import Scaling, change_rms, terrain_to_model
from .paths import data_root

EROSION_PARAMS = ("K_ref", "K", "D", "m", "U")


class Split:
    """One split of a generated dataset, as datagen writes it.

    - ``columns``: the run table (``table.py``), one array per column, one row per sample: ``index``, each
      erosion parameter (``EROSION_PARAMS``), and each ``Terrain`` field (``TERRAIN_FIELDS``).
    - ``times``: the snapshot times (yr), the same for every sample.
    - ``snapshots``: (samples, snapshots, rows, cols) heights in m, memory-mapped on first use.
    - ``config``: the datagen config the dataset was made with; ``meta``: all of ``meta.json``.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.columns = table.load(self.path)
        self.meta = json.loads((self.path / "meta.json").read_text())
        self.times = np.array(self.meta["times"])
        self.config = self.meta["config"]
        self._snapshots = None

    @classmethod
    def open(cls, profile: str, split: str, root: Path | None = None) -> "Split":
        """``artifacts/data/<profile>/<split>/``, or ``<root>/<split>/`` when ``root`` is given."""
        return cls(data_root(profile, root) / split)

    @property
    def snapshots(self) -> np.ndarray:
        if self._snapshots is None:
            self._snapshots = np.load(self.path / "snapshots.npy", mmap_mode="r")
        return self._snapshots

    def __getstate__(self) -> dict:
        # Each DataLoader worker opens its own memory map.
        return self.__dict__ | {"_snapshots": None}

    def __len__(self) -> int:
        return len(self.columns["index"])

    def terrain(self, i: int) -> Terrain:
        """The starting terrain of sample ``i``. Its heights at (0, 0) are snapshot 0."""
        return Terrain(**{name: self.columns[name][i].item() for name in TERRAIN_FIELDS})

    def erosion(self, i: int) -> dict[str, float]:
        """The erosion parameters of sample ``i``."""
        return {name: float(self.columns[name][i]) for name in EROSION_PARAMS}

    def select(self, **filters) -> np.ndarray:
        """Indices of samples matching every filter; see ``table.select``."""
        return table.select(self.columns, **filters)


class ErosionPairs(Dataset):
    """Every (run, start, end) snapshot pair of the selected runs, as tensors.

    Each item is a dict:

    - ``x``: (1, H, W) start heightmap, ``terrain_to_model(z_a)``.
    - ``y``: (1, H, W) target, ``scaling.change.to_model(z_b - z_a, K_ref, dT)``.
    - ``cond``: (len(PARAMETER_NAMES),) erosion parameters and elapsed time, ``scaling.parameters.scale``.
    - ``z_a``, ``z_b``: (H, W) the raw heightmaps in m, for evaluation.
    - ``run``, ``start``, ``end``: sample and snapshot indices.

    ``lags`` limits which elapsed times are used, in snapshot steps (1 = one snapshot apart). Use it to
    hold elapsed times out of training.
    """

    def __init__(self, data: Split, scaling: Scaling, runs: np.ndarray | None = None, lags=None):
        self.data = data
        self.scaling = scaling
        self.columns, self.times = data.columns, data.times
        self.runs = np.arange(len(data)) if runs is None else np.asarray(runs)
        count = len(self.times)
        pairs = [(a, b) for a in range(count) for b in range(a + 1, count)]
        if lags is not None:
            lags = set(int(k) for k in lags)
            pairs = [(a, b) for a, b in pairs if b - a in lags]
        self.pairs = np.array(pairs, dtype=np.int64).reshape(-1, 2)

    @property
    def snapshots(self) -> np.ndarray:
        return self.data.snapshots

    def __len__(self) -> int:
        return len(self.runs) * len(self.pairs)

    def locate(self, index):
        """(run, start, end) snapshot indices of item ``index``. Works on an array of indices too."""
        index = np.asarray(index)
        a, b = self.pairs[index % len(self.pairs)].T
        return self.runs[index // len(self.pairs)], a, b

    def elapsed(self, index):
        """Elapsed time (yr) of item ``index``, or of an array of indices."""
        _, a, b = self.locate(index)
        return self.times[b] - self.times[a]

    def __getitem__(self, index: int) -> dict:
        run, a, b = (int(v) for v in self.locate(index))
        z_a = np.array(self.snapshots[run, a], dtype=np.float32)  # copies out of the read-only memory map
        z_b = np.array(self.snapshots[run, b], dtype=np.float32)
        dT = self.times[b] - self.times[a]
        cond = self.scaling.parameters.scale(self.columns, run, dT)
        y = self.scaling.change.to_model(z_b - z_a, self.columns["K_ref"][run], dT)
        return {
            "x": torch.from_numpy(terrain_to_model(z_a).astype(np.float32))[None],
            "y": torch.from_numpy(y.astype(np.float32))[None],
            "cond": torch.from_numpy(cond.astype(np.float32)),
            "z_a": torch.from_numpy(z_a),
            "z_b": torch.from_numpy(z_b),
            "run": run,
            "start": a,
            "end": b,
        }


def make_datasets(cfg: Config, data_root: Path | None = None) -> tuple[ErosionPairs, ErosionPairs, Scaling, float]:
    """Erosion training and validation pairs for ``cfg``, the ``Scaling`` fitted on the training runs, and
    ``sigma_data``, the RMS of the training target.

    World levels use ``ErosionCoarse`` or ``ErosionCrops`` (``cfg.level``); the coarse level's change
    scaling and ``sigma_data`` are measured on block means, as it sees them.
    """
    from .world.data import ErosionCoarse, ErosionCrops  # world.data imports this module

    train_data, val_data = (Split.open(cfg.data_profile, split, data_root) for split in ("train", "val"))
    train_runs, val_runs = (data.select(**dict(cfg.data)) for data in (train_data, val_data))
    block = cfg.block if cfg.level == "coarse" else 1
    snapshots, params, times = train_data.snapshots, train_data.columns, train_data.times
    scaling = Scaling.fit(snapshots, params, train_runs, times, block=block)
    sigma_data = change_rms(snapshots, params, train_runs, times, scaling.change, block=block)
    if cfg.level == "coarse":
        make = partial(ErosionCoarse, block=cfg.block, crop=cfg.coarse_crop, margin=cfg.margin, seed=cfg.seed)
    elif cfg.level == "fine":
        make = partial(ErosionCrops, crop=cfg.crop, block=cfg.block, margin=cfg.margin, seed=cfg.seed)
    else:
        make = ErosionPairs
    extra = {"repeats": cfg.repeats} if cfg.level else {}
    train = make(train_data, scaling, runs=train_runs, lags=cfg.lags, **extra)
    val = make(val_data, scaling, runs=val_runs, lags=cfg.lags)
    return train, val, scaling, sigma_data
