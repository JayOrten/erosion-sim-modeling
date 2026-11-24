"""One split of a generated erosion dataset.

A dataset comes from ``datagen/`` (see ``datagen/README.md``): one folder per (group, split),
``artifacts/data/<dataset>/<group>/<split>/``, holding the run table (``runs.npz``), the snapshots, and
``meta.json``. Each run is one simulation: a starting terrain (a ``Terrain``), an erodibility map,
erosion parameters, and its snapshots over time.
"""

import json
from pathlib import Path

import numpy as np
from erosion_terrain import TERRAIN_FIELDS, ErodibilityMap, Terrain, map_from_columns

from ..paths import dataset_dir
from . import table


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
        """``artifacts/data/<profile>/<split>/``, or ``<root>/<split>/`` when ``root`` is given. For an
        ``erosion_master`` group, ``profile`` is ``"erosion_master/<group>"``."""
        return cls(dataset_dir(profile, root) / split)

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

    def erodibility_map(self, i: int) -> ErodibilityMap:
        """The erodibility map of sample ``i`` (uniform for datasets made before maps existed)."""
        if "map_shape" not in self.columns:
            return ErodibilityMap()
        return map_from_columns({k: v[i] for k, v in self.columns.items()})

    def select(self, **filters) -> np.ndarray:
        """Indices of samples matching every filter; see ``table.select``."""
        return table.select(self.columns, **filters)
