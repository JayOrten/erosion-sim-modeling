"""DLA datasets, and training pairs drawn from them.

A split is one folder, ``artifacts/data/<profile>/<split>/``:

- ``arrivals.npy``: int16, shape (runs, size, size), one arrival image per run (``simulate.py``).
- ``runs.npz``: the run table (``data/table.py``). Columns: ``index``, ``seed``, ``p``, and ``start_id`` (the
  shared start a test run continues, or -1 for a run from the seed particle).
- ``meta.json``: the config and package versions. Written last, so it marks a finished split.

A training pair is one run at two times: the cluster after ``t`` particles in, the cluster after ``t + x``
out, with ``p`` and ``x`` as conditioning.
"""

import json
import multiprocessing
import os
import time
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..config import Config
from ..data import table
from ..data.normalization import dla_conditioning
from ..paths import code_version, data_root
from .config import DataConfig
from .simulate import occupancy, run_seed, simulate

NO_START = -1


def _log_uniform(rng: np.random.Generator, low: float, high: float, n: int) -> np.ndarray:
    return np.exp(rng.uniform(np.log(low), np.log(high), n))


def plan_split(cfg: DataConfig, split: str) -> dict[str, np.ndarray]:
    """The run table of one split, before simulating: every run's ``index``, ``seed``, ``p``, ``start_id``."""
    n = cfg.runs(split)
    if split == "test":
        fresh = np.repeat(cfg.test_p, cfg.test_runs_per_p)
        p = np.concatenate([fresh, np.repeat(cfg.start_p, cfg.runs_per_start)])
        start_id = np.concatenate([np.full(len(fresh), NO_START),
                                   np.repeat(np.arange(len(cfg.start_p)), cfg.runs_per_start)])
    else:
        p = _log_uniform(np.random.Generator(np.random.PCG64(run_seed(cfg.profile, split, "p"))), *cfg.p_range, n)
        start_id = np.full(n, NO_START)
    return {
        "index": np.arange(n),
        "seed": np.array([run_seed(cfg.profile, split, i) for i in range(n)], dtype=np.int64),
        "p": p.astype(np.float64),
        "start_id": start_id.astype(np.int64),
    }


def shared_starts(cfg: DataConfig) -> np.ndarray:
    """The test split's shared start clusters: int16, shape (len(start_p), size, size)."""
    return np.stack([simulate(p, cfg.start_particles, run_seed(cfg.profile, "test", "start", s), cfg.size)
                     for s, p in enumerate(cfg.start_p)])


def _run_one(task):
    index, p, seed, particles, size, start = task
    return index, simulate(p, particles, seed, size, start)


def _simulate_all(tasks: list, processes: int):
    """Yield ``(index, arrival)`` for every task, in any order."""
    if processes == 1:
        yield from map(_run_one, tasks)
        return
    with multiprocessing.get_context("spawn").Pool(processes) as pool:
        yield from pool.imap_unordered(_run_one, tasks, chunksize=max(1, len(tasks) // (16 * processes)))


def make_split(cfg: DataConfig, split: str, out_dir: Path | None = None, progress_every: int = 0) -> Path:
    """Simulate every run of one split and write it to ``out_dir`` (default ``artifacts/data/<profile>/<split>``).

    Runs are a fixed function of the profile, the split, and the run index, so the result does not
    depend on the number of processes.
    """
    out_dir = Path(out_dir) if out_dir is not None else data_root(cfg.profile) / split
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "meta.json").unlink(missing_ok=True)
    columns = plan_split(cfg, split)
    starts = shared_starts(cfg) if split == "test" else None
    tasks = [(i, float(columns["p"][i]), int(columns["seed"][i]), cfg.particles, cfg.size,
              None if columns["start_id"][i] == NO_START else starts[columns["start_id"][i]])
             for i in range(len(columns["index"]))]
    arrivals = np.empty((len(tasks), cfg.size, cfg.size), dtype=np.int16)
    # The cores this process may use: on a Slurm node, the job's cores, not the whole node's.
    cores = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    processes = min(cfg.processes or cores or 1, len(tasks))
    start = time.perf_counter()
    for done, (i, arrival) in enumerate(_simulate_all(tasks, processes), start=1):
        arrivals[i] = arrival
        if progress_every and done % progress_every == 0:
            print(f"{cfg.profile}/{split}: {done}/{len(tasks)} runs, {time.perf_counter() - start:.0f} s", flush=True)
    np.save(out_dir / "arrivals.npy", arrivals)
    np.savez(out_dir / "runs.npz", **columns)
    meta = {"config": asdict(cfg), "split": split, "code": code_version(), "seconds": time.perf_counter() - start,
            "versions": {"numba": version("numba"), "numpy": np.__version__}}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return out_dir


class DLASplit:
    """One split of a DLA dataset: ``columns`` (the run table), ``arrivals`` (memory-mapped on first use),
    and ``config`` (the ``DataConfig`` fields it was made with)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.columns = table.load(self.path)
        self.meta = json.loads((self.path / "meta.json").read_text())
        self.config = self.meta["config"]
        self._arrivals = None

    @classmethod
    def open(cls, profile: str, split: str, root: Path | None = None) -> "DLASplit":
        return cls(data_root(profile, root) / split)

    @property
    def arrivals(self) -> np.ndarray:
        if self._arrivals is None:
            self._arrivals = np.load(self.path / "arrivals.npy", mmap_mode="r")
        return self._arrivals

    def __getstate__(self) -> dict:
        return self.__dict__ | {"_arrivals": None}  # each DataLoader worker opens its own memory map

    def __len__(self) -> int:
        return len(self.columns["index"])

    def select(self, **filters) -> np.ndarray:
        return table.select(self.columns, **filters)


class DLAPairs(Dataset):
    """Training pairs from the selected runs, ``repeats`` items per run, each with its own ``t`` and ``x``.

    - ``t``: uniform over ``t_range`` (both ends included). Default 1 to 90% of the particle count.
    - ``x``: log-uniform over 1 to ``particles - t``, as an integer.

    Each item's ``t`` and ``x`` are a fixed function of ``seed`` and the item index, so a fixed validation
    set is just a fixed list of indices. With many repeats, training rarely sees one pair twice.

    Each item is a dict: ``x`` (1, H, W) occupancy at ``t``; ``y`` (1, H, W) occupancy at ``t + x``;
    ``cond`` (2,) ``dla_conditioning(p, x)``; and ``run``, ``t``, ``steps`` (the jump ``x``) as integers.
    """

    def __init__(self, data: DLASplit, runs=None, repeats: int = 1, seed: int = 0,
                 t_range: tuple[int, int] | None = None):
        self.data = data
        self.runs = np.arange(len(data)) if runs is None else np.asarray(runs)
        self.repeats, self.seed = repeats, seed
        self.particles = data.config["particles"]
        self.p_range = tuple(data.config["p_range"])
        self.t_range = t_range or (1, int(0.9 * self.particles))
        if not 1 <= self.t_range[0] <= self.t_range[1] < self.particles:
            raise ValueError(f"t_range {self.t_range} must lie in 1 .. {self.particles - 1}")

    def __len__(self) -> int:
        return len(self.runs) * self.repeats

    def draw(self, index: int) -> tuple[int, int, int]:
        """(run, t, x) of item ``index``."""
        rng = np.random.Generator(np.random.PCG64(run_seed("pairs", self.seed, int(index))))
        t = int(rng.integers(self.t_range[0], self.t_range[1] + 1))
        most = self.particles - t
        x = min(int(np.exp(rng.uniform(0, np.log(most + 1)))), most)
        return int(self.runs[index // self.repeats]), t, max(x, 1)

    def __getitem__(self, index: int) -> dict:
        run, t, x = self.draw(index)
        arrival = np.asarray(self.data.arrivals[run])
        p = float(self.data.columns["p"][run])
        cond = dla_conditioning(p, x, self.p_range, self.particles)
        return {
            "x": torch.from_numpy(occupancy(arrival, t).astype(np.float32))[None],
            "y": torch.from_numpy(occupancy(arrival, t + x).astype(np.float32))[None],
            "cond": torch.from_numpy(cond.astype(np.float32)),
            "run": run,
            "t": t,
            "steps": x,
        }


def target_std(data: Dataset, count: int = 512, seed: int = 0) -> float:
    """Standard deviation of the target ``y`` over ``count`` random items of ``data``."""
    rng = np.random.default_rng(seed)
    ys = [data[int(i)]["y"].numpy() for i in rng.choice(len(data), min(count, len(data)), replace=False)]
    return float(np.std(np.stack(ys)))


def make_datasets(cfg: Config, data_root: Path | None = None) -> tuple[DLAPairs, DLAPairs, float]:
    """DLA training and validation pairs for ``cfg``, and ``sigma_data``, the standard deviation of the
    training target. There is no fitted scaling: the conditioning ranges are fixed by the dataset."""
    train_data, val_data = (DLASplit.open(cfg.data_profile, split, data_root) for split in ("train", "val"))
    train_runs, val_runs = (data.select(**dict(cfg.data)) for data in (train_data, val_data))
    train = DLAPairs(train_data, runs=train_runs, repeats=cfg.repeats, seed=cfg.seed, t_range=cfg.t_range)
    val = DLAPairs(val_data, runs=val_runs, seed=cfg.seed, t_range=cfg.t_range)
    return train, val, target_std(train, seed=cfg.seed)
