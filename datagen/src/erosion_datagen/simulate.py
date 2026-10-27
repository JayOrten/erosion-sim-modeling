"""Landlab erosion runs with richdem pit filling, and dataset generation."""

import os

# Landlab's components run slower on many threads at this grid size, so use one per process. This must
# be set before Landlab loads. See docs/simulator-timing.md.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import hashlib
import json
import multiprocessing
import time
import warnings
from pathlib import Path

import numpy as np

with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)  # richdem imports the deprecated pkg_resources
    import richdem
from landlab import RasterModelGrid
from landlab.components import FastscapeEroder, FlowAccumulator, LinearDiffuser
from landlab.utils.suppress_output import suppress_output

from .config import ARTIFACTS_DIR, Config
from .terrain import fbm

# Columns of the Latin hypercube, in order.
PARAMS = ("K_ref", "D", "m", "U", "period", "persistence")


def keyed_rng(cfg: Config, split: str, role: str, index: int | None = None) -> np.random.Generator:
    """Derive an independent, stable random stream from a canonical JSON record."""
    payload = json.dumps(["erosion-datagen-v1", cfg.profile, split, role, index], separators=(",", ":")).encode()
    key = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    return np.random.Generator(np.random.PCG64(key))


def latin_hypercube(rng: np.random.Generator, n: int, dims: int) -> np.ndarray:
    """``n`` points in [0, 1)^dims. Each dimension has exactly one point in each of its ``n`` equal slices."""
    slices = np.stack([rng.permutation(n) for _ in range(dims)], axis=1)
    return (slices + rng.random((n, dims))) / n


def _log_uniform(u: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.exp(np.log(low) + u * (np.log(high) - np.log(low)))


def sample_params(cfg: Config, split: str) -> dict[str, np.ndarray]:
    """Draw every run's parameters for one split with Latin Hypercube Sampling."""
    n = cfg.runs(split)
    u = latin_hypercube(keyed_rng(cfg, split, "params"), n, len(PARAMS))
    K_ref = _log_uniform(u[:, 0], *cfg.K_ref)
    m = cfg.m[0] + u[:, 2] * (cfg.m[1] - cfg.m[0])
    # The uplift column picks no uplift for its lowest U_zero_fraction, then spreads the rest over U_over_K.
    uplifted = u[:, 3] >= cfg.U_zero_fraction
    u_rest = (u[:, 3] - cfg.U_zero_fraction) / (1 - cfg.U_zero_fraction)
    return {
        "K": erodibility(cfg, K_ref, m),
        "K_ref": K_ref,
        "D": K_ref * _log_uniform(u[:, 1], *cfg.D_over_K),
        "m": m,
        "U": np.where(uplifted, K_ref * _log_uniform(np.clip(u_rest, 0, 1), *cfg.U_over_K), 0.0),
        "period": _log_uniform(u[:, 4], *cfg.period),
        "persistence": cfg.persistence[0] + u[:, 5] * (cfg.persistence[1] - cfg.persistence[0]),
        "terrain_seed": np.array([keyed_rng(cfg, split, "terrain", i).integers(2**32) for i in range(n)]),
    }


def erodibility(cfg: Config, K_ref, m):
    """The K the simulator uses: K_ref rescaled so a channel draining cfg.A_ref erodes at the same rate for any m."""
    return K_ref * cfg.A_ref ** (cfg.m_ref - m)


def starting_terrain(cfg: Config, terrain_seed: int, period: float, persistence: float) -> np.ndarray:
    return cfg.height * fbm(
        cfg.shape,
        seed=int(terrain_seed),
        period=period,
        octaves=cfg.octaves,
        persistence=persistence,
        lacunarity=cfg.lacunarity,
    )


def fill_pits(z: np.ndarray) -> np.ndarray:
    """Return a copy of the 2D surface ``z`` with every pit filled to its spill point.

    richdem adds a tiny slope across each filled area ("epsilon" filling), so every node has a downhill
    neighbor and water routed over the result always reaches an edge.
    """
    filled = richdem.rdarray(z.astype(np.float64), no_data=-9999)
    filled.geotransform = [0, 1, 0, 0, 0, -1]
    with suppress_output():  # richdem prints a progress bar on every call
        richdem.FillDepressions(filled, epsilon=True, in_place=True)
    return np.asarray(filled)


def drainage_area(z: np.ndarray, cell_size: float = 1.0) -> np.ndarray:
    """Drainage area (m²) of every node of the 2D surface ``z``, routed the same way as in ``simulate``."""
    grid = RasterModelGrid(z.shape, xy_spacing=cell_size)
    grid.add_field("topographic__elevation", z.ravel().astype(np.float64), at="node")
    grid.add_field("filled__elevation", fill_pits(z).ravel(), at="node")
    FlowAccumulator(grid, surface="filled__elevation", flow_director="D8").run_one_step()
    return grid.at_node["drainage_area"].reshape(z.shape)


def simulate(
    z0: np.ndarray,
    K: float,
    D: float,
    m: float,
    U: float,
    *,
    dt: float,
    steps: int,
    snapshot_every: int,
    cell_size: float = 1.0,
) -> np.ndarray:
    """Erode the 2D surface ``z0``. Return snapshots at steps 0, snapshot_every, 2 * snapshot_every, ...

    Each step: uplift interior nodes by ``U * dt``, route water over the pit-filled surface, cut channels
    (stream power law with n = 1), and diffuse hillslopes. Edges are open and keep their height.
    """
    if steps % snapshot_every:
        raise ValueError("steps must be a multiple of snapshot_every")
    grid = RasterModelGrid(z0.shape, xy_spacing=cell_size)
    z = grid.add_field("topographic__elevation", z0.ravel().astype(np.float64), at="node")
    filled = grid.add_field("filled__elevation", z.copy(), at="node")
    flow = FlowAccumulator(grid, surface="filled__elevation", flow_director="D8")
    incision = FastscapeEroder(grid, K_sp=K, m_sp=m, n_sp=1.0)
    diffusion = LinearDiffuser(grid, linear_diffusivity=D)

    snapshots = np.empty((steps // snapshot_every + 1, *z0.shape), dtype=np.float32)
    snapshots[0] = z0
    for step in range(1, steps + 1):
        z[grid.core_nodes] += U * dt
        filled[:] = fill_pits(z.reshape(z0.shape)).ravel()
        flow.run_one_step()
        incision.run_one_step(dt)
        diffusion.run_one_step(dt)
        if step % snapshot_every == 0:
            snapshots[step // snapshot_every] = z.reshape(z0.shape)
    return snapshots


def _run_one(task: tuple[Config, int, dict]) -> tuple[int, np.ndarray]:
    cfg, index, p = task
    z0 = starting_terrain(cfg, p["terrain_seed"], p["period"], p["persistence"])
    snapshots = simulate(
        z0, p["K"], p["D"], p["m"], p["U"],
        dt=cfg.dt, steps=cfg.steps, snapshot_every=cfg.snapshot_every, cell_size=cfg.cell_size,
    )
    return index, snapshots


def run_many(cfg: Config, params: dict[str, np.ndarray]):
    """Simulate one run per entry of ``params`` in parallel. Yield ``(index, snapshots)`` as runs finish.

    ``params`` holds equal-length arrays with the keys returned by ``sample_params``.
    """
    n = len(params["K"])
    tasks = [(cfg, i, {k: v[i] for k, v in params.items()}) for i in range(n)]
    processes = min(cfg.processes or os.cpu_count() or 1, n)
    # Spawned workers start fresh, so they see OMP_NUM_THREADS before Landlab loads.
    with multiprocessing.get_context("spawn").Pool(processes) as pool:
        yield from pool.imap_unordered(_run_one, tasks)


def dataset_dir(cfg: Config, split: str) -> Path:
    return ARTIFACTS_DIR / "data" / cfg.profile / split


def make_dataset(cfg: Config, split: str, out_dir: Path | None = None, progress_every: int = 0) -> Path:
    """Simulate every run of one split and write it to ``out_dir``. Return ``out_dir``.

    Files written:

    - ``snapshots.npy``: float32, shape (runs, snapshots, rows, cols). Written as a memory map, so
      readers can load it lazily with ``np.load(path, mmap_mode="r")``.
    - ``params.npz``: one array per parameter, length ``runs``, plus ``times`` (yr) for the snapshots.
    - ``meta.json``: the config and package versions.
    """
    out_dir = Path(out_dir) if out_dir is not None else dataset_dir(cfg, split)
    out_dir.mkdir(parents=True, exist_ok=True)
    params = sample_params(cfg, split)
    n = cfg.runs(split)
    snapshot_count = cfg.steps // cfg.snapshot_every + 1
    times = np.arange(snapshot_count) * cfg.snapshot_every * cfg.dt

    snapshots = np.lib.format.open_memmap(
        out_dir / "snapshots.npy", mode="w+", dtype=np.float32, shape=(n, snapshot_count, *cfg.shape)
    )
    start = time.perf_counter()
    for done, (index, result) in enumerate(run_many(cfg, params), start=1):
        snapshots[index] = result
        if progress_every and done % progress_every == 0:
            elapsed = time.perf_counter() - start
            print(f"{split}: {done}/{n} runs, {elapsed:.0f} s elapsed, ~{elapsed / done * (n - done):.0f} s left", flush=True)
    snapshots.flush()
    del snapshots

    np.savez(out_dir / "params.npz", times=times, **params)
    meta = {
        "config": {k: v for k, v in vars(cfg).items()},
        "split": split,
        "versions": {"landlab": _version("landlab"), "richdem": _version("richdem"), "numpy": np.__version__},
        "seconds": time.perf_counter() - start,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return out_dir


def load_dataset(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray], dict]:
    """Return (snapshots as a read-only memory map, params, meta) for a directory written by make_dataset."""
    path = Path(path)
    snapshots = np.load(path / "snapshots.npy", mmap_mode="r")
    with np.load(path / "params.npz") as f:
        params = dict(f)
    return snapshots, params, json.loads((path / "meta.json").read_text())


def _version(package: str) -> str:
    from importlib.metadata import version

    return version(package)
