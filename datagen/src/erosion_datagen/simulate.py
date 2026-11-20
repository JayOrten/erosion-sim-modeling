"""Landlab erosion runs with richdem pit filling, and dataset generation."""

import os

# Landlab's components run slower on many threads at this grid size, so use one per process. This must
# be set before Landlab loads. See docs/notes/simulator-timing.md.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import hashlib
import json
import multiprocessing
import time
import warnings
from pathlib import Path
from typing import Callable

import numpy as np

with warnings.catch_warnings():
    warnings.simplefilter(
        "ignore", UserWarning
    )  # richdem imports the deprecated pkg_resources
    import richdem
from erosion_terrain import SEED_LIMIT, TERRAIN_FIELDS, Terrain, map_from_columns
from landlab import RasterModelGrid
from landlab.components import FastscapeEroder, FlowAccumulator, LinearDiffuser
from landlab.utils.suppress_output import suppress_output

from .config import ARTIFACTS_DIR, OOD_RANGES, Group


def code_version() -> dict:
    """The repo's commit, and whether ``src/``, ``datagen/``, or ``terrain/`` had uncommitted changes, as
    ``erosion_sim.paths.code_version`` records them (this environment cannot import it). Recorded in each
    split's ``meta.json``; never checked."""
    import subprocess

    repo = Path(__file__).resolve().parents[3]

    def git(*args) -> str:
        return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True,
                              check=True).stdout.strip()

    try:
        changes = git("status", "--porcelain", "--", "src", "datagen", "terrain")
        return {"commit": git("rev-parse", "HEAD"), "uncommitted_changes": bool(changes)}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "uncommitted_changes": None}


def run_seed(group: Group, split: str, index: int) -> int:
    """A run's 63-bit seed: SHA-256 of (dataset, group, split, index) as canonical JSON. It seeds the
    run's PCG64 stream, from which every column of the run is drawn."""
    payload = json.dumps(["erosion-datagen-v2", group.dataset, group.name, split, int(index)],
                         separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") >> 1


def _log_uniform(rng: np.random.Generator, low_high: tuple[float, float]) -> float:
    low, high = low_high
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def draw_run(group: Group, split: str, index: int) -> dict:
    """Every column of one run, drawn from its own stream (``run_seed``). Each column is independent.

    Every draw happens in a fixed order whether or not the run uses it (a warp strength for an unwarped
    run, say), so one setting never shifts another's draws.
    """
    seed = run_seed(group, split, index)
    rng = np.random.Generator(np.random.PCG64(seed))
    noise = str(rng.choice(group.noise_types))
    fractal = str(rng.choice(group.fractal_types))
    warped = rng.uniform() < group.warp_fraction
    warp = float(rng.uniform(*group.warp))
    row = {
        "seed": int(rng.integers(SEED_LIMIT)),  # the noise seed (erosion_terrain.Terrain.seed)
        "period": _log_uniform(rng, group.period),
        "persistence": float(rng.uniform(*group.persistence)),
        "octaves": int(rng.integers(group.octaves[0], group.octaves[1] + 1)),
        "lacunarity": float(rng.uniform(*group.lacunarity)),
        "height": _log_uniform(rng, group.height),
        "noise": noise,
        "fractal": fractal,
        "warp": warp if warped else 0.0,
    }
    ranges = {"K_ref": group.K_ref, "D_over_K": group.D_over_K}
    m_range, ood_param = group.m, ""
    if group.ood:
        ood_param, out_of_range = OOD_RANGES[index % len(OOD_RANGES)]
        if ood_param == "m":
            m_range = out_of_range
        else:
            ranges[ood_param] = out_of_range
    K_ref = _log_uniform(rng, ranges["K_ref"])
    D_over_K = _log_uniform(rng, ranges["D_over_K"])
    m = float(rng.uniform(*m_range))
    shapes, shares = zip(*group.maps)
    shape = shapes[int(np.searchsorted(np.cumsum(shares), rng.uniform() * sum(shares), side="right"))]
    contrast = _log_uniform(rng, group.map_contrast)
    map_period = _log_uniform(rng, group.map_period)
    band_width = _log_uniform(rng, group.band_width)
    band_angle = float(rng.uniform(0.0, 180.0))
    map_seed = int(rng.integers(SEED_LIMIT))
    return row | {
        "K_ref": K_ref,
        "K": erodibility(group, K_ref, m),
        "D": K_ref * D_over_K,
        "m": m,
        "U": 0.0,
        "ood_param": ood_param,
        "map_shape": shape,
        "map_contrast": contrast if shape != "uniform" else 1.0,
        "map_period": map_period,
        "band_width": band_width,
        "band_angle": band_angle,
        "map_seed": map_seed,
    }


def plan_split(group: Group, split: str) -> dict[str, np.ndarray]:
    """The run table of one (group, split) before simulating: one array per column, one row per run.

    Columns: ``group``, ``split``, ``index``, ``run_seed``, and every column of ``draw_run``.
    """
    n = group.runs(split)
    rows = [draw_run(group, split, i) for i in range(n)]
    columns = {name: np.array([row[name] for row in rows]) for name in rows[0]} if rows else {}
    return {"group": np.full(n, group.name), "split": np.full(n, split), "index": np.arange(n),
            "run_seed": np.array([run_seed(group, split, i) for i in range(n)], dtype=np.int64)} | columns


def erodibility(group: Group, K_ref, m):
    """The K the simulator uses: K_ref rescaled so a channel draining A_ref erodes at the same rate for any m."""
    return K_ref * group.A_ref ** (group.m_ref - m)


def run_terrain(row: dict) -> Terrain:
    """The ``Terrain`` of one run, from its run-table row (a dict of scalars)."""
    return Terrain(**{name: np.asarray(row[name]).item() for name in TERRAIN_FIELDS})


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
    FlowAccumulator(
        grid, surface="filled__elevation", flow_director="D8"
    ).run_one_step()
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
    K_map: np.ndarray | None = None,
) -> np.ndarray:
    """Erode the 2D surface ``z0``. Return snapshots at steps 0, snapshot_every, 2 * snapshot_every, ...

    Each step: uplift interior nodes by ``U * dt``, route water over the pit-filled surface, cut channels
    (stream power law with n = 1, erodibility ``K * K_map`` per cell), and diffuse hillslopes. Edges are
    open and keep their height. ``K_map`` (same shape as ``z0``) defaults to 1 everywhere.
    """
    if steps % snapshot_every:
        raise ValueError("steps must be a multiple of snapshot_every")
    grid = RasterModelGrid(z0.shape, xy_spacing=cell_size)
    z = grid.add_field(
        "topographic__elevation", z0.ravel().astype(np.float64), at="node"
    )
    filled = grid.add_field("filled__elevation", z.copy(), at="node")
    flow = FlowAccumulator(grid, surface="filled__elevation", flow_director="D8")
    K_sp = K if K_map is None else K * np.asarray(K_map, dtype=np.float64).ravel()
    incision = FastscapeEroder(grid, K_sp=K_sp, m_sp=m, n_sp=1.0)
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


def _run_one(task: tuple[Group, int, dict]) -> tuple[int, np.ndarray]:
    group, index, row = task
    z0 = run_terrain(row).heights(0, 0, group.size, group.size)
    K_map = map_from_columns(row).factors(0, 0, group.size, group.size)
    snapshots = simulate(z0, row["K"], row["D"], row["m"], row["U"], dt=group.dt, steps=group.steps,
                         snapshot_every=group.snapshot_every, cell_size=group.cell_size, K_map=K_map)
    return index, snapshots


def run_many(group: Group, columns: dict[str, np.ndarray], indices=None):
    """Simulate one run per row of ``columns`` (a run table from ``plan_split``) in parallel. Yield
    ``(index, snapshots)`` as runs finish. ``indices`` limits the runs to those; None runs all."""
    indices = range(len(columns["index"])) if indices is None else list(indices)
    tasks = [(group, i, {k: v[i] for k, v in columns.items()}) for i in indices]
    if not tasks:
        return
    processes = min(group.processes or os.cpu_count() or 1, len(tasks))
    # Spawned workers start fresh, so they see OMP_NUM_THREADS before Landlab loads.
    with multiprocessing.get_context("spawn").Pool(processes) as pool:
        yield from pool.imap_unordered(_run_one, tasks)


def dataset_dir(group: Group, split: str) -> Path:
    """``artifacts/data/<dataset>/<group>/<split>``."""
    return ARTIFACTS_DIR / "data" / group.dataset / group.name / split


def make_dataset(
    group: Group,
    split: str,
    out_dir: Path | None = None,
    progress_every: int = 0,
    progress: Callable[[int, int], None] | None = None,
) -> Path:
    """Simulate every run of one (group, split) and write it to ``out_dir`` (default ``dataset_dir``).
    Return ``out_dir``.

    Files written:

    - ``snapshots.npy``: float32, shape (runs, snapshots, rows, cols). Written as a memory map, so
      readers can load it lazily with ``np.load(path, mmap_mode="r")``.
    - ``runs.npz``: the run table (``plan_split``). One array of length ``runs`` per column: ``group``,
      ``split``, ``index``, ``run_seed``; each ``Terrain`` field (``seed`` is the noise seed); the erosion
      parameters ``K_ref``, ``K``, ``D``, ``m``, ``U`` (0), and ``ood_param`` ("" unless an ood group);
      and the erodibility map columns ``map_shape``, ``map_contrast``, ``map_period``, ``band_width``,
      ``band_angle``, ``map_seed`` (``erosion_terrain.map_from_columns``).
    - ``meta.json``: the group settings, the snapshot ``times`` (yr), and package versions. Written
      last, so it marks a finished split.

    Resumable: while it runs, ``progress.json`` holds the settings and the runs already on disk (each
    added only after its snapshots are flushed). Called again after an interruption, with the same
    settings, it simulates only the missing runs. Runs are a fixed function of the settings and the run
    index, so the result is the same as an uninterrupted run. With different settings, it starts over.

    ``progress(done, total)`` is called after each run is saved, and ``progress_every`` > 0 prints a line
    every that many runs.
    """
    out_dir = Path(out_dir) if out_dir is not None else dataset_dir(group, split)
    out_dir.mkdir(parents=True, exist_ok=True)
    columns = plan_split(group, split)
    n = group.runs(split)
    times = np.arange(group.snapshots) * group.snapshot_every * group.dt
    shape = (n, group.snapshots, group.size, group.size)

    path, record = out_dir / "snapshots.npy", out_dir / "progress.json"
    settings = json.loads(
        json.dumps({"config": vars(group), "split": split, "shape": shape})
    )
    done: list[int] = []
    if path.exists() and record.exists():
        saved = json.loads(record.read_text())
        if saved["settings"] == settings:
            done = saved["done"]
        else:
            print(
                f"{split}: settings differ from the interrupted run; starting over",
                flush=True,
            )
    if done:
        snapshots = np.lib.format.open_memmap(path, mode="r+")
        print(f"{split}: resuming, {len(done)}/{n} runs already on disk", flush=True)
    else:
        (out_dir / "meta.json").unlink(missing_ok=True)
        snapshots = np.lib.format.open_memmap(
            path, mode="w+", dtype=np.float32, shape=shape
        )

    def save():
        tmp = record.with_suffix(".tmp")
        tmp.write_text(json.dumps({"settings": settings, "done": done}))
        tmp.replace(
            record
        )  # atomic, so an interruption never leaves a half-written record

    save()
    finished = set(done)
    start, first = time.perf_counter(), len(done)
    if progress:
        progress(len(done), n)
    for index, result in run_many(
        group, columns, [i for i in range(n) if i not in finished]
    ):
        snapshots[index] = result
        snapshots.flush()
        done.append(int(index))
        save()
        if progress:
            progress(len(done), n)
        new = len(done) - first
        if progress_every and new % progress_every == 0:
            elapsed = time.perf_counter() - start
            print(
                f"{split}: {len(done)}/{n} runs, {elapsed:.0f} s elapsed, ~{elapsed / new * (n - len(done)):.0f} s left",
                flush=True,
            )
    snapshots.flush()
    del snapshots

    np.savez(out_dir / "runs.npz", **columns)
    meta = {
        "config": {k: v for k, v in vars(group).items()},
        "split": split,
        "code": code_version(),
        "times": times.tolist(),
        "versions": {
            "landlab": _version("landlab"),
            "richdem": _version("richdem"),
            "numpy": np.__version__,
        },
        "seconds": time.perf_counter() - start,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    record.unlink()
    return out_dir


def _run_chunk(task: tuple[Group, int, dict, int, int, int]) -> tuple[int, int, int, np.ndarray]:
    group, index, row, y0, x0, chunk = task
    z0 = run_terrain(row).heights(y0, x0, chunk, chunk)
    K_map = map_from_columns(row).factors(y0, x0, chunk, chunk)
    snapshots = simulate(z0, row["K"], row["D"], row["m"], row["U"], dt=group.dt, steps=group.steps,
                         snapshot_every=group.snapshot_every, cell_size=group.cell_size, K_map=K_map)
    return index, y0, x0, snapshots


def make_chunked(group: Group, split: str, chunk: int, out_dir: Path | None = None, progress_every: int = 0) -> Path:
    """Simulate every run of an existing split again as separate closed ``chunk``² pieces, stitched: the
    "independent chunks" baseline of experiment C8. Each chunk's fixed edges are outlets the run does not
    have. Writes ``chunked_<chunk>.npy`` next to the split's ``snapshots.npy``, with the same shape."""
    out_dir = Path(out_dir) if out_dir is not None else dataset_dir(group, split)
    if group.size % chunk:
        raise ValueError(f"{group.size}² runs do not split into {chunk}² chunks")
    with np.load(out_dir / "runs.npz") as f:
        columns = dict(f)
    n = len(columns["index"])
    path = out_dir / f"chunked_{chunk}.npy"
    tmp = path.with_suffix(".partial.npy")
    out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=(n, group.snapshots, group.size, group.size))
    tasks = [(group, i, {k: v[i] for k, v in columns.items()}, y, x, chunk)
             for i in range(n) for y in range(0, group.size, chunk) for x in range(0, group.size, chunk)]
    processes = min(group.processes or os.cpu_count() or 1, len(tasks))
    with multiprocessing.get_context("spawn").Pool(processes) as pool:
        for done, (i, y, x, snaps) in enumerate(pool.imap_unordered(_run_chunk, tasks), start=1):
            out[i, :, y:y + chunk, x:x + chunk] = snaps
            if progress_every and done % progress_every == 0:
                print(f"{group.name}/{split} chunks: {done}/{len(tasks)}", flush=True)
    out.flush()
    del out
    tmp.replace(path)
    return path


def load_dataset(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray], dict]:
    """Return (snapshots as a read-only memory map, run table columns, meta) for a directory written by
    make_dataset."""
    path = Path(path)
    snapshots = np.load(path / "snapshots.npy", mmap_mode="r")
    with np.load(path / "runs.npz") as f:
        columns = dict(f)
    return snapshots, columns, json.loads((path / "meta.json").read_text())


def _version(package: str) -> str:
    from importlib.metadata import version

    return version(package)
