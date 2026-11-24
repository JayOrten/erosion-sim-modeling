"""The world evaluation suite (track C), written as a results folder (``evaluation/results.py``).

Worlds. ``Settings.worlds`` names each world and the models that form its levels, coarsest first, for
example ``(("base", ("coarse", "fine")),)``. A third element overrides the sampler for that world:
``(("base_T1", ("coarse", "fine"), (("splits", ()),)),)`` (keys ``steps`` and ``splits``). The
``evaluate`` command fills it from a world's file in ``configs/worlds/``.

What is scored, on test runs, from the start terrain to each of ``ends_yr``:

- World benchmark: each world is run over the run's start terrain, erodibility map, and parameters, and
  compared with the run's snapshot on its interior, ``margin`` cells from the fixed edges (rounded up to
  the top level's cells). The run's own edges change its drainage everywhere (about 10 to 15% of the
  change, docs/notes/datagen-measurements.md), which the world cannot see, so expect a floor near that.
- Levels alone: each level on windows of the run with the true change of its parent as context, as in
  training (``level:<label>``). Separates a level's own error from the pipeline's.
- Baselines (``baselines``), on the same interior: ``analytical`` on the whole run (the bounded method on
  the true domain); ``analytical_chunks`` on closed ``chunk``² pieces, stitched; ``analytical_blended``
  on pieces at half-chunk stride, blended with tents; ``model_chunks``, the whole-run model
  ``chunk_model`` on closed pieces; ``simulator_chunks``, the simulator on closed pieces, read from the
  ``chunked_<chunk>.npy`` file that ``erosion-datagen chunked`` writes next to the split.
- Near chunk edges: each entry's RMSE within ``chunk_edge`` cells of an internal chunk boundary
  (``chunk_edge_rmse``), where the chunked baselines' false outlets are.
- Seams: ``world.checks.window_disagreement`` per level of each world, on the first run.
- Time per world query (``seconds_per_query``).

Results layout: as the erosion suite (``erosion/suite.py``): scalars ``entries`` and ``by`` (elapsed time),
plus ``seams``; arrays of per-pair metrics; examples of heights on the interior.
"""

import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from ..data.pairs import ErosionPairs, input_channels
from ..data.splits import Split
from ..erosion import metrics
from ..erosion.baselines import (
    analytical_stream_power,
    blended,
    chunk_edge_mask,
    chunked,
)
from ..evaluation.results import model_records, write_results
from ..objectives.edm import DEFAULT_STEPS, initial_noise
from ..objectives.sampling import generate, is_generative
from ..training.checkpoints import load_checkpoint
from .checks import window_disagreement
from .level import SPLIT, world_params
from .world import World

BASELINES = ("analytical", "analytical_chunks", "analytical_blended", "model_chunks", "simulator_chunks")


@dataclass(frozen=True)
class Settings:
    worlds: tuple[tuple, ...] = ()  # (world name, level labels coarsest first[, sampler overrides])
    data_profile: str | None = None  # test dataset; None uses the first level's training dataset
    split: str = "test"
    data: tuple[tuple[str, object], ...] = ()
    runs: int | None = 8
    ends_yr: tuple[float, ...] = (25_000.0, 50_000.0)
    margin: int = 32
    steps: int = DEFAULT_STEPS
    splits: tuple[float, ...] = (SPLIT,)
    level_pairs_per_run: int = 4
    baselines: tuple[str, ...] = ("analytical", "analytical_chunks", "analytical_blended")
    chunk: int = 256
    chunk_edge: int = 16
    chunk_model: str | None = None  # name of a whole-run model, for model_chunks
    seams: bool = True
    seam_windows: int = 4
    batch_size: int = 16
    examples: int = 2
    seed: int = 0
    bf16: bool = True


@torch.no_grad()
def _model_on_region(model, scaling, cfg, z0: np.ndarray, log_map: np.ndarray, params: dict, dT: float, key: tuple,
                     steps: int, bf16: bool) -> np.ndarray:
    """A whole-run model's end heights (m) for start heights ``z0`` treated as a closed run."""
    device = next(model.parameters()).device
    x = torch.as_tensor(input_channels(z0, 1, cfg.inputs, log_map)[None], dtype=torch.float32, device=device)
    cond = torch.as_tensor(scaling.parameters.scale(params, [0], dT), dtype=torch.float32, device=device)
    noise = initial_noise([key], z0.shape, model.fixed_edges) if is_generative(model) else None
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16 and device.type == "cuda"):
        y = generate(model, x, cond, noise, steps).float()[0, 0].cpu().double().numpy()
    return z0 + scaling.change.from_model(y, params["K_ref"][0], dT)


def _baselines(settings: Settings, z0: np.ndarray, end: int, end_yr: float, run: int, columns: dict,
               factors: np.ndarray, chunk_model: tuple | None, chunk_file: Path) -> dict[str, np.ndarray]:
    """End heights (m) over the whole run for each baseline in ``settings.baselines``."""
    K, m = columns["K"][run], columns["m"][run]

    def analytical(piece, origin):
        y, x = origin
        return analytical_stream_power(piece, K, m, end_yr, factors[y:y + piece.shape[0], x:x + piece.shape[1]])

    out = {}
    if "analytical" in settings.baselines:
        out["analytical"] = analytical(z0, (0, 0))
    if "analytical_chunks" in settings.baselines:
        out["analytical_chunks"] = chunked(analytical, z0, settings.chunk)
    if "analytical_blended" in settings.baselines:
        out["analytical_blended"] = blended(analytical, z0, settings.chunk)
    if "model_chunks" in settings.baselines:
        model, scaling, cfg = chunk_model
        params = world_params(K_ref=columns["K_ref"][run], D=columns["D"][run], m=m)
        log_map = np.log10(factors)

        def on_chunk(piece, origin):
            y, x = origin
            return _model_on_region(model, scaling, cfg, piece, log_map[y:y + piece.shape[0], x:x + piece.shape[1]],
                                    params, end_yr, (settings.seed, int(run), end, y, x), settings.steps, settings.bf16)

        out["model_chunks"] = chunked(on_chunk, z0, settings.chunk)
    if "simulator_chunks" in settings.baselines:
        out["simulator_chunks"] = np.asarray(np.load(chunk_file, mmap_mode="r")[run, end], dtype=np.float64)
    return out


@torch.no_grad()
def _levels_alone(label: str, level: tuple, test: Split, runs: np.ndarray, ends: list, settings: Settings) -> dict:
    """Per-pair metrics of one level on windows of the test runs, with its parent's true change as context."""
    model, scaling, cfg, _ = level
    device = next(model.parameters()).device
    dataset = ErosionPairs(test, scaling, cell=cfg.cell, window=cfg.window, context_cell=cfg.context_cell,
                           margin=cfg.margin, inputs=cfg.inputs, seed=settings.seed)
    values = []
    for i, r in enumerate(runs):
        for k in range(settings.level_pairs_per_run):
            end = ends[k % len(ends)][1]
            item = dataset.item(int(r), 0, end, index=i * settings.level_pairs_per_run + k)
            x, cond = item["x"][None].to(device), item["cond"][None].to(device)
            noise = initial_noise([(settings.seed, label, int(r), k)], x.shape[-2:], model.fixed_edges)
            y = generate(model, x, cond, noise if is_generative(model) else None, settings.steps)
            dT = test.times[end] - test.times[0]
            z_a, z_b = item["z_a"].double().numpy(), item["z_b"].double().numpy()
            z = z_a + scaling.change.from_model(y.float()[0, 0].cpu().double().numpy(), test.columns["K_ref"][r], dT)
            values.append(metrics.pair_metrics(z, metrics.Truth(z_a, z_b, test.columns["m"][r], cell=cfg.cell)))
    return {k: np.array([v[k] for v in values]) for k in metrics.PAIR_METRICS}


def evaluate_world(models: dict[str, Path], out: Path, settings: Settings, data_root: Path | None = None,
                   device: str | None = None, record: dict | None = None) -> Path:
    """Score the worlds and levels of ``settings`` (model label -> folder) and write the results to ``out``."""
    loaded = {label: load_checkpoint(folder, device=device) for label, folder in models.items()}
    if not settings.worlds:
        raise ValueError("settings.worlds names no world")
    worlds = [(w[0], tuple(w[1]), dict(w[2]) if len(w) > 2 else {}) for w in settings.worlds]
    level_labels = list(dict.fromkeys(label for _, labels, _ in worlds for label in labels))
    profile = settings.data_profile or loaded[level_labels[0]][2].data_profile
    test = Split.open(profile, settings.split, data_root)
    runs = test.select(**dict(settings.data))[: settings.runs]
    size = test.snapshots.shape[-1]
    for name, labels, _ in worlds:
        if loaded[labels[-1]][2].cell != 1:
            raise ValueError(f"world {name}'s finest level must have 1 m cells to be compared with the runs")
    top = max(loaded[labels[0]][2].cell for _, labels, _ in worlds)
    margin = int(np.ceil(settings.margin / top) * top)
    side = size - 2 * margin
    interior = (slice(margin, margin + side), slice(margin, margin + side))
    times = test.times - test.times[0]
    ends = [(e, int(np.flatnonzero(np.isclose(times, e))[0])) for e in settings.ends_yr if np.isclose(times, e).any()]
    if not ends:
        raise ValueError(f"none of ends_yr {settings.ends_yr} is a snapshot time")
    chunk_file = test.path / f"chunked_{settings.chunk}.npy"
    if "simulator_chunks" in settings.baselines and not chunk_file.exists():
        raise FileNotFoundError(f"{chunk_file} is missing: run `erosion-datagen chunked` for this split first")
    edge_mask = None
    if size % settings.chunk == 0:
        edge_mask = chunk_edge_mask((size, size), settings.chunk, settings.chunk_edge)[interior]
    chunk_model = loaded[settings.chunk_model][:3] if "model_chunks" in settings.baselines else None

    # The world benchmark and the baselines, on each run's interior.
    heights: dict[str, list[np.ndarray]] = {}
    truths, rows, seconds, seams = [], [], {}, {}
    c = test.columns
    for r in runs:
        terrain, erodibility = test.terrain(r), test.erodibility_map(r)
        z0 = np.asarray(test.snapshots[r, 0], dtype=np.float64)
        factors = erodibility.factors(0, 0, size, size)
        for end_yr, end in ends:
            z_b = np.asarray(test.snapshots[r, end], dtype=np.float64)
            truths.append(metrics.Truth(z0[interior], z_b[interior], c["m"][r]))
            rows.append((r, 0, end))
            heights.setdefault("no change", []).append(z0[interior])
            for name, labels, sampler in worlds:
                world = World(settings.seed, end_yr, [loaded[label][:3] for label in labels], K_ref=c["K_ref"][r],
                              D=c["D"][r], m=c["m"][r], terrain=terrain, erodibility=erodibility,
                              splits=tuple(sampler.get("splits", settings.splits)),
                              steps=sampler.get("steps", settings.steps), batch_size=settings.batch_size,
                              bf16=settings.bf16)
                begin = time.perf_counter()
                heights.setdefault(name, []).append(world.get(margin, margin, side, side))
                seconds.setdefault(name, []).append(time.perf_counter() - begin)
                if settings.seams and r == runs[0] and end == ends[0][1]:
                    seams[name] = {level.name: window_disagreement(level, margin // level.cfg.cell, margin // level.cfg.cell,
                                                                   settings.seam_windows)["phases"]
                                   for level in world.levels}
            for name, z in _baselines(settings, z0, end, end_yr, r, c, factors, chunk_model, chunk_file).items():
                heights.setdefault(name, []).append(z[interior])

    per_pair = {}
    for entry, zs in heights.items():
        values = [metrics.pair_metrics(z, t) for z, t in zip(zs, truths)]
        per_pair[entry] = {k: np.array([v[k] for v in values]) for k in metrics.PAIR_METRICS}
        if edge_mask is not None:
            per_pair[entry]["chunk_edge_sq_error"] = np.array([np.mean((z - t.z_b)[edge_mask] ** 2)
                                                               for z, t in zip(zs, truths)])
    for label in level_labels:
        per_pair[f"level:{label}"] = _levels_alone(f"level:{label}", loaded[label], test, runs, ends, settings)

    rows = np.array(rows, dtype=np.int64)
    dT_rows = test.times[rows[:, 2]] - test.times[rows[:, 1]]
    scalars = {"entries": {}, "by": {}, "seams": seams}
    for entry, pp in per_pair.items():
        summary = metrics.summarize(pp)
        if "chunk_edge_sq_error" in pp:
            summary["chunk_edge_rmse"] = float(np.sqrt(pp["chunk_edge_sq_error"].mean()))
        if entry in seconds:
            summary["seconds_per_query"] = float(np.mean(seconds[entry]))
        scalars["entries"][entry] = summary
    if len(ends) > 1:
        scalars["by"]["dT"] = {f"{e:g}": {entry: metrics.summarize(pp, dT_rows == e) for entry, pp in per_pair.items()
                                          if not entry.startswith("level:")} for e, _ in ends}

    arrays = {"pairs/run": rows[:, 0], "pairs/end": rows[:, 2], "pairs/dT": dT_rows}
    arrays |= {f"{entry}/{k}": v for entry, pp in per_pair.items() for k, v in pp.items()}
    n = min(settings.examples, len(truths))
    examples = {"z_a": np.stack([t.z_a for t in truths[:n]]), "z_b": np.stack([t.z_b for t in truths[:n]])}
    examples |= {entry: np.stack(zs[:n]) for entry, zs in heights.items()}
    kinds = {name: "world" for name, _, _ in worlds} | {f"level:{label}": "level" for label in level_labels}
    manifest = {
        "suite": "world",
        "models": model_records(models),
        "worlds": {name: {"levels": list(labels), "sampler": sampler} for name, labels, sampler in worlds},
        "entries": {entry: {"kind": kinds.get(entry, "baseline")} for entry in per_pair},
        "test": {"data_profile": profile, "split": settings.split, "runs": int(len(runs)), "pairs": int(len(truths)),
                 "margin": margin, "interior": side},
        "settings": asdict(settings),
    }
    return write_results(out, scalars, arrays, examples, {**manifest, **(record or {})})
