"""The DLA evaluation suite: every DLA metric for some models on the test split, written as a results
folder (``experiments/results.py``).

Test conditions, from the test split's run table (``dla/data.py``):

- ``p=<value>``: each fixed ``p`` of the test grid, grown from the seed particle (``t`` = 1) to the full
  particle count. One model sample per simulator run.
- ``start=<id>``: each shared start cluster, continued from ``start_particles`` to the full count. One
  model sample per simulator continuation.

For each model and condition, every metric in ``metrics.PER_CLUSTER`` is computed for the model's samples
and for the simulator's runs, and summarized with ``metrics.compare``.

Results layout:

Each scored (model, sampler steps) pair is an entry. Without a step sweep an entry is named by the model's
label; with one (``Settings.step_sweep``, experiment A4) it is ``<label>@<steps>``. The manifest's
``entries`` lists each entry's model, steps, and network calls per sample.

- scalars: ``[entry][condition][metric]`` -> ``compare`` dict, plus ``[entry]["seconds_per_sample"]``.
- arrays: ``<entry>/<condition>/<metric>`` per-sample values, ``simulator/<condition>/<metric>``, and
  ``.../radial_density`` curves with ``radial_edges``.
- examples: ``<entry>/<condition>`` and ``simulator/<condition>``: the first few clusters, uint8;
  ``start/<condition>``: the start cluster.
"""

import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..experiments.results import write_results
from ..objectives.edm import DEFAULT_STEPS
from ..objectives.sampling import network_calls
from ..training.checkpoints import load_checkpoint
from . import metrics
from .data import NO_START, DLASplit
from .sample import grow
from .simulate import occupancy


@dataclass(frozen=True)
class Settings:
    steps: int = DEFAULT_STEPS  # sampler steps
    batch_size: int = 64
    seed: int = 0  # noise keys are (seed, condition, sample index), the same for every model
    examples: int = 4  # clusters kept per condition for figures
    bf16: bool = True
    only: str | None = None  # score only conditions whose name starts with this, e.g. "p=" for A5
    step_sweep: tuple[int, ...] | None = None  # score every model at each of these step counts (A4)


@dataclass(frozen=True)
class Condition:
    name: str
    runs: np.ndarray  # test runs it compares against
    t: int  # particles in the start cluster
    x: int  # particles added


def conditions(test: DLASplit) -> list[Condition]:
    """Every test condition of the split, in table order: the fixed-``p`` grid, then the shared starts."""
    cfg, cols = test.config, test.columns
    particles, out = cfg["particles"], []
    fresh = cols["start_id"] == NO_START
    for p in cfg["test_p"]:
        out.append(Condition(f"p={p:g}", np.flatnonzero(fresh & (cols["p"] == p)), 1, particles - 1))
    for s in range(len(cfg["start_p"])):
        runs = np.flatnonzero(cols["start_id"] == s)
        out.append(Condition(f"start={s}", runs, cfg["start_particles"], particles - cfg["start_particles"]))
    return out


def evaluate_dla(models: dict[str, Path], out: Path, data_profile: str | None = None, data_root: Path | None = None,
                 settings: Settings = Settings(), device: str | None = None) -> Path:
    """Score each model (label -> model folder) on every test condition and write the results to ``out``.

    The test split is ``data_profile``'s (default: the first model's training dataset). ``data_root``
    overrides where it is read from, as in training.
    """
    loaded = {label: load_checkpoint(folder, device=device) for label, folder in models.items()}
    first_cfg = next(iter(loaded.values()))[2]
    test = DLASplit.open(data_profile or first_cfg.data_profile, "test", data_root)
    particles, p_range = test.config["particles"], tuple(test.config["p_range"])
    edges = np.arange(0, test.config["size"] // 2 + 1, 2.0)

    conds = [c for c in conditions(test) if settings.only is None or c.name.startswith(settings.only)]
    scalars, arrays, examples = {}, {"radial_edges": edges}, {}
    simulated = {}
    for cond in conds:
        final = occupancy(np.asarray(test.arrivals[cond.runs]), particles)
        start = occupancy(np.asarray(test.arrivals[cond.runs]), cond.t)
        simulated[cond.name] = (final, start)
        examples[f"simulator/{cond.name}"] = final[: settings.examples]
        examples[f"start/{cond.name}"] = start[0]
        arrays[f"simulator/{cond.name}/radial_density"] = metrics.radial_density(final, edges)
        for name, metric in metrics.PER_CLUSTER.items():
            arrays[f"simulator/{cond.name}/{name}"] = metric(final, start)

    entries = {}
    for label, (model, _, cfg, _) in loaded.items():
        if cfg.task != "dla":
            raise ValueError(f"{label} is a {cfg.task} model, not DLA")
        for steps in settings.step_sweep or (settings.steps,):
            entry = f"{label}@{steps}" if settings.step_sweep else label
            entries[entry] = {"model": label, "steps": steps, "network_calls": network_calls(model, steps)}
            scalars[entry] = {}
            seconds, count = 0.0, 0
            for cond in conds:
                final, start = simulated[cond.name]
                p = test.columns["p"][cond.runs]
                keys = [(settings.seed, cond.name, i) for i in range(len(cond.runs))]
                begin = time.perf_counter()
                samples = grow(model, start, p, cond.x, keys, p_range, particles, steps, settings.batch_size,
                               settings.bf16)
                seconds += time.perf_counter() - begin
                count += len(samples)
                examples[f"{entry}/{cond.name}"] = samples[: settings.examples]
                arrays[f"{entry}/{cond.name}/radial_density"] = metrics.radial_density(samples, edges)
                scalars[entry][cond.name] = {}
                for name, metric in metrics.PER_CLUSTER.items():
                    values = metric(samples, start)
                    arrays[f"{entry}/{cond.name}/{name}"] = values
                    scalars[entry][cond.name][name] = metrics.compare(values, arrays[f"simulator/{cond.name}/{name}"])
            scalars[entry]["seconds_per_sample"] = seconds / max(count, 1)

    manifest = {
        "suite": "dla",
        "models": {label: {"folder": f"{Path(folder).parent.name}/{Path(folder).name}"}
                   for label, folder in models.items()},
        "test": {"data_profile": test.path.parent.name, "split": "test", "runs": len(test)},
        "entries": entries,
        "conditions": [c.name for c in conds],
        "settings": asdict(settings),
    }
    return write_results(out, scalars, arrays, examples, manifest)


@dataclass(frozen=True)
class AgreementSettings:
    steps: int = DEFAULT_STEPS
    batch_size: int = 64
    seed: int = 0
    # Containment: jumps as fractions of the particles left after the start (600 - 200 = 400 for
    # dla_master, so 25, 50, 100, 200, 400).
    jump_fractions: tuple[float, ...] = (1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0)
    fresh_starts: int = 100  # containment also from this many test clusters cut at start_particles
    chains: tuple[int, ...] = (1, 2, 4)  # reach the full count in this many equal jumps
    examples: int = 4
    bf16: bool = True


def missing_fraction(small: np.ndarray, large: np.ndarray) -> np.ndarray:
    """Per pair, the fraction of ``small``'s cells absent from ``large``. 0 when ``small`` is contained."""
    small, large = small.astype(bool), large.astype(bool)
    missing = (small & ~large).reshape(len(small), -1).sum(axis=1)
    return missing / np.maximum(small.reshape(len(small), -1).sum(axis=1), 1)


def evaluate_agreement(models: dict[str, Path], out: Path, data_profile: str | None = None,
                       data_root: Path | None = None, settings: AgreementSettings = AgreementSettings(),
                       device: str | None = None) -> Path:
    """Agreement across steps (experiment A6), written as a results folder.

    - Containment: from each start, samples at every jump size share one noise key. For every pair of
      jumps ``x1 < x2``, the fraction of the ``x1`` sample's cells missing from the ``x2`` sample. Starts:
      the shared starts (every continuation's start), and ``fresh_starts`` test clusters from the seed
      particle cut at ``start_particles``.
    - Chaining: from each shared start, reach the full count in 1, 2, or 4 equal jumps, one sample per
      simulator continuation. Every per-cluster metric of each scheme is compared with the simulator's
      continuations.

    Scalars: ``[model]["containment"][group]["<x1><<x2>"]`` (mean missing fraction) and
    ``[model]["chain"]["<n>x<jump>"][metric]`` (a ``metrics.compare`` dict, pooled over shared starts).
    """
    loaded = {label: load_checkpoint(folder, device=device) for label, folder in models.items()}
    test = DLASplit.open(data_profile or next(iter(loaded.values()))[2].data_profile, "test", data_root)
    cfg, cols = test.config, test.columns
    particles, p_range, t0 = cfg["particles"], tuple(cfg["p_range"]), cfg["start_particles"]
    room = particles - t0
    jumps = sorted({max(1, round(f * room)) for f in settings.jump_fractions})
    shared = np.flatnonzero(cols["start_id"] != NO_START)
    from_seed = np.flatnonzero(cols["start_id"] == NO_START)
    fresh = from_seed[np.unique(np.linspace(0, len(from_seed) - 1, settings.fresh_starts).astype(int))]
    groups = {"shared": shared, "fresh": fresh}
    simulator = occupancy(np.asarray(test.arrivals[shared]), particles)
    shared_start = occupancy(np.asarray(test.arrivals[shared]), t0)

    scalars, arrays, examples = {}, {}, {"start": shared_start[: settings.examples],
                                         "simulator": simulator[: settings.examples]}
    for name, metric in metrics.PER_CLUSTER.items():
        arrays[f"simulator/chain/{name}"] = metric(simulator, shared_start)
    for label, (model, _, mcfg, _) in loaded.items():
        if mcfg.task != "dla":
            raise ValueError(f"{label} is a {mcfg.task} model, not DLA")
        run = dict(p_range=p_range, particles=particles, steps=settings.steps, batch_size=settings.batch_size,
                   bf16=settings.bf16)
        scalars[label] = {"containment": {}, "chain": {}}
        for group, runs in groups.items():
            start = occupancy(np.asarray(test.arrivals[runs]), t0)
            p = cols["p"][runs]
            keys = [(settings.seed, "containment", group, i) for i in range(len(runs))]
            grown = {x: grow(model, start, p, x, keys, **run) for x in jumps}
            rows = {}
            for i, x1 in enumerate(jumps):
                for x2 in jumps[i + 1:]:
                    values = missing_fraction(grown[x1], grown[x2])
                    arrays[f"{label}/containment/{group}/{x1}<{x2}"] = values
                    rows[f"{x1}<{x2}"] = float(values.mean())
            scalars[label]["containment"][group] = rows
            examples[f"{label}/containment/{group}"] = np.stack([grown[x][0] for x in jumps])
        p = cols["p"][shared]
        for n in settings.chains:
            jump = room // n
            if jump * n != room:
                raise ValueError(f"{room} particles do not split into {n} equal jumps")
            cluster = shared_start
            for k in range(n):
                keys = [(settings.seed, "chain", n, k, i) for i in range(len(shared))]
                cluster = grow(model, cluster, p, jump, keys, **run)
            scheme = f"{n}x{jump}"
            examples[f"{label}/chain/{scheme}"] = cluster[: settings.examples]
            scalars[label]["chain"][scheme] = {}
            for name, metric in metrics.PER_CLUSTER.items():
                values = metric(cluster, shared_start)
                arrays[f"{label}/chain/{scheme}/{name}"] = values
                scalars[label]["chain"][scheme][name] = metrics.compare(values, arrays[f"simulator/chain/{name}"])

    manifest = {
        "suite": "dla_agreement",
        "models": {label: {"folder": f"{Path(folder).parent.name}/{Path(folder).name}"}
                   for label, folder in models.items()},
        "test": {"data_profile": test.path.parent.name, "split": "test", "runs": len(test)},
        "entries": {label: {"model": label, "steps": settings.steps,
                            "network_calls": network_calls(loaded[label][0], settings.steps)} for label in models},
        "jumps": jumps,
        "settings": asdict(settings),
    }
    return write_results(out, scalars, arrays, examples, manifest)
