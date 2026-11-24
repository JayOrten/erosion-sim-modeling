"""The erosion evaluation suite (track B): every terrain metric for single-window models and baselines on
test runs, written as a results folder (``evaluation/results.py``).

Pairs. Everything is scored on the same (run, start, end) pairs of the test runs: ``pairs_per_run``
random pairs per run (fixed by ``seed``), or with ``ends_yr``, the pairs from the start terrain to each of
those elapsed times. Whole runs, every cell scored.

Entries. Each scored method is an entry:

- each baseline (``baselines``: name -> a baseline predictor file's contents), by its name. Methods:
  ``no_change`` (the start terrain) and ``analytical`` (``baselines.py``).
- each model, at ``steps`` sampler steps, or at each of ``step_sweep`` as ``<label>@<steps>``. A generative
  model draws ``samples`` samples per pair; sample 0 is scored, and the per-cell spread across samples
  is compared with the error.
- with ``fill_pits``, each model's output after pit filling, as ``<label> + pit filling`` (experiment B7).
- with ``chains``, each model called ``n`` times in a row for ``n × dT`` years, as
  ``<label> chained <n>x<dT>``, on the pairs from the start terrain to that time (experiment B10).

Results layout:

- scalars: ``entries`` (entry -> ``metrics.summarize`` over all pairs, plus ``fine_detail`` and, for
  sampled models, ``spread_correlation``); ``by`` (factor -> group -> entry -> summary), for every factor in
  ``BREAKDOWNS`` with more than one group among the pairs; ``seconds_per_pair`` per model entry.
- arrays: ``pairs/<column>`` (run, start, end, dT, and each factor's value per pair);
  ``<entry>/<metric>`` per-pair metrics; ``spectrum/<entry>``, ``spectrum/simulator``, and
  ``spectrum/wavelength``: mean change spectra.
- examples: ``z_a``, ``z_b``, and ``<entry>`` heights for the first few pairs.
"""

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from ..data.hydrology import fill_pits
from ..data.pairs import ErosionPairs
from ..data.splits import Split
from ..evaluation.results import model_records, write_results
from ..objectives.edm import DEFAULT_STEPS, initial_noise
from ..objectives.sampling import generate, is_generative, network_calls
from ..training.checkpoints import load_checkpoint
from . import metrics
from .baselines import analytical_stream_power

# Factors results are broken down by. Categorical ones group by value; continuous ones by 4 bins evenly
# spaced on their sampling scale; elapsed time by every value.
CATEGORICAL = ("noise_setup", "fractal", "map_shape", "ood_param")
CONTINUOUS = {"K_ref": "log", "D_over_K": "log", "m": "linear", "map_contrast": "log"}
BREAKDOWNS = (*CATEGORICAL, *CONTINUOUS, "dT")


@dataclass(frozen=True)
class Settings:
    data_profile: str | None = None  # test dataset; None uses the first model's training dataset
    split: str = "test"
    data: dict = field(default_factory=dict)  # filters on the test runs (table.select)
    runs: int | None = None  # use at most this many runs, the first in table order
    pairs_per_run: int = 4
    ends_yr: tuple[float, ...] | None = None  # pairs from the start to each of these times, instead of random
    samples: int = 2  # samples per pair for generative models (spread needs 2 or more)
    steps: int = DEFAULT_STEPS
    step_sweep: tuple[int, ...] | None = None
    fill_pits: bool = False
    chains: tuple[tuple[int, float], ...] = ()  # (calls, years per call)
    batch_size: int = 32
    examples: int = 3
    seed: int = 0
    bf16: bool = True


def _factor_values(test: Split, run: np.ndarray, dT: np.ndarray) -> dict[str, np.ndarray]:
    """Each breakdown factor's value per pair, where the test runs have that column."""
    c = test.columns
    out = {"dT": dT, "K_ref": c["K_ref"][run], "D_over_K": (c["D"] / c["K_ref"])[run], "m": c["m"][run]}
    if "noise" in c:
        warp = np.where(c["warp"][run] > 0, "warp", "no warp")
        out["noise_setup"] = np.array([f"{n}/{f}/{w}" for n, f, w in zip(c["noise"][run], c["fractal"][run], warp)])
        out["fractal"] = c["fractal"][run]
    for name in ("map_shape", "map_contrast", "ood_param"):
        if name in c:
            out[name] = c[name][run]
    return out


def _groups(values: np.ndarray, scale: str | None) -> dict[str, np.ndarray]:
    """Group label -> row mask. Categorical (``scale`` None) by value; continuous by 4 bins."""
    if scale is None:
        return {str(v): values == v for v in np.unique(values)}
    v = np.log(values) if scale == "log" else values.astype(np.float64)
    if np.ptp(v) == 0:
        return {f"{values[0]:.3g}": np.ones(len(values), dtype=bool)}
    edges = np.linspace(v.min(), v.max(), 5)
    which = np.clip(np.searchsorted(edges, v, side="right") - 1, 0, 3)
    back = np.exp(edges) if scale == "log" else edges
    return {f"{back[b]:.3g}-{back[b + 1]:.3g}": which == b for b in range(4) if (which == b).any()}


def choose_pairs(test: Split, runs: np.ndarray, settings: Settings) -> np.ndarray:
    """(run, start, end) rows for every scored pair."""
    times, rows = test.times, []
    for run in runs:
        if settings.ends_yr is not None:
            for end in settings.ends_yr:
                matches = np.flatnonzero(np.isclose(times - times[0], end))
                if len(matches):
                    rows.append((run, 0, int(matches[0])))
            continue
        rng = np.random.default_rng([settings.seed, int(run)])
        all_pairs = [(a, b) for a in range(len(times)) for b in range(a + 1, len(times))]
        for k in rng.choice(len(all_pairs), min(settings.pairs_per_run, len(all_pairs)), replace=False):
            rows.append((run, *all_pairs[k]))
    return np.array(rows, dtype=np.int64).reshape(-1, 3)


@torch.no_grad()
def _predict(model, dataset: ErosionPairs, pairs: np.ndarray, settings: Settings, steps: int, samples: int,
             starts: np.ndarray | None = None) -> np.ndarray:
    """End heights (m) per pair and sample: (pairs, samples, H, W). ``starts`` replaces the start heights
    (for chained calls). Sample ``k`` of pair ``i`` uses noise key ``(seed, i, k)``, the same for every model."""
    device = next(model.parameters()).device
    generative = is_generative(model)
    samples = samples if generative else 1
    autocast = torch.autocast(device.type, dtype=torch.bfloat16, enabled=settings.bf16 and device.type == "cuda")
    out = None
    per_batch = max(1, settings.batch_size // samples)
    for i0 in range(0, len(pairs), per_batch):
        rows = range(i0, min(i0 + per_batch, len(pairs)))
        items = [dataset.item(*pairs[i], z_a=None if starts is None else starts[i]) for i in rows]
        x = torch.stack([it["x"] for it in items]).repeat_interleave(samples, 0).to(device)
        cond = torch.stack([it["cond"] for it in items]).repeat_interleave(samples, 0).to(device)
        noise = None
        if generative:
            keys = [(settings.seed, i, k) for i in rows for k in range(samples)]
            noise = initial_noise(keys, x.shape[-2:], model.fixed_edges)
        with autocast:
            y = generate(model, x, cond, noise, steps).float()[:, 0].cpu().double().numpy()
        y = y.reshape(len(items), samples, *y.shape[1:])
        for j, (i, it) in enumerate(zip(rows, items)):
            run, a, b = pairs[i]
            dT = dataset.times[b] - dataset.times[a]
            dz = dataset.scaling.change.from_model(y[j], dataset.columns["K_ref"][run], dT)
            if out is None:
                out = np.empty((len(pairs), samples, *dz.shape[-2:]))
            out[i] = it["z_a"].double().numpy() + dz
    return out


BASELINES = ("no_change", "analytical")


def evaluate_erosion(models: dict[str, Path], out: Path, settings: Settings = Settings(), data_root: Path | None = None,
                     device: str | None = None, record: dict | None = None,
                     baselines: dict[str, dict] | None = None) -> Path:
    """Score each model (name -> model folder) and each baseline (name -> {"method": ...}) on the test pairs
    and write the results to ``out``."""
    baselines = baselines or {}
    for name, spec in baselines.items():
        if spec["method"] not in BASELINES:
            raise ValueError(f"{name}: the erosion suite has no {spec['method']} baseline; it has {BASELINES}")
    loaded = {label: load_checkpoint(folder, device=device) for label, folder in models.items()}
    for label, (_, _, cfg, _) in loaded.items():
        if cfg.task != "erosion" or cfg.window is not None or cfg.cell != 1:
            raise ValueError(f"{label} is not a whole-run erosion model; the world suite scores world levels")
    if not (loaded or baselines):
        raise ValueError("nothing to score: no models and no baselines")
    if not (settings.data_profile or loaded):
        raise ValueError("settings.data_profile is needed when only baselines are scored")
    profile = settings.data_profile or next(iter(loaded.values()))[2].data_profile
    test = Split.open(profile, settings.split, data_root)
    runs = test.select(**settings.data)[: settings.runs]
    pairs = choose_pairs(test, runs, settings)
    if not len(pairs):
        raise ValueError("no test pairs: check the filters and ends_yr")
    run, start, end = pairs.T
    z_a = np.asarray(test.snapshots[run, start], dtype=np.float64)
    z_b = np.asarray(test.snapshots[run, end], dtype=np.float64)
    dT = test.times[end] - test.times[start]
    truths = [metrics.Truth(a, b, test.columns["m"][r]) for a, b, r in zip(z_a, z_b, run)]

    heights: dict[str, np.ndarray] = {}
    rows_of: dict[str, np.ndarray] = {}  # entries scored on a subset of pairs (chains)
    entries: dict[str, dict] = {}
    spreads, seconds = {}, {}
    for name, spec in baselines.items():
        if spec["method"] == "no_change":
            heights[name] = z_a
        else:
            heights[name] = np.stack([
                analytical_stream_power(z_a[i], test.columns["K"][r], test.columns["m"][r], dT[i],
                                        test.erodibility_map(r).factors(0, 0, *z_a.shape[-2:]))
                for i, r in enumerate(run)])
        entries[name] = {"kind": "baseline", "method": spec["method"]}

    for label, (model, scaling, cfg, _) in loaded.items():
        dataset = ErosionPairs(test, scaling, inputs=cfg.inputs)
        for steps in settings.step_sweep or (settings.steps,):
            entry = f"{label}@{steps}" if settings.step_sweep else label
            begin = time.perf_counter()
            z = _predict(model, dataset, pairs, settings, steps, settings.samples)
            seconds[entry] = (time.perf_counter() - begin) / (len(pairs) * z.shape[1])
            heights[entry] = z[:, 0]
            entries[entry] = {"kind": "model", "model": label, "steps": steps, "network_calls": network_calls(model, steps)}
            if z.shape[1] > 1:
                spreads[entry] = z.std(axis=1)
            if settings.fill_pits:
                heights[f"{entry} + pit filling"] = np.stack([fill_pits(h) for h in z[:, 0]])
                entries[f"{entry} + pit filling"] = {**entries[entry], "kind": "postprocessed"}
        for calls, years in settings.chains:
            rows = np.flatnonzero(np.isclose(dT, calls * years) & (start == 0))
            step_end = np.flatnonzero(np.isclose(test.times - test.times[0], years))
            if not len(rows) or not len(step_end):
                continue
            current = np.asarray(test.snapshots[run[rows], 0], dtype=np.float64)
            step_pairs = np.stack([run[rows], np.zeros(len(rows), int), np.full(len(rows), step_end[0])], axis=1)
            for _ in range(calls):
                current = _predict(model, dataset, step_pairs, settings, settings.steps, 1, starts=current)[:, 0]
            entry = f"{label} chained {calls}x{years:g}"
            full = np.full_like(z_a, np.nan)
            full[rows] = current
            heights[entry], rows_of[entry] = full, rows
            entries[entry] = {"kind": "chained", "model": label, "calls": calls, "years_per_call": years,
                              "network_calls": calls * network_calls(model, settings.steps)}

    per_pair, spectra = {}, {"wavelength": metrics.radial_power_spectrum(z_a[0])[0],
                             "simulator": np.mean([t.change_spectrum for t in truths], axis=0)}
    for entry, z in heights.items():
        rows = rows_of.get(entry, np.arange(len(pairs)))
        values = [metrics.pair_metrics(z[i], truths[i]) for i in rows]
        per_pair[entry] = {k: np.full(len(pairs), np.nan) for k in metrics.PAIR_METRICS}
        for k in metrics.PAIR_METRICS:
            per_pair[entry][k][rows] = [v[k] for v in values]
        spectra[entry] = np.mean([metrics.radial_power_spectrum(z[i] - z_a[i])[1] for i in rows], axis=0)

    factors = _factor_values(test, run, dT)
    scalars = {"entries": {}, "by": {}}
    for entry, info in entries.items():
        rows = rows_of.get(entry)
        summary = metrics.summarize(per_pair[entry], rows)
        summary["fine_detail"] = metrics.fine_detail(spectra["wavelength"], spectra[entry], spectra["simulator"])
        if entry in spreads:
            summary["spread_correlation"] = metrics.spread_correlation(spreads[entry], np.abs(heights[entry] - z_b))
        if entry in seconds:
            summary["seconds_per_sample"] = seconds[entry]
        scalars["entries"][entry] = summary
    for factor, values in factors.items():
        groups = _groups(values, None if factor in CATEGORICAL or factor == "dT" else CONTINUOUS[factor])
        if len(groups) < 2:
            continue
        scalars["by"][factor] = {group: {entry: metrics.summarize(per_pair[entry], mask & np.isfinite(
            per_pair[entry]["sq_error"])) for entry in entries} for group, mask in groups.items()}

    arrays = {f"pairs/{k}": v for k, v in {"run": run, "start": start, "end": end, **factors}.items()}
    arrays |= {f"{entry}/{k}": v for entry, d in per_pair.items() for k, v in d.items()}
    arrays |= {f"spectrum/{k}": v for k, v in spectra.items()}
    n = min(settings.examples, len(pairs))
    examples = {"z_a": z_a[:n], "z_b": z_b[:n]} | {entry: z[:n] for entry, z in heights.items()}
    manifest = {
        "suite": "erosion",
        "models": model_records(models),
        "entries": entries,
        "test": {"data_profile": profile, "split": settings.split, "runs": int(len(runs)), "pairs": int(len(pairs))},
        "settings": asdict(settings),
    }
    return write_results(out, scalars, arrays, examples, {**manifest, **(record or {})})
