"""Compare model versions of the world on one fixed set of large simulator runs.

Every version is graded on the same answer key, so their numbers can be compared. The answer key is a
set of samples from a generated dataset (by default the first 16 test samples of ``world_master``). Each
sample is a square region with its own terrain and erosion parameters, simulated whole. Each version's
world is built on the same terrain and queried on the same region, and graded on the interior, ``margin``
cells in from the simulator's fixed edges.

- ``Version``: a pair of checkpoints (coarse and fine level) plus sampler settings.
- ``compare_versions``: every version against the simulator and the baseline, on every sample and
  elapsed time. Returns the ``pairs`` (format of ``erosion_sim.evaluate``) and summary tables.
- ``example_index``: which pair is (sample, elapsed time), for ``evaluate.show_examples``.

A version's predictions are saved in ``artifacts/compare/``, keyed by everything they depend on
(checkpoint files, sampler settings, samples, elapsed times, margin). Adding a version to a comparison
then only computes the new one.
"""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ..data import Split
from ..evaluate.pairs import add_method, make_pairs
from ..paths import ARTIFACTS_DIR, checkpoint_path
from .evaluate import by_elapsed, headline, run_world, smooth_upsample
from .level import SPLIT
from .world import load_levels

CACHE_DIR = ARTIFACTS_DIR / "compare"
DTS = (10_000.0, 30_000.0, 50_000.0)


@dataclass(frozen=True)
class Version:
    """One model version of the world.

    ``coarse`` and ``fine`` are training profiles; their checkpoints are
    ``artifacts/checkpoints/diffusion-<profile>.pt``. ``steps`` (None: the default), ``splits``, and
    ``bf16`` are the sampler settings, as in ``World``.
    """

    coarse: str
    fine: str
    steps: int | None = None
    splits: tuple[float, ...] = (SPLIT,)
    bf16: bool = False

    def paths(self) -> tuple[Path, Path]:
        return checkpoint_path(self.coarse, "diffusion"), checkpoint_path(self.fine, "diffusion")

    def load(self, device: str | None = None) -> list[tuple]:
        """``[(model, scaling, config)]`` for the coarse and fine level."""
        return load_levels(*self.paths(), device=device)

    def key(self) -> dict:
        """Everything a version's predictions depend on, except the samples."""
        files = {str(p.name): [p.stat().st_size, p.stat().st_mtime_ns] for p in self.paths()}
        return asdict(self) | {"files": files}


def _snapshot(data: Split, dT: float) -> int:
    k = int(np.argmin(np.abs(data.times - dT)))
    if abs(data.times[k] - dT) > 1e-6:
        raise ValueError(f"{data.path} has no snapshot at {dT} yr")
    return k


def _predict(version: Version, data: Split, runs: list[int], dTs, margin: int, show: bool) -> dict:
    """Per pair, the version's world over the interior, and its coarse level alone (smoothly upsampled)."""
    levels = version.load()
    n = data.snapshots.shape[-1] - 2 * margin
    world_rows, coarse_rows = [], []
    for i in tqdm(runs, desc="samples", disable=not show):
        z0 = np.asarray(data.snapshots[i, 0, margin:margin + n, margin:margin + n], dtype=np.float64)
        for dT in dTs:
            world = run_world(data, i, dT, *levels, splits=version.splits, steps=version.steps, bf16=version.bf16)
            b = world.block
            world_rows.append(world.get(margin, margin, n, n).astype(np.float32))
            coarse = world.coarse_change(margin // b, margin // b, n // b, n // b)
            coarse_rows.append((z0 + smooth_upsample(coarse, b)).astype(np.float32))
    return {"world": np.stack(world_rows), "coarse": np.stack(coarse_rows)}


def _cached_predict(version: Version, source: dict, data: Split, runs: list[int], dTs, margin: int,
                    show: bool) -> dict:
    """``_predict``, read from ``CACHE_DIR`` when nothing it depends on has changed. ``source`` names the
    dataset (``profile``, ``split``) for the cache key."""
    key = source | {"runs": runs, "dTs": list(dTs), "margin": margin} | version.key()
    digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]
    path = CACHE_DIR / f"{source['profile']}-{version.coarse}-{version.fine}-{digest}.npz"
    if path.exists():
        with np.load(path) as f:
            return {"world": f["world"], "coarse": f["coarse"]}
    out = _predict(version, data, runs, dTs, margin, show)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **out)
    return out


def compare_versions(versions: dict[str, Version], profile: str = "world_master", split: str = "test",
                     runs=range(16), dTs=DTS, margin: int = 64, coarse_only: bool = False,
                     root: Path | None = None, show: bool = True) -> dict:
    """Every version's world against the simulator, on every sample and elapsed time.

    ``runs`` are sample indices of ``Split.open(profile, split, root)``. Methods, each graded against the
    simulator's run over the interior:

    - ``"no change"``: the starting terrain. The floor any model must beat.
    - one per version (its name): the version's world, queried once for the whole interior.
    - with ``coarse_only``, also ``"<name>, coarse level only"``: its coarse change, smoothly upsampled.

    Returns:

    - ``pairs``: the ``pairs`` dict, pair ``r * len(dTs) + k`` being (``runs[r]``, ``dTs[k]``);
      ``example_index`` finds a pair.
    - ``table``: per method, ``rmse`` (m), ``skill``, ``iou`` (channel IoU within one cell), ``pits``
      (``headline``).
    - ``by_dT``: ``rmse`` and ``iou`` per method at each elapsed time (``by_elapsed``).
    - ``runs``: each sample's index, erosion parameters, ``D_over_K``, and terrain fields.
    """
    data = Split.open(profile, split, root)
    runs = [int(i) for i in runs]
    ends = [_snapshot(data, dT) for dT in dTs]
    inner = (slice(margin, data.snapshots.shape[-1] - margin),) * 2
    z_a = np.stack([np.asarray(data.snapshots[i, 0][inner], np.float64) for i in runs for _ in ends])
    z_b = np.stack([np.asarray(data.snapshots[i, k][inner], np.float64) for i in runs for k in ends])
    pairs = make_pairs(data, np.repeat(runs, len(ends)), 0, np.tile(ends, len(runs)), z_a, z_b)

    names = []
    for name, version in versions.items():
        if show:
            print(f"{name}: {version.coarse} + {version.fine}")
        out = _cached_predict(version, {"profile": profile, "split": split}, data, runs, dTs, margin, show)
        add_method(pairs, name, out["world"].astype(np.float64), "sample")
        names.append(name)
        if coarse_only:
            add_method(pairs, f"{name}, coarse level only", out["coarse"].astype(np.float64), "baseline")
            names.append(f"{name}, coarse level only")

    if show:
        print(f"\n{len(runs)} samples × {len(dTs)} elapsed times, interior {z_a.shape[-1]}², against the "
              "simulator run on the whole sample.")
    table = headline(pairs, show=show)
    by_dT = by_elapsed(pairs, names, bins=len(dTs), show=show)  # equal-count bins: one per elapsed time
    samples = []
    for i in runs:
        e = data.erosion(i)
        samples.append({"run": i, **e, "D_over_K": e["D"] / e["K_ref"], "terrain": asdict(data.terrain(i))})
    return {"pairs": pairs, "table": table, "by_dT": by_dT, "runs": samples}


def example_index(result: dict, run: int, dT: float) -> int:
    """The index in ``result["pairs"]`` of dataset sample ``run`` at elapsed time ``dT``."""
    pairs = result["pairs"]
    return int(np.flatnonzero((pairs["run"] == run) & (pairs["dT"] == dT))[0])
