"""Erosion dataset groups: grid, ranges, run counts, and how each run's settings are drawn.

A dataset (``erosion_master``) is one table of simulator runs, stored in blocks called groups
(docs/plans/experiments-plan.md, "Erosion data"). A group is one set of generator settings. Each
(group, split) is one folder, ``artifacts/data/<dataset>/<group>/<split>/``.
"""

import os
from dataclasses import dataclass
from pathlib import Path

# Datasets go where the training code reads them: $EROSION_ARTIFACTS if set, else the main repo's
# artifacts/ folder. Same rule as erosion_sim.paths, which this environment cannot import.
ARTIFACTS_DIR = Path(os.environ.get("EROSION_ARTIFACTS") or Path(__file__).resolve().parents[3] / "artifacts")

ALL_NOISE = ("perlin", "opensimplex2", "value_cubic")
ALL_FRACTAL = ("fbm", "ridged")
# map_shape shares in the main groups: most runs vary only the noise.
MAIN_MAPS = (("uniform", 2 / 3), ("patches", 1 / 6), ("bands", 1 / 6))
UNIFORM_MAP = (("uniform", 1.0),)
# Out-of-range draws for ood groups, chosen by index % 3. All above the training range.
OOD_RANGES = (("K_ref", (5e-4, 1e-3)), ("D_over_K", (3.0, 6.0)), ("m", (0.6, 0.7)))


@dataclass(frozen=True)
class Group:
    """Settings for one group. Ranges are (low, high), both ends included for integers.

    Every column of a run is drawn independently from the run's own random stream, seeded by
    ``run_seed`` (SHA-256 of the dataset, group, split, and index). So a run's settings do not depend on
    how many runs the split has.
    """

    name: str
    size: int  # grid cells per side, at 1 m
    train_runs: int
    val_runs: int
    test_runs: int
    steps: int = 500
    snapshot_every: int = 25
    dataset: str = "erosion_master"
    dt: float = 100.0  # yr per step
    cell_size: float = 1.0  # m
    # Starting terrain (erosion_terrain.fractal_noise)
    period: tuple[float, float] = (32.0, 64.0)  # cells, log-uniform
    height: tuple[float, float] = (32.0, 128.0)  # m per unit of noise, log-uniform
    octaves: tuple[int, int] = (4, 7)  # uniform integer
    persistence: tuple[float, float] = (0.35, 0.65)  # uniform
    lacunarity: tuple[float, float] = (1.8, 2.4)  # uniform
    noise_types: tuple[str, ...] = ALL_NOISE  # uniform
    fractal_types: tuple[str, ...] = ALL_FRACTAL  # uniform
    warp_fraction: float = 0.5  # share of runs with domain warp
    warp: tuple[float, float] = (0.1, 0.5)  # uniform, when on
    # Erosion. K is drawn as K_ref, the erodibility at m = m_ref. The simulator uses
    # K = K_ref * A_ref ** (m_ref - m), so a channel draining A_ref m^2 cuts at the same rate for any m,
    # and m changes channel shape rather than overall rate. D is drawn as a ratio to K_ref.
    K_ref: tuple[float, float] = (1e-4, 5e-4)  # log-uniform
    D_over_K: tuple[float, float] = (0.3, 3.0)  # log-uniform
    m: tuple[float, float] = (0.3, 0.6)  # uniform
    m_ref: float = 0.4
    A_ref: float = 30.0  # m^2; see notebooks/datagen/parameter-ranges.ipynb
    # Erodibility map (erosion_terrain.ErodibilityMap)
    maps: tuple[tuple[str, float], ...] = MAIN_MAPS  # (shape, share)
    map_contrast: tuple[float, float] = (1.0, 10.0)  # log-uniform; 1 for uniform maps
    map_period: tuple[float, float] = (32.0, 128.0)  # cells, log-uniform, patches only
    band_width: tuple[float, float] = (8.0, 64.0)  # cells, log-uniform, bands only
    ood: bool = False  # one erosion parameter out of range per run (OOD_RANGES)
    processes: int | None = None  # None uses every core

    def runs(self, split: str) -> int:
        return {"train": self.train_runs, "val": self.val_runs, "test": self.test_runs}[split]

    @property
    def splits(self) -> tuple[str, ...]:
        return tuple(s for s in ("train", "val", "test") if self.runs(s))

    @property
    def snapshots(self) -> int:
        return self.steps // self.snapshot_every + 1


def _scaled(low_high: tuple[float, float], k: float) -> tuple[float, float]:
    return (low_high[0] * k, low_high[1] * k)


def _main(name: str, size: int, runs: tuple[int, int, int], scale: float = 1.0, snapshot_every: int = 25) -> Group:
    """A main group: band 1's ranges times ``scale`` (bands 2 and 3 are × 2 and × 4)."""
    base = Group(name="", size=0, train_runs=0, val_runs=0, test_runs=0)
    return Group(name=name, size=size, train_runs=runs[0], val_runs=runs[1], test_runs=runs[2],
                 snapshot_every=snapshot_every, period=_scaled(base.period, scale), height=_scaled(base.height, scale),
                 map_period=_scaled(base.map_period, scale), band_width=_scaled(base.band_width, scale))


def _test_only(name: str, size: int, runs: int, **changes) -> Group:
    """A test-only group: Perlin, fBm, no warp, uniform map, in-range parameters, plus ``changes``."""
    return Group(name=name, size=size, train_runs=0, val_runs=0, test_runs=runs, noise_types=("perlin",),
                 fractal_types=("fbm",), warp_fraction=0.0, maps=UNIFORM_MAP, **changes)


GROUPS = {
    # Seconds on CPU, for tests: every noise choice and map shape, on a tiny grid.
    "smoke": Group(name="smoke", dataset="erosion_smoke", size=32, train_runs=6, val_runs=3, test_runs=3, steps=20,
                   snapshot_every=5, period=(8.0, 16.0), map_period=(8.0, 16.0), band_width=(4.0, 8.0), processes=2),
    "smoke_ood": _test_only("smoke_ood", 32, 3, dataset="erosion_smoke", steps=20, snapshot_every=5,
                            period=(8.0, 16.0), ood=True, processes=2),
    # Main groups, one per scale band (track B: main_256; track C: main_512, main_1024, main_2048).
    "main_256": _main("main_256", 256, (4_000, 400, 400)),
    "main_512": _main("main_512", 512, (3_000, 300, 300)),
    "main_1024": _main("main_1024", 1024, (1_200, 150, 150), scale=2, snapshot_every=50),
    "main_2048": _main("main_2048", 2048, (300, 50, 50), scale=4, snapshot_every=50),
    # Test-only groups.
    "ood_256": _test_only("ood_256", 256, 200, ood=True),
    "ood_512": _test_only("ood_512", 512, 200, ood=True),
    "long_256": _test_only("long_256", 256, 200, steps=1_500, snapshot_every=50),
    "long_512": _test_only("long_512", 512, 200, steps=1_500, snapshot_every=50),
}
