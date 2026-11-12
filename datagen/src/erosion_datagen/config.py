"""Dataset profiles: simulator settings, parameter ranges, and run counts."""

from dataclasses import dataclass
from pathlib import Path

# Datasets go in the main repo's artifacts/ folder, where the training code reads them.
ARTIFACTS_DIR = Path(__file__).resolve().parents[3] / "artifacts"


@dataclass(frozen=True)
class Config:
    """Settings for one dataset. Ranges are (low, high); see docs/plans/unet-baseline-plan.md."""

    profile: str
    shape: tuple[int, int]
    train_runs: int
    val_runs: int
    test_runs: int
    steps: int
    snapshot_every: int
    dt: float = 100.0  # yr per step
    cell_size: float = 1.0  # m
    # m per unit of noise. A 128-cell window at period 32 then spans about 67 m.
    height: float = 64.0
    # Starting terrain (erosion_terrain.fractal_noise, in ../terrain). period is in cells.
    period: tuple[float, float] = (32.0, 256.0)  # log-uniform
    persistence: tuple[float, float] = (0.35, 0.65)  # uniform
    octaves: int = 5
    lacunarity: float = 2.0
    # Terrain variety, drawn per run when set (each run stores what it drew). None keeps the fixed value
    # above. Types are names from erosion_terrain.NOISE_TYPES and FRACTAL_TYPES, drawn uniformly.
    noise_types: tuple[str, ...] = ("perlin",)
    fractal_types: tuple[str, ...] = ("fbm",)
    octave_range: tuple[int, int] | None = None  # uniform, both ends included
    lacunarity_range: tuple[float, float] | None = None  # uniform
    height_range: tuple[float, float] | None = None  # log-uniform, m per unit of noise
    warp_fraction: float = 0.0  # fraction of runs with domain warp
    warp: tuple[float, float] = (0.1, 0.5)  # uniform warp strength, for the warped runs
    # Erosion. K is drawn as K_ref, the erodibility at m = m_ref. The simulator uses
    # K = K_ref * A_ref ** (m_ref - m), so a channel draining A_ref m^2 cuts at the same rate for any m,
    # and m changes channel shape rather than overall rate. D and U are drawn as ratios to K_ref, so
    # random draws avoid combinations that only blur or only build a plateau.
    K_ref: tuple[float, float] = (1e-4, 5e-4)  # log-uniform
    m_ref: float = 0.4
    # m^2. Chosen so the overall change over a run is about the same across the m range; see
    # notebooks/datagen/parameter-ranges.ipynb.
    A_ref: float = 30.0
    D_over_K: tuple[float, float] = (0.3, 3.0)  # log-uniform, D / K_ref
    m: tuple[float, float] = (0.3, 0.6)  # uniform
    U_zero_fraction: float = 1 / 3  # fraction of runs with no uplift
    U_over_K: tuple[float, float] = (0.1, 1.5)  # log-uniform, U / K_ref, for the other runs
    processes: int | None = None  # None uses every core

    def runs(self, split: str) -> int:
        return {"train": self.train_runs, "val": self.val_runs, "test": self.test_runs}[split]


PROFILES = {
    # Seconds on CPU, for tests.
    "smoke": Config(
        profile="smoke",
        shape=(32, 32),
        train_runs=6,
        val_runs=3,
        test_runs=3,
        steps=20,
        snapshot_every=5,
        period=(8.0, 64.0),
        processes=2,
    ),
    # About a minute on 32 cores. For checking the pipeline end to end before the full run.
    "small": Config(
        profile="small",
        shape=(128, 128),
        train_runs=512,
        val_runs=64,
        test_runs=64,
        steps=500,
        snapshot_every=10,
    ),
    # About 10 minutes on 32 cores and 33 GB on disk.
    "full": Config(
        profile="full",
        shape=(128, 128),
        train_runs=8_000,
        val_runs=1_000,
        test_runs=1_000,
        steps=500,
        snapshot_every=10,
    ),
    # The full dataset's size, with the period fixed at the infinite world's 32 cells
    # (docs/plans/infinite-plan.md).
    "world": Config(
        profile="world",
        shape=(128, 128),
        train_runs=8_000,
        val_runs=1_000,
        test_runs=1_000,
        steps=500,
        snapshot_every=10,
        period=(32.0, 32.0),
    ),
    # The master dataset for the infinite world (docs/plans/infinite-plan.md, step 9): large domains, so
    # windows can be cropped far from the simulator's fixed edges; periods from the tested 32 cells up to
    # 64; and varied terrain. Each run stores its terrain settings, so experiments pick runs by them.
    # No uplift, and a snapshot every 5,000 yr to fit on disk: about 11.5 MB per run, 28 GB in all.
    # 16 workers: on this machine (8 performance and 16 efficiency cores), more share memory bandwidth
    # and run slower; 512² runs peak at 16 (27 M cell-steps per second), about 3 hours for the dataset.
    "world_master": Config(
        profile="world_master",
        shape=(512, 512),
        train_runs=2_000,
        val_runs=200,
        test_runs=200,
        steps=500,
        snapshot_every=50,
        period=(32.0, 64.0),
        U_zero_fraction=1.0,
        noise_types=("perlin", "opensimplex2", "value_cubic"),
        fractal_types=("fbm", "ridged"),
        octave_range=(4, 7),
        lacunarity_range=(1.8, 2.4),
        height_range=(32.0, 128.0),
        warp_fraction=0.5,
        warp=(0.1, 0.5),
        processes=16,
    ),
}
