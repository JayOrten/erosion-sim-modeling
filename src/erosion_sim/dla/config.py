"""DLA dataset profiles: grid, particle count, and the runs in each split (docs/plans/experiments-plan.md)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DataConfig:
    """Settings for one DLA dataset. Every run is grown to ``particles`` particles on a ``size``² grid.

    - train and val: runs from the seed particle, ``p`` log-uniform over ``p_range``.
    - test: ``test_runs_per_p`` runs from the seed particle at each of ``test_p``; then, for each entry of
      ``start_p``, one start cluster grown to ``start_particles`` at that ``p`` and continued
      ``runs_per_start`` times with different seeds.
    """

    profile: str
    size: int = 128
    # At 1,000 particles, 11% of clusters at p = 1 outgrow 128². At 600, the largest radius in 10,000
    # runs at p = 1 was 58 of the 60 that fit.
    particles: int = 600
    train_runs: int = 200_000
    val_runs: int = 2_000
    p_range: tuple[float, float] = (0.05, 1.0)  # log-uniform
    test_p: tuple[float, ...] = (0.05, 0.1, 0.2, 0.5, 1.0)
    test_runs_per_p: int = 200
    start_p: tuple[float, ...] = (0.1,) * 10 + (1.0,) * 10  # one entry per shared start
    start_particles: int = 200
    runs_per_start: int = 50
    processes: int | None = None  # None uses every core the process may use

    def runs(self, split: str) -> int:
        if split == "test":
            return len(self.test_p) * self.test_runs_per_p + len(self.start_p) * self.runs_per_start
        return {"train": self.train_runs, "val": self.val_runs}[split]


SPLITS = ("train", "val", "test")

PROFILES = {
    # Seconds on CPU, for tests.
    "dla_smoke": DataConfig(
        profile="dla_smoke",
        size=48,
        particles=60,
        train_runs=8,
        val_runs=4,
        test_p=(0.1, 1.0),
        test_runs_per_p=3,
        start_p=(0.1, 1.0),
        start_particles=20,
        runs_per_start=2,
        processes=1,
    ),
    "dla_master": DataConfig(profile="dla_master"),
}
