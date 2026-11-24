"""Erosion datasets: one file per group in the repo's ``configs/data/<dataset>/<group>.yaml``.

A dataset (``erosion_master``) is one table of simulator runs, stored in blocks called groups
(docs/plans/experiments-plan.md, "Erosion data"). A group is one set of generator settings. Its name is
the file's path under ``configs/data/``, such as ``erosion_master/main_256``, and its data goes to the same
path under ``artifacts/data/``, one folder per split.

A group file has ``generator: erosion`` and every field of ``Group`` but ``name`` and ``dataset``, which
come from the path.
"""

import json
from dataclasses import dataclass, fields
from pathlib import Path

import yaml

# Datasets go where the training code reads them: the "artifacts" path in the main repo's local.json if
# it exists, else the repo's artifacts/ folder. Same rule as erosion_sim.paths, which this environment
# cannot import.
_REPO_DIR = Path(__file__).resolve().parents[3]
_LOCAL = _REPO_DIR / "local.json"
ARTIFACTS_DIR = (Path(json.loads(_LOCAL.read_text())["artifacts"]).expanduser() if _LOCAL.exists()
                 else _REPO_DIR / "artifacts")
DATA_CONFIGS = _REPO_DIR / "configs" / "data"


@dataclass(frozen=True)
class Group:
    """Settings for one group. Ranges are (low, high), both ends included for integers.

    Every column of a run is drawn independently from the run's own random stream, seeded by
    ``run_seed`` (SHA-256 of the dataset, group, split, and index). So a run's settings do not depend on
    how many runs the split has.
    """

    name: str  # "main_256"
    dataset: str  # "erosion_master"
    size: int  # grid cells per side, at 1 m
    train_runs: int
    val_runs: int
    test_runs: int
    steps: int
    snapshot_every: int
    dt: float  # yr per step
    cell_size: float  # m
    # Starting terrain (erosion_terrain.fractal_noise)
    period: tuple[float, float]  # cells, log-uniform
    height: tuple[float, float]  # m per unit of noise, log-uniform
    octaves: tuple[int, int]  # uniform integer
    persistence: tuple[float, float]  # uniform
    lacunarity: tuple[float, float]  # uniform
    noise_types: tuple[str, ...]  # uniform
    fractal_types: tuple[str, ...]  # uniform
    warp_fraction: float  # share of runs with domain warp
    warp: tuple[float, float]  # uniform, when on
    # Erosion. K is drawn as K_ref, the erodibility at m = m_ref. The simulator uses
    # K = K_ref * A_ref ** (m_ref - m), so a channel draining A_ref m^2 cuts at the same rate for any m,
    # and m changes channel shape rather than overall rate. D is drawn as a ratio to K_ref.
    K_ref: tuple[float, float]  # log-uniform
    D_over_K: tuple[float, float]  # log-uniform
    m: tuple[float, float]  # uniform
    m_ref: float
    A_ref: float  # m^2; see notebooks/datagen/parameter-ranges.ipynb
    # Erodibility map (erosion_terrain.ErodibilityMap)
    maps: tuple[tuple[str, float], ...]  # (shape, share)
    map_contrast: tuple[float, float]  # log-uniform; 1 for uniform maps
    map_period: tuple[float, float]  # cells, log-uniform, patches only
    band_width: tuple[float, float]  # cells, log-uniform, bands only
    # Out-of-range draws: run i replaces parameter ood_ranges[i % len][0]'s range with ood_ranges[i % len][1]
    # (parameters K_ref, D_over_K, m). Empty: every run in range.
    ood_ranges: tuple[tuple[str, tuple[float, float]], ...]

    @property
    def profile(self) -> str:
        """``<dataset>/<name>``: the file under ``configs/data/`` and the folder under ``artifacts/data/``."""
        return f"{self.dataset}/{self.name}"

    def runs(self, split: str) -> int:
        return {"train": self.train_runs, "val": self.val_runs, "test": self.test_runs}[split]

    @property
    def splits(self) -> tuple[str, ...]:
        return tuple(s for s in ("train", "val", "test") if self.runs(s))

    @property
    def snapshots(self) -> int:
        return self.steps // self.snapshot_every + 1


def _tuples(value):
    return tuple(_tuples(v) for v in value) if isinstance(value, list) else value


def load_group(profile: str, root: Path | None = None) -> Group:
    """The group in ``configs/data/<profile>.yaml``, for example ``erosion_master/main_256``. The file must
    list every field but ``name`` and ``dataset``."""
    path = (Path(root) if root is not None else DATA_CONFIGS) / f"{profile}.yaml"
    if profile.count("/") != 1 or not path.exists():
        raise KeyError(f"no erosion group {profile!r}: expected a file configs/data/<dataset>/<group>.yaml; "
                       f"groups: {', '.join(group_names(root))}")
    values = yaml.safe_load(path.read_text()) or {}
    if values.pop("generator", None) != "erosion":
        raise ValueError(f"{path}: not an erosion group (generator must be erosion)")
    required = {f.name for f in fields(Group)} - {"name", "dataset"}
    missing, unknown = sorted(required - set(values)), sorted(set(values) - required)
    if missing or unknown:
        raise ValueError(f"{path}: missing fields {missing}, unknown fields {unknown}")
    dataset, name = profile.split("/")
    return Group(name=name, dataset=dataset, **{k: _tuples(v) for k, v in values.items()})


def group_names(root: Path | None = None) -> list[str]:
    """Every erosion group under ``configs/data/``, as ``<dataset>/<group>``."""
    root = Path(root) if root is not None else DATA_CONFIGS
    return sorted(str(p.relative_to(root).with_suffix("")) for p in root.glob("*/*.yaml")
                  if (yaml.safe_load(p.read_text()) or {}).get("generator") == "erosion")
