"""Load generated erosion datasets as training pairs.

A dataset directory comes from ``datagen/`` (see ``datagen/README.md``). It holds every snapshot of
every run. A training sample is a pair of snapshots from one run: a start ``z_a``, a later ``z_b``,
and the elapsed time between them.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "artifacts"

# Scalars the model is conditioned on, in order. D and K do not depend on height scale when n = 1;
# U does, so it is divided by the start's relief.
COND_NAMES = ("log10_dT", "log10_K", "log10_D", "m", "U_over_relief")


def dataset_dir(profile: str, split: str) -> Path:
    return ARTIFACTS_DIR / "data" / profile / split


def load_params(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    """Return (params, meta) for a dataset directory. ``params["times"]`` holds the snapshot times in yr."""
    path = Path(path)
    with np.load(path / "params.npz") as f:
        params = dict(f)
    return params, json.loads((path / "meta.json").read_text())


def select_runs(params: dict[str, np.ndarray], **ranges: tuple[float, float]) -> np.ndarray:
    """Indices of runs whose parameters fall in every given inclusive range.

    Ranges can name any parameter in ``params`` plus ``D_over_K`` and ``U_over_K`` (ratios to ``K_ref``).
    Example: ``select_runs(params, U=(0, 0), K_ref=(1e-4, 2e-4))``.
    """
    derived = {"D_over_K": params["D"] / params["K_ref"], "U_over_K": params["U"] / params["K_ref"]}
    keep = np.ones(len(params["K"]), dtype=bool)
    for name, (low, high) in ranges.items():
        values = derived[name] if name in derived else params[name]
        keep &= (values >= low) & (values <= high)
    return np.flatnonzero(keep)


def cond_values(params, runs, dT, relief) -> np.ndarray:
    """Raw conditioning scalars, shape (len(runs), len(COND_NAMES))."""
    return np.stack([
        np.log10(dT),
        np.log10(params["K"][runs]),
        np.log10(params["D"][runs]),
        params["m"][runs],
        params["U"][runs] / relief,
    ], axis=-1)


@dataclass(frozen=True)
class Normalizer:
    """Fixed scalings, fitted on the training split and reused for every split and for inference.

    - Conditioning scalars are mapped linearly so the training range becomes [0, 1].
    - The target is ``(z_b - z_a) / relief(z_a)``. With ``scale="fitted"``, it is further divided by
      ``coef * (K_ref * dT) ** power``, the typical RMS (root mean square) of that change, so short
      and long elapsed times have targets of similar size. With ``scale="relief"``, it is not.
    """

    cond_low: tuple[float, ...]
    cond_high: tuple[float, ...]
    scale: str = "fitted"
    coef: float = 1.0
    power: float = 0.0

    @classmethod
    def fit(cls, path: Path, scale: str = "fitted", runs: np.ndarray | None = None, sample: int = 400, seed: int = 0):
        """Fit on a sample of runs from the dataset at ``path``."""
        if scale not in ("fitted", "relief"):
            raise ValueError("scale must be 'fitted' or 'relief'")
        params, _ = load_params(path)
        snapshots = np.load(Path(path) / "snapshots.npy", mmap_mode="r")
        times = params["times"]
        runs = np.arange(len(params["K"])) if runs is None else np.asarray(runs)
        rng = np.random.default_rng(seed)
        chosen = np.sort(rng.choice(runs, min(sample, len(runs)), replace=False))

        cond, kdt, rms = [], [], []
        count = len(times)
        starts = sorted({0, count // 5, count // 2, 4 * count // 5} - {count - 1})  # a few start points per run
        for r in chosen:
            s = np.asarray(snapshots[r], dtype=np.float64)
            reliefs = s.max(axis=(1, 2)) - s.min(axis=(1, 2))
            # Conditioning ranges cover every start, since U / relief grows as relief drops.
            cond.append(cond_values(params, np.full(count, r), np.full(count, times[1] - times[0]), reliefs))
            for a in starts:
                dT = times[a + 1:] - times[a]
                kdt.append(params["K_ref"][r] * dT)
                rms.append(np.sqrt((((s[a + 1:] - s[a]) / reliefs[a]) ** 2).mean(axis=(1, 2))))
        cond = np.concatenate(cond)
        # Longest elapsed time gives the conditioning range for dT; the others come from the runs.
        low, high = cond.min(axis=0), cond.max(axis=0)
        low[0], high[0] = np.log10(times[1] - times[0]), np.log10(times[-1] - times[0])
        coef, power = 1.0, 0.0
        if scale == "fitted":
            power, log_coef = np.polyfit(np.log(np.concatenate(kdt)), np.log(np.concatenate(rms)), 1)
            coef = float(np.exp(log_coef))
        return cls(tuple(map(float, low)), tuple(map(float, high)), scale, coef, float(power))

    def cond(self, raw: np.ndarray) -> np.ndarray:
        low, high = np.array(self.cond_low), np.array(self.cond_high)
        return (raw - low) / np.where(high > low, high - low, 1.0)

    def target_scale(self, K_ref, dT):
        """Typical size of the relief-normalized change. Works on numbers, arrays, or tensors."""
        if self.scale == "relief":
            return 1.0 + 0.0 * K_ref * dT
        return self.coef * (K_ref * dT) ** self.power

    def to_dict(self) -> dict:
        return asdict(self)


class ErosionPairs(Dataset):
    """Every (run, start, end) snapshot pair of the selected runs, as tensors.

    Each item is a dict:

    - ``x``: (1, H, W) start heightmap, scaled to [0, 1] by its own min and relief.
    - ``y``: (1, H, W) target, the normalized change (see ``Normalizer``).
    - ``cond``: (len(COND_NAMES),) conditioning scalars, scaled by the normalizer.
    - ``z_a``, ``z_b``: (H, W) the raw heightmaps in m, for evaluation.
    - ``run``, ``start``, ``end``: snapshot indices; ``relief`` and ``scale`` to undo the normalization.

    ``lags`` limits which elapsed times are used, in snapshot steps (1 = one snapshot apart). Use it to
    hold elapsed times out of training.
    """

    def __init__(self, path: Path, normalizer: Normalizer, runs: np.ndarray | None = None, lags=None):
        self.path = Path(path)
        self.normalizer = normalizer
        self.params, self.meta = load_params(self.path)
        self.times = self.params["times"]
        self.runs = np.arange(len(self.params["K"])) if runs is None else np.asarray(runs)
        count = len(self.times)
        pairs = [(a, b) for a in range(count) for b in range(a + 1, count)]
        if lags is not None:
            lags = set(int(k) for k in lags)
            pairs = [(a, b) for a, b in pairs if b - a in lags]
        self.pairs = np.array(pairs, dtype=np.int64).reshape(-1, 2)
        self._snapshots = None  # opened lazily, so each DataLoader worker opens its own memory map

    @property
    def snapshots(self) -> np.ndarray:
        if self._snapshots is None:
            self._snapshots = np.load(self.path / "snapshots.npy", mmap_mode="r")
        return self._snapshots

    def __len__(self) -> int:
        return len(self.runs) * len(self.pairs)

    def __getitem__(self, index: int) -> dict:
        run = int(self.runs[index // len(self.pairs)])
        a, b = (int(v) for v in self.pairs[index % len(self.pairs)])
        z_a = np.array(self.snapshots[run, a], dtype=np.float32)  # copies out of the read-only memory map
        z_b = np.array(self.snapshots[run, b], dtype=np.float32)
        low = float(z_a.min())
        relief = float(z_a.max()) - low
        dT = float(self.times[b] - self.times[a])
        scale = float(self.normalizer.target_scale(self.params["K_ref"][run], dT))
        cond = self.normalizer.cond(cond_values(self.params, np.array([run]), np.array([dT]), relief)[0])
        return {
            "x": torch.from_numpy((z_a - low) / relief)[None],
            "y": torch.from_numpy((z_b - z_a) / (relief * scale))[None],
            "cond": torch.from_numpy(cond.astype(np.float32)),
            "z_a": torch.from_numpy(z_a),
            "z_b": torch.from_numpy(z_b),
            "run": run,
            "start": a,
            "end": b,
            "relief": relief,
            "scale": scale,
        }


def denormalize(y: torch.Tensor, z_a: torch.Tensor, relief, scale) -> torch.Tensor:
    """Turn a normalized change ``y`` (batch, 1, H, W) back into an end heightmap in m, shape (batch, H, W)."""
    relief = torch.as_tensor(relief, dtype=y.dtype, device=y.device).reshape(-1, 1, 1)
    scale = torch.as_tensor(scale, dtype=y.dtype, device=y.device).reshape(-1, 1, 1)
    return z_a + y[:, 0] * relief * scale
