"""Conversions between real units (meters, years) and the numbers the model sees.

There are three, kept separate:

1. Terrain. ``terrain_to_model`` takes the start heightmap, subtracts its mean, and divides by
   ``HEIGHT_UNIT`` (64 m). This is a fixed number, not fitted, so every sample and every window of the
   infinite world uses the same scale. It is the model input.

2. Erosion parameters. ``ParameterScaling`` maps the five conditioning parameters (``PARAMETER_NAMES``)
   linearly to [0, 1]: each one's minimum over the training runs becomes 0, its maximum 1.

3. Height change. ``ChangeScaling`` divides the change by its typical size, which grows with
   erodibility and elapsed time: ``coef * (K_ref * dT) ** power`` meters. Changes range from about 0.01 m
   to tens of meters; after this they are all about size 1, so small changes are not drowned out by
   large ones in training. It is the training target, and its inverse turns model output into meters.

``ParameterScaling`` and ``ChangeScaling`` are fitted on the training runs and saved with the model,
together as a ``Scaling``.

``change_rms`` is not a conversion. It measures the size of the training target, which the diffusion
model needs (``sigma_data``).
"""

from dataclasses import dataclass

import numpy as np

# Meters per model height unit, for the terrain input. The datagen terrain's height scale, so values are
# near 1.
HEIGHT_UNIT = 64.0

# The parameters the model is conditioned on, in order, before scaling to [0, 1]. Elapsed time, K, and D
# span orders of magnitude, so they are scaled as logarithms.
PARAMETER_NAMES = ("log10_dT", "log10_K", "log10_D", "m", "U")

def _starts(count: int) -> list[int]:
    """Start snapshots used when measuring change: a few spread over the run."""
    return sorted({0, count // 5, count // 2, 4 * count // 5} - {count - 1})


def terrain_to_model(z: np.ndarray) -> np.ndarray:
    """Heightmaps ``z`` (..., H, W) in m to model input: each map minus its mean, divided by ``HEIGHT_UNIT``."""
    z = np.asarray(z, dtype=np.float64)
    return (z - z.mean(axis=(-2, -1), keepdims=True)) / HEIGHT_UNIT


def parameter_values(params: dict, runs, dT) -> np.ndarray:
    """The unscaled ``PARAMETER_NAMES`` for each row: run ``runs[i]``'s parameters over elapsed time
    ``dT[i]`` (yr). Shape (rows, len(PARAMETER_NAMES))."""
    runs = np.asarray(runs)
    return np.stack([
        np.log10(np.broadcast_to(dT, runs.shape)),
        np.log10(params["K"][runs]),
        np.log10(params["D"][runs]),
        params["m"][runs],
        params["U"][runs],
    ], axis=-1)


@dataclass(frozen=True)
class ParameterScaling:
    """Maps each of ``PARAMETER_NAMES`` linearly so ``low`` becomes 0 and ``high`` becomes 1.

    Parameters outside the training range scale to values outside [0, 1]; the model is then extrapolating.
    """

    low: tuple[float, ...]
    high: tuple[float, ...]

    @classmethod
    def fit(cls, params: dict, runs, times) -> "ParameterScaling":
        """The range of each parameter over ``runs``. Elapsed time ranges from one snapshot step to the
        whole run (``times`` are the snapshot times)."""
        values = parameter_values(params, runs, 1.0)
        low, high = values.min(axis=0), values.max(axis=0)
        low[0], high[0] = np.log10(times[1] - times[0]), np.log10(times[-1] - times[0])
        return cls(tuple(map(float, low)), tuple(map(float, high)))

    def scale(self, params: dict, runs, dT) -> np.ndarray:
        """``parameter_values`` mapped to [0, 1] over the training range. Shape (rows, len(PARAMETER_NAMES))."""
        low, high = np.array(self.low), np.array(self.high)
        span = np.where(high > low, high - low, 1.0)  # a parameter fixed in training maps to 0
        return (parameter_values(params, runs, dT) - low) / span


@dataclass(frozen=True)
class ChangeScaling:
    """Divides a height change by its typical size, ``typical_change(K_ref, dT) = coef * (K_ref * dT) ** power`` m.

    ``K_ref`` is the erodibility as datagen draws it, ``dT`` the elapsed time (yr). Every method works on
    numbers, arrays, or tensors.
    """

    coef: float
    power: float

    @classmethod
    def fit(cls, snapshots: np.ndarray, params: dict, runs, times, sample: int = 400, seed: int = 0,
            block: int = 1) -> "ChangeScaling":
        """A straight line through log(RMS change in m) against log(K_ref * dT), over every snapshot pair
        from a few start points in a random sample of ``runs``.

        ``block`` > 1 fits on ``block`` × ``block`` means of the snapshots, as the coarse world level sees them.
        """
        kdt, rms = [], []
        for r, s in _sample_runs(snapshots, runs, sample, seed, block):
            for a in _starts(len(times)):
                kdt.append(params["K_ref"][r] * (times[a + 1:] - times[a]))
                rms.append(np.sqrt(((s[a + 1:] - s[a]) ** 2).mean(axis=(1, 2))))
        power, log_coef = np.polyfit(np.log(np.concatenate(kdt)), np.log(np.concatenate(rms)), 1)
        return cls(float(np.exp(log_coef)), float(power))

    def typical_change(self, K_ref, dT):
        """Typical RMS height change (m) after ``dT`` years at erodibility ``K_ref``."""
        return self.coef * (K_ref * dT) ** self.power

    def to_model(self, dz, K_ref, dT):
        """A height change ``dz`` (m) to model units: divided by ``typical_change``."""
        return dz / self.typical_change(K_ref, dT)

    def from_model(self, y, K_ref, dT):
        """A change in model units back to m: multiplied by ``typical_change``."""
        return y * self.typical_change(K_ref, dT)


@dataclass(frozen=True)
class Scaling:
    """The two fitted conversions of one model, saved in its checkpoint."""

    parameters: ParameterScaling
    change: ChangeScaling

    @classmethod
    def fit(cls, snapshots, params: dict, runs, times, block: int = 1) -> "Scaling":
        return cls(ParameterScaling.fit(params, runs, times), ChangeScaling.fit(snapshots, params, runs, times,
                                                                                 block=block))

    @classmethod
    def from_dict(cls, d: dict) -> "Scaling":
        p = d["parameters"]
        return cls(ParameterScaling(tuple(p["low"]), tuple(p["high"])), ChangeScaling(**d["change"]))


def change_rms(snapshots: np.ndarray, params: dict, runs, times, change: ChangeScaling, sample: int = 400,
               seed: int = 0, block: int = 1) -> float:
    """Root mean square of the training target, ``change.to_model(z_b - z_a, K_ref, dT)``, over the same
    pairs ``ChangeScaling.fit`` uses. The diffusion model's ``sigma_data``. Near 1 when the fit is good."""
    squares = []
    for r, s in _sample_runs(snapshots, runs, sample, seed, block):
        for a in _starts(len(times)):
            y = change.to_model(s[a + 1:] - s[a], params["K_ref"][r], (times[a + 1:] - times[a])[:, None, None])
            squares.append((y ** 2).mean(axis=(1, 2)))
    return float(np.sqrt(np.concatenate(squares).mean()))


def _sample_runs(snapshots, runs, sample: int, seed: int, block: int):
    """``(run, snapshots of that run)`` for a random sample of ``runs``, in float64, as block means if ``block`` > 1."""
    runs = np.asarray(runs)
    chosen = np.sort(np.random.default_rng(seed).choice(runs, min(sample, len(runs)), replace=False))
    for r in chosen:
        s = np.asarray(snapshots[r], dtype=np.float64)
        if block > 1:
            count, h, w = s.shape
            s = s.reshape(count, h // block, block, w // block, block).mean(axis=(2, 4))
        yield r, s
