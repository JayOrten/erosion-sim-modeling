"""Terrain metrics, as plain functions on arrays. Heights are in m.

Each metric is computed once per pair (``pair_metrics``), so the overall numbers and every breakdown
(by noise setup, parameter, elapsed time) are aggregations of the same per-pair arrays (``summarize``).

- Height error: squared error of the end heights, against the no-change baseline's.
- Erosion ratio: how much material a method moves, over the simulator's.
- Hydrology: channels are the cells with the top 5% of the simulator's drainage area in that pair
  (``CHANNEL_QUANTILE``). Channel IoU (intersection over union), with and without one cell of slack;
  RMSE of log10 drainage area; pits per km².
- Slope-area exponent: ``hydrology.slope_area_exponent`` on the method's terrain. In steady channels the
  stream power law with n = 1 predicts it equals ``m`` (Flint's law), but 50,000 yr runs are not steady:
  on the simulator's own output the fit ranges from about -0.4 to 0.8 for m from 0.3 to 0.6. So the
  error is against the simulator's exponent on the same pair, not against ``m``.
- Spectrum: radial power spectrum of the change, for detail at each wavelength.
"""

import numpy as np

from ..data.hydrology import drainage_area, pits, slope_area_exponent

CHANNEL_QUANTILE = 0.95


def _dilate(mask: np.ndarray) -> np.ndarray:
    """Grow a boolean mask by one cell in all 8 directions."""
    padded = np.pad(mask, 1)
    rows, cols = mask.shape
    return np.any([padded[1 + di:1 + di + rows, 1 + dj:1 + dj + cols] for di in (-1, 0, 1) for dj in (-1, 0, 1)],
                  axis=0)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else 1.0


def radial_power_spectrum(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Radially averaged power spectrum of a square (H, W) field. Returns (wavelength in cells, power)."""
    size = field.shape[0]
    power = np.abs(np.fft.fftshift(np.fft.fft2(field))) ** 2
    ii, jj = np.indices(field.shape) - size // 2
    radius = np.rint(np.hypot(ii, jj)).astype(int)
    total = np.bincount(radius.ravel(), power.ravel())
    counts = np.bincount(radius.ravel())
    k = np.arange(1, size // 2)
    return size / k, total[k] / counts[k]


class Truth:
    """What every method of one pair is compared with: the simulator's start and end heights, the end's
    drainage area, channels, and slope-area exponent, and the run's ``m``. Computed once per pair."""

    def __init__(self, z_a: np.ndarray, z_b: np.ndarray, m: float, cell: float = 1.0):
        self.z_a, self.z_b, self.m, self.cell = z_a, z_b, float(m), cell
        self.area = drainage_area(z_b, cell)
        self.threshold = float(np.quantile(self.area, CHANNEL_QUANTILE))
        self.channels = self.area >= self.threshold
        self.channels_wide = _dilate(self.channels)
        self.change_spectrum = radial_power_spectrum(z_b - z_a)[1]
        self.exponent = slope_area_exponent(z_b, self.threshold, cell)


PAIR_METRICS = ("sq_error", "sq_error_zero", "abs_change", "abs_change_true", "iou", "iou_1", "log_area_rmse",
                "pits_per_km2", "slope_area_exponent", "slope_area_error")


def pair_metrics(z: np.ndarray, truth: Truth) -> dict[str, float]:
    """Every per-pair metric of end heights ``z`` against ``truth`` (see ``PAIR_METRICS``)."""
    area = drainage_area(z, truth.cell)
    channels = area >= truth.threshold
    exponent = slope_area_exponent(z, truth.threshold, truth.cell)
    km2 = z.size * truth.cell**2 / 1e6
    return {
        "sq_error": float(np.mean((z - truth.z_b) ** 2)),
        "sq_error_zero": float(np.mean((truth.z_a - truth.z_b) ** 2)),
        "abs_change": float(np.mean(np.abs(z - truth.z_a))),
        "abs_change_true": float(np.mean(np.abs(truth.z_b - truth.z_a))),
        "iou": _iou(channels, truth.channels),
        "iou_1": _iou(_dilate(channels), truth.channels_wide),
        "log_area_rmse": float(np.sqrt(np.mean((np.log10(area) - np.log10(truth.area)) ** 2))),
        "pits_per_km2": float(pits(z).sum() / km2),
        "slope_area_exponent": exponent,
        "slope_area_error": abs(exponent - truth.exponent),
    }


def _finite_mean(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else float("nan")


def summarize(per_pair: dict[str, np.ndarray], rows: np.ndarray | None = None) -> dict[str, float]:
    """Aggregate per-pair metrics (one array per name in ``PAIR_METRICS``) over ``rows`` (default all):

    - ``rmse`` (m): root of the mean squared error. ``skill``: ``1 - MSE / MSE(no change)`` over the rows
      together. ``median_skill``: the median of the per-pair skills.
    - ``erosion_ratio``: the method's mean absolute change over the simulator's.
    - ``iou``, ``iou_1``, ``log_area_rmse``, ``pits_per_km2``, ``slope_area_exponent``, ``slope_area_error``:
      means over the rows (NaN exponents, where too few channel cells, are left out).
    """
    p = {k: v if rows is None else v[rows] for k, v in per_pair.items()}
    if not len(p["sq_error"]):
        return {}
    per_skill = 1 - p["sq_error"] / np.maximum(p["sq_error_zero"], 1e-30)
    return {
        "pairs": int(len(p["sq_error"])),
        "rmse": float(np.sqrt(p["sq_error"].mean())),
        "skill": float(1 - p["sq_error"].sum() / max(p["sq_error_zero"].sum(), 1e-30)),
        "median_skill": float(np.median(per_skill)),
        "erosion_ratio": float(p["abs_change"].mean() / max(p["abs_change_true"].mean(), 1e-30)),
        **{k: _finite_mean(p[k]) for k in ("iou", "iou_1", "log_area_rmse", "pits_per_km2", "slope_area_exponent",
                                           "slope_area_error")},
    }


def fine_detail(wavelength: np.ndarray, method: np.ndarray, simulator: np.ndarray, cutoff: float = 4.0) -> float:
    """Mean ratio of the method's change power to the simulator's at wavelengths up to ``cutoff`` cells.
    1 is matching detail; below 1 is smoothed away; above 1 is noise."""
    fine = wavelength <= cutoff
    return float(np.mean(method[fine] / np.maximum(simulator[fine], 1e-30)))


def spread_correlation(spread: np.ndarray, error: np.ndarray) -> float:
    """Pearson correlation of per-cell sample spread with absolute error, over interior cells. High when
    a sampled model is unsure where it is wrong."""
    s, e = spread[..., 1:-1, 1:-1].ravel(), error[..., 1:-1, 1:-1].ravel()
    if s.std() == 0 or e.std() == 0:
        return float("nan")
    return float(np.corrcoef(s, e)[0, 1])
