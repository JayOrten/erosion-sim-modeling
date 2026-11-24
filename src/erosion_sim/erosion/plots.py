"""Figures and tables from an erosion results folder (``suite.py``). Nothing here recomputes."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LightSource

from ..evaluation.results import Results

TABLE = ("rmse", "skill", "median_skill", "erosion_ratio", "iou", "iou_1", "log_area_rmse", "pits_per_km2",
         "slope_area_error", "fine_detail")


def entries(results: Results, kinds: tuple[str, ...] | None = None) -> list[str]:
    """Scored entries in order, optionally only those of the given kinds (baseline, model, ...)."""
    return [e for e, info in results.manifest["entries"].items() if kinds is None or info["kind"] in kinds]


def table(results: Results, columns: tuple[str, ...] = TABLE) -> None:
    """Print every entry's summary over all pairs."""
    rows = results.scalars["entries"]
    width = max(14, *map(len, rows))
    print(f"{results.manifest['test']['pairs']} pairs from {results.manifest['test']['runs']} runs of "
          f"{results.manifest['test']['data_profile']}/{results.manifest['test']['split']}")
    widths = [max(11, len(c) + 2) for c in columns]
    print(f"{'entry':<{width}}" + "".join(f"{c:>{w}}" for c, w in zip(columns, widths)))
    for entry, row in rows.items():
        print(f"{entry:<{width}}" + "".join(f"{row.get(c, float('nan')):>{w}.4g}" for c, w in zip(columns, widths)))


def by_table(results: Results, factor: str, metric: str = "skill", kinds=("baseline", "model")) -> None:
    """Print ``metric`` per group of ``factor`` (columns) and entry (rows)."""
    groups = results.scalars["by"].get(factor)
    if not groups:
        print(f"no breakdown by {factor}")
        return
    names = entries(results, kinds)
    width = max(14, *map(len, names))
    print(f"{metric} by {factor}")
    print(f"{'entry':<{width}}" + "".join(f"{g:>22}" for g in groups))
    for entry in names:
        print(f"{entry:<{width}}" + "".join(f"{groups[g][entry].get(metric, float('nan')):>22.4g}" for g in groups))


def plot_by(results: Results, factor: str, metric: str = "skill", kinds=("baseline", "model"), show: bool = True):
    """``metric`` per group of ``factor``, one line per entry."""
    groups = results.scalars["by"].get(factor, {})
    fig, ax = plt.subplots(figsize=(max(5, 0.6 * len(groups)), 3.6), layout="constrained")
    for entry in entries(results, kinds):
        ax.plot(range(len(groups)), [groups[g][entry].get(metric, np.nan) for g in groups], "o-", label=entry)
    ax.set_xticks(range(len(groups)), list(groups), rotation=45, ha="right", fontsize=8)
    ax.set(xlabel=factor, ylabel=metric)
    ax.legend(fontsize=7, frameon=False)
    if show:
        plt.show()
    return fig


def plot_spectrum(results: Results, kinds=("baseline", "model", "postprocessed"), show: bool = True):
    """Each entry's change spectrum over the simulator's: 1 where it has the simulator's detail at that
    wavelength, below 1 where it is smoother, above where it is noisier."""
    a = results.arrays
    fig, ax = plt.subplots(figsize=(6, 3.8), layout="constrained")
    for entry in entries(results, kinds):
        if entry != "no change":
            ax.semilogx(a["spectrum/wavelength"], a[f"spectrum/{entry}"] / a["spectrum/simulator"], label=entry)
    ax.axhline(1, color="black", lw=0.8)
    ax.invert_xaxis()
    ax.set(xlabel="wavelength (cells); fine detail to the right", ylabel="power / simulator's", ylim=(0, 2))
    ax.legend(fontsize=7, frameon=False)
    if show:
        plt.show()
    return fig


def plot_examples(results: Results, kinds=("baseline", "model"), show: bool = True):
    """For each example pair: the start terrain, then the change of the simulator and of each entry."""
    ex = results.examples
    names = [e for e in entries(results, kinds) if e != "no change"]
    count = len(ex["z_a"])
    fig, axes = plt.subplots(count, 2 + len(names), figsize=(2.6 * (2 + len(names)), 2.6 * count), squeeze=False,
                             layout="constrained")
    for i in range(count):
        z_a = ex["z_a"][i]
        change = ex["z_b"][i] - z_a
        limit = np.abs(change).max() or 1.0
        panels = [("start", hillshade(z_a), None), ("simulator", change, limit)]
        panels += [(n, ex[n][i] - z_a, limit) for n in names]
        for ax, (title, image, lim) in zip(axes[i], panels):
            if lim is None:
                ax.imshow(image, cmap="gray")
            else:
                ax.imshow(image, cmap="RdBu", vmin=-lim, vmax=lim)
            ax.set_title(title, fontsize=8)
            ax.set_axis_off()
    if show:
        plt.show()
    return fig


def plot_steps(results: Results, metrics=("rmse", "iou", "fine_detail"), show: bool = True):
    """A step sweep (experiment B5): each metric against network calls per sample, one line per model."""
    info = results.manifest["entries"]
    models = {}
    for entry, i in info.items():
        if i["kind"] == "model":
            models.setdefault(i["model"], []).append((i["network_calls"], entry))
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.6 * len(metrics), 3.3), layout="constrained", squeeze=False)
    for ax, metric in zip(axes[0], metrics):
        for model, rows in models.items():
            rows = sorted(rows)
            ax.plot([c for c, _ in rows], [results.scalars["entries"][e][metric] for _, e in rows], "o-", label=model)
        ax.set(xscale="log", xlabel="network calls per sample", title=metric)
    axes[0, 0].legend(fontsize=7, frameon=False)
    if show:
        plt.show()
    return fig


def hillshade(z: np.ndarray, z_range: tuple[float, float] | None = None) -> np.ndarray:
    """RGB image of a heightmap lit from the upper left, colored by height over ``z_range`` (or its own range)."""
    low, high = z_range if z_range is not None else (z.min(), z.max())
    light = LightSource(azdeg=315, altdeg=45)
    return light.shade(np.asarray(z, dtype=float), cmap=plt.cm.terrain, blend_mode="soft", vmin=low, vmax=high)
