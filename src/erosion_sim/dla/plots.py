"""Figures from a DLA results folder (``suite.py``). Nothing here recomputes; it only reads ``Results``."""

import matplotlib.pyplot as plt
import numpy as np

from ..experiments.results import Results

METRIC_LABELS = {
    "particles": "particles",
    "connected_fraction": "fraction joined to the seed",
    "start_kept": "fraction of start kept",
    "gyration_radius": "radius of gyration (cells)",
    "fractal_dimension": "mass dimension",
}


def _finite_mean(values) -> float:
    """Mean of the finite values, NaN if there are none (a metric undefined for every cluster)."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else float("nan")


def models(results: Results) -> list[str]:
    """The scored entries: model labels, or ``<label>@<steps>`` in a step sweep."""
    return list(results.manifest.get("entries", results.manifest["models"]))


def plot_metric(results: Results, metric: str, show: bool = True):
    """Mean ± standard deviation of ``metric`` per test condition, each model beside the simulator."""
    conds = results.manifest["conditions"]
    labels = models(results)
    fig, ax = plt.subplots(figsize=(max(6, 0.45 * len(conds)), 3.8), layout="constrained")
    width = 0.8 / (len(labels) + 1)
    x = np.arange(len(conds))
    first = results.scalars[labels[0]]
    ax.errorbar(x - 0.4 + width / 2, [first[c][metric]["simulator_mean"] for c in conds],
                [first[c][metric]["simulator_std"] for c in conds], fmt="o", color="black", label="simulator")
    for k, label in enumerate(labels, start=1):
        rows = results.scalars[label]
        ax.errorbar(x - 0.4 + width * (k + 0.5), [rows[c][metric]["model_mean"] for c in conds],
                    [rows[c][metric]["model_std"] for c in conds], fmt="o", label=label)
    ax.set_xticks(x, conds, rotation=60, ha="right")
    ax.set(ylabel=METRIC_LABELS.get(metric, metric), title=metric)
    ax.legend(frameon=False)
    if show:
        plt.show()
    return fig


def plot_examples(results: Results, conditions: list[str] | None = None, count: int = 3, panel: float = 3.0,
                  show: bool = True):
    """For each condition (default: every fixed-``p`` one): the start, then ``count`` simulator clusters,
    then ``count`` clusters from each model. ``panel`` is each image's width in inches."""
    conds = conditions or [c for c in results.manifest["conditions"] if c.startswith("p=")]
    sources = ["simulator", *models(results)]
    cols = 1 + count * len(sources)
    fig, axes = plt.subplots(len(conds), cols, figsize=(panel * cols, (panel + 0.1) * len(conds)), squeeze=False,
                             layout="constrained")
    for r, cond in enumerate(conds):
        panels = [("start", results.examples[f"start/{cond}"])]
        for source in sources:
            clusters = results.examples[f"{source}/{cond}"]
            panels += [(source, clusters[i]) for i in range(min(count, len(clusters)))]
        for c, ax in enumerate(axes[r]):
            ax.set_xticks([])
            ax.set_yticks([])
            if c < len(panels):
                name, image = panels[c]
                ax.imshow(image, cmap="gray_r", interpolation="nearest")
                if r == 0:
                    ax.set_title(name, fontsize=11)
        axes[r, 0].set_ylabel(cond, fontsize=11)
    if show:
        plt.show()
    return fig


def plot_radial_density(results: Results, conditions: list[str] | None = None, show: bool = True):
    """Mean occupancy against distance from the seed, one panel per condition."""
    conds = conditions or [c for c in results.manifest["conditions"] if c.startswith("p=")]
    edges = results.arrays["radial_edges"]
    centers = 0.5 * (edges[1:] + edges[:-1])
    fig, axes = plt.subplots(1, len(conds), figsize=(3 * len(conds), 2.8), squeeze=False, layout="constrained",
                             sharey=True)
    for ax, cond in zip(axes[0], conds):
        ax.plot(centers, results.arrays[f"simulator/{cond}/radial_density"], color="black", label="simulator")
        for label in models(results):
            ax.plot(centers, results.arrays[f"{label}/{cond}/radial_density"], label=label)
        ax.set(title=cond, xlabel="distance from seed (cells)")
    axes[0, 0].set_ylabel("occupancy")
    axes[0, 0].legend(frameon=False, fontsize=8)
    if show:
        plt.show()
    return fig


def table(results: Results, metrics: list[str] | None = None) -> None:
    """Print each model's mean ± standard deviation per condition and metric, beside the simulator's."""
    names = metrics or list(METRIC_LABELS)
    for label in models(results):
        print(f"== {label}  (model mean ± std | simulator mean ± std)")
        print(f"{'condition':<12}" + "".join(f"{m:>34}" for m in names))
        for cond in results.manifest["conditions"]:
            row = results.scalars[label][cond]
            cells = [f"{row[m]['model_mean']:7.3f} ± {row[m]['model_std']:5.3f} | "
                     f"{row[m]['simulator_mean']:7.3f} ± {row[m]['simulator_std']:5.3f}" for m in names]
            print(f"{cond:<12}" + "".join(f"{c:>34}" for c in cells))
        print(f"seconds per sample: {results.scalars[label]['seconds_per_sample']:.4f}")


def plot_containment(results: Results, show: bool = True):
    """Agreement results: the mean fraction of the smaller sample's cells missing from the larger one, for
    every pair of jumps. One panel per model and start group. 0 everywhere is perfect agreement."""
    jumps = results.manifest["jumps"]
    panels = [(label, group) for label in models(results) for group in results.scalars[label]["containment"]]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 3.2), squeeze=False, layout="constrained")
    for ax, (label, group) in zip(axes[0], panels):
        grid = np.full((len(jumps), len(jumps)), np.nan)
        for key, value in results.scalars[label]["containment"][group].items():
            x1, x2 = (int(v) for v in key.split("<"))
            grid[jumps.index(x1), jumps.index(x2)] = value
        image = ax.imshow(grid, vmin=0, cmap="magma_r")
        ax.set_xticks(range(len(jumps)), jumps)
        ax.set_yticks(range(len(jumps)), jumps)
        ax.set(xlabel="x2", ylabel="x1", title=f"{label}, {group} starts")
        fig.colorbar(image, ax=ax, shrink=0.8)
    if show:
        plt.show()
    return fig


def plot_chain(results: Results, metric: str, show: bool = True):
    """Agreement results: ``metric`` for each chaining scheme (1, 2, 4 jumps) beside the simulator's."""
    labels = models(results)
    schemes = list(results.scalars[labels[0]]["chain"])
    fig, ax = plt.subplots(figsize=(5, 3.5), layout="constrained")
    ref = results.scalars[labels[0]]["chain"][schemes[0]][metric]
    ax.axhspan(ref["simulator_mean"] - ref["simulator_std"], ref["simulator_mean"] + ref["simulator_std"],
               color="0.85", label="simulator ± std")
    ax.axhline(ref["simulator_mean"], color="black", lw=1)
    for k, label in enumerate(labels):
        rows = results.scalars[label]["chain"]
        ax.errorbar(np.arange(len(schemes)) + 0.1 * k, [rows[s][metric]["model_mean"] for s in schemes],
                    [rows[s][metric]["model_std"] for s in schemes], fmt="o", label=label)
    ax.set_xticks(range(len(schemes)), schemes)
    ax.set(xlabel="jumps × particles per jump", ylabel=METRIC_LABELS.get(metric, metric), title=metric)
    ax.legend(frameon=False)
    if show:
        plt.show()
    return fig


def plot_steps(results: Results, metrics: list[str] | None = None, show: bool = True):
    """Step sweep results (A4): each metric's distance to the simulator (1-Wasserstein, averaged over the
    test conditions) against network calls per sample, one line per model, and seconds per sample."""
    names = metrics or list(METRIC_LABELS)
    entries = results.manifest["entries"]
    by_model = {}
    for entry, info in entries.items():
        by_model.setdefault(info["model"], []).append((info["network_calls"], entry))
    conds = results.manifest["conditions"]
    fig, axes = plt.subplots(1, len(names) + 1, figsize=(3.3 * (len(names) + 1), 3.2), layout="constrained")
    for ax, metric in zip(axes, names):
        for model, rows in by_model.items():
            rows = sorted(rows)
            distance = [_finite_mean([results.scalars[e][c][metric]["wasserstein"] for c in conds]) for _, e in rows]
            ax.plot([calls for calls, _ in rows], distance, "o-", label=model)
        ax.set(xscale="log", xlabel="network calls per sample", title=metric)
    axes[0].set_ylabel("distance to simulator")
    axes[0].legend(frameon=False, fontsize=8)
    for model, rows in by_model.items():
        rows = sorted(rows)
        axes[-1].plot([c for c, _ in rows], [results.scalars[e]["seconds_per_sample"] for _, e in rows], "o-",
                      label=model)
    axes[-1].set(xscale="log", yscale="log", xlabel="network calls per sample", title="seconds per sample")
    if show:
        plt.show()
    return fig

