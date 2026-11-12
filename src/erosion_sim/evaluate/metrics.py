"""Accuracy metrics on the output of ``predict_pairs``. Errors are in meters."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter, ScalarFormatter

from ..hydrology import drainage_area, pits
from .pairs import FACTORS, model_methods, per_pair_skill


def error_metrics(pairs: dict, show: bool = True) -> dict:
    """Height error of each method against the simulator, over all pairs.

    - ``rmse``: root mean squared error of the end heightmap (m). Equal to the error of the change.
    - ``skill``: ``1 - MSE / MSE(no change)`` over all pairs together; ``median_skill`` per pair.
    - ``erosion_ratio``: mean absolute change of the method over that of the simulator. 1 means it moves
      as much material as the simulator; below 1, too little.
    """
    true_change = np.abs(pairs["z_b"] - pairs["z_a"]).mean()
    mse_zero = ((pairs["z_a"] - pairs["z_b"]) ** 2).mean()
    result = {}
    for method in pairs["methods"]:
        mse = ((pairs[method] - pairs["z_b"]) ** 2).mean()
        result[method] = {
            "rmse": float(np.sqrt(mse)),
            "skill": float(1 - mse / mse_zero),
            "median_skill": float(np.median(per_pair_skill(pairs, method))),
            "erosion_ratio": float(np.abs(pairs[method] - pairs["z_a"]).mean() / true_change),
        }
    if show:
        w = max(16, *map(len, pairs["methods"]))
        print(f"{len(pairs['z_a'])} pairs; simulator's mean absolute change {true_change:.3f} m\n")
        print(f"{'method':>{w}} {'RMSE (m)':>9} {'skill':>7} {'median skill':>13} {'erosion ratio':>14}")
        for method, r in result.items():
            print(f"{method:>{w}} {r['rmse']:9.3f} {r['skill']:7.3f} {r['median_skill']:13.3f} {r['erosion_ratio']:14.3f}")
    return result


def _dilate(mask: np.ndarray) -> np.ndarray:
    """Grow a boolean mask by one cell in all 8 directions."""
    padded = np.pad(mask, 1)
    rows, cols = mask.shape
    return np.any([padded[1 + di:1 + di + rows, 1 + dj:1 + dj + cols]
                   for di in (-1, 0, 1) for dj in (-1, 0, 1)], axis=0)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else 1.0


def hydrology_metrics(pairs: dict, quantile: float = 0.95, show: bool = True, example: int = 0) -> dict:
    """Compare the rivers of each method's terrain with the simulator's.

    Water is routed with ``hydrology.drainage_area`` (D8, no pit filling) on every end heightmap.
    Channels are cells whose drainage area is at least one fixed threshold: the ``quantile`` of the
    simulator's drainage area over all pairs (0.95 marks the top 5% of cells as channel).

    - ``iou``: intersection over union of the method's channels and the simulator's, per pair, cell by
      cell. A channel one cell off counts as a miss.
    - ``iou_1``: the same after growing both masks by one cell, so channels one cell off still match.
    - ``log_area_rmse``: RMSE of log10 drainage area over all cells.
    - ``pits``: interior cells with no lower neighbor, per terrain. The simulator has a few too.

    Per-pair values are averaged. With ``show``, also plots channels for pair ``example``.
    """
    methods = pairs["methods"]
    areas = {name: np.stack([drainage_area(z) for z in pairs[name]]) for name in ("z_b", *methods)}
    threshold = float(np.quantile(areas["z_b"], quantile))
    channels = {name: a >= threshold for name, a in areas.items()}
    sim = channels["z_b"]
    sim_wide = np.stack([_dilate(c) for c in sim])
    result = {"threshold": threshold, "simulator_pits": float(np.mean([pits(z).sum() for z in pairs["z_b"]]))}
    for method in methods:
        mine = channels[method]
        mine_wide = np.stack([_dilate(c) for c in mine])
        result[method] = {
            "iou": float(np.mean([_iou(a, b) for a, b in zip(mine, sim)])),
            "iou_1": float(np.mean([_iou(a, b) for a, b in zip(mine_wide, sim_wide)])),
            "log_area_rmse": float(np.sqrt(((np.log10(areas[method]) - np.log10(areas["z_b"])) ** 2).mean())),
            "pits": float(np.mean([pits(z).sum() for z in pairs[method]])),
        }
    if show:
        w = max(16, *map(len, methods))
        print(f"channel threshold: drainage area >= {threshold:.0f} cells (simulator's {quantile:.0%} quantile)")
        print(f"simulator: {result['simulator_pits']:.1f} pits per terrain\n")
        print(f"{'method':>{w}} {'channel IoU':>12} {'IoU, 1 cell':>12} {'log10 area RMSE':>16} {'pits':>7}")
        for method in methods:
            r = result[method]
            print(f"{method:>{w}} {r['iou']:12.3f} {r['iou_1']:12.3f} {r['log_area_rmse']:16.3f} {r['pits']:7.1f}")
        fig, axes = plt.subplots(1, len(methods), figsize=(5.5 * len(methods), 5.8), layout="constrained", squeeze=False)
        for ax, method in zip(axes[0], methods):
            image = np.ones((*sim[example].shape, 3))
            image[sim[example] & ~channels[method][example]] = (0.85, 0.2, 0.2)
            image[channels[method][example] & ~sim[example]] = (0.2, 0.4, 0.9)
            image[sim[example] & channels[method][example]] = (0.1, 0.1, 0.1)
            ax.imshow(image, origin="lower")
            ax.set_title(f"pair {example}, {method}", fontsize=9)
            ax.set_axis_off()
        fig.suptitle("channels: black both, red simulator only, blue method only", fontsize=10)
        plt.show()
    return result


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


def spectrum_comparison(pairs: dict, show: bool = True) -> dict:
    """Power spectrum of the change (end minus start), averaged over pairs, for the simulator and each method.

    Shows at which scales a method has too little change (smoothed away) or too much (noise). The ratio
    to the simulator is 1 where the method matches it. The change is zero at the fixed edges, so no
    window is needed.
    """
    result = {}
    for name in ("z_b", *model_methods(pairs)):
        spectra = [radial_power_spectrum(z - z_a)[1] for z, z_a in zip(pairs[name], pairs["z_a"])]
        result["simulator" if name == "z_b" else name] = np.mean(spectra, axis=0)
    result["wavelength"] = radial_power_spectrum(pairs["z_a"][0])[0]
    if show:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2), layout="constrained")
        for name in ("simulator", *model_methods(pairs)):
            ax1.loglog(result["wavelength"], result[name], label=name)
        for i, name in enumerate(model_methods(pairs), start=1):
            ax2.semilogx(result["wavelength"], result[name] / result["simulator"], color=f"C{i}", label=name)
        ax2.axhline(1, color="k", lw=0.8)
        for ax in (ax1, ax2):
            ax.set_xlabel("wavelength (cells); fine detail to the right")
            ax.invert_xaxis()
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.legend(fontsize=8)
        ax1.set_ylabel("power of the change")
        ax2.set_ylabel("power / simulator's power")
        ax2.set_ylim(0, 2)
        plt.show()
    return result


def error_by_condition(pairs: dict, bins: int = 6, show: bool = True) -> dict:
    """Per-pair skill of each method against each factor in ``FACTORS``, as medians in equal-count bins.

    Shows where a method fails instead of averaging it away. Returns {factor: {"edges", method: medians}}.
    Factors with one value (for example U_over_K when every run has no uplift) are skipped.
    """
    skills = {m: per_pair_skill(pairs, m) for m in model_methods(pairs)}
    result = {}
    for factor in FACTORS:
        values = pairs[factor]
        if np.ptp(values) == 0:
            continue
        edges = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)))
        which = np.clip(np.searchsorted(edges, values, side="right") - 1, 0, len(edges) - 2)
        result[factor] = {"edges": edges}
        for m, skill in skills.items():
            result[factor][m] = np.array([np.median(skill[which == b]) if (which == b).any() else np.nan
                                          for b in range(len(edges) - 1)])
    if show:
        low = min(s.min() for s in skills.values())
        fig, axes = plt.subplots(1, len(result), figsize=(3.4 * len(result), 3.4), layout="constrained", squeeze=False)
        for ax, (factor, r) in zip(axes[0], result.items()):
            centers = (r["edges"][:-1] + r["edges"][1:]) / 2
            for i, (m, skill) in enumerate(skills.items(), start=1):
                ax.scatter(pairs[factor], skill, s=3, alpha=0.2, color=f"C{i}")
                ax.plot(centers, r[m], "o-", color=f"C{i}", label=m)
            if factor == "K_ref":
                ax.set_xscale("log")
            ax.set_xlabel(factor)
            ax.set_ylim(max(-0.5, low - 0.05), 1.02)
        axes[0, 0].set_ylabel("skill per pair (lines: medians)")
        axes[0, 0].legend(fontsize=7)
        plt.show()
    return result


def sample_spread(pairs: dict, bins: int = 10, show: bool = True, example: int = 0) -> dict:
    """Where each diffusion model's samples disagree, and whether that is where the errors are.

    The spread is the per-pixel standard deviation across a diffusion model's samples (m). Interior
    pixels of all pairs are pooled and grouped into ``bins`` equal-count bins of spread. For each method,
    the mean absolute error in each bin shows whether errors are larger where the samples disagree.
    ``correlation`` is the Pearson correlation of spread and absolute error over all interior pixels.
    Returns {diffusion name: {"mean_spread", "edges", "centers", method: {"error", "correlation"}}}.
    """
    result = {}
    for name, spread in pairs["spread"].items():
        s = spread[:, 1:-1, 1:-1].ravel()
        edges = np.quantile(s, np.linspace(0, 1, bins + 1))
        which = np.clip(np.searchsorted(edges, s, side="right") - 1, 0, bins - 1)
        counts = np.bincount(which, minlength=bins)
        r = {"mean_spread": float(s.mean()), "edges": edges,
             "centers": np.bincount(which, s, minlength=bins) / counts}
        for m in model_methods(pairs):
            err = np.abs(pairs[m] - pairs["z_b"])[:, 1:-1, 1:-1].ravel()
            r[m] = {"error": np.bincount(which, err, minlength=bins) / counts,
                    "correlation": float(np.corrcoef(s, err)[0, 1])}
        result[name] = r
    if show:
        for name, r in result.items():
            w = max(16, *map(len, model_methods(pairs)))
            print(f"{name}: mean spread across samples {r['mean_spread']:.3f} m\n")
            print(f"{'method':>{w}} {'corr(spread, |error|)':>22}")
            for m in model_methods(pairs):
                print(f"{m:>{w}} {r[m]['correlation']:22.3f}")
            maps = [(pairs["spread"][name][example], f"{name}: spread across samples (m)")]
            maps += [(np.abs(pairs[m] - pairs["z_b"])[example], f"|error| of {m} (m)") for m in model_methods(pairs)]
            fig, axes = plt.subplots(1, 1 + len(maps), figsize=(4.6 * (1 + len(maps)), 4.2), layout="constrained")
            for i, m in enumerate(model_methods(pairs), start=1):
                axes[0].plot(r["centers"], r[m]["error"], "o-", color=f"C{i}", label=m)
            axes[0].plot(r["centers"], r["centers"], "k--", lw=0.8, label="error = spread")
            axes[0].set(xscale="log", yscale="log", xlabel=f"spread of {name} (m)", ylabel="mean |error| (m)",
                        title="error against spread, interior pixels")
            axes[0].legend(fontsize=7)
            limit = max(np.quantile(img, 0.99) for img, _ in maps)
            for ax, (img, title) in zip(axes[1:], maps):
                im = ax.imshow(img, origin="lower", cmap="magma", vmin=0, vmax=limit)
                ax.set_title(f"pair {example}, {title}", fontsize=8)
                ax.set_axis_off()
            plt.colorbar(im, ax=axes[1:], shrink=0.8)
            plt.show()
    return result


def error_by_elapsed(pairs: dict, bins: int = 10, quantile: float = 0.95, show: bool = True) -> dict:
    """Error of each method against the elapsed time of each pair, for models trained on many elapsed
    times.

    Pairs are grouped into ``bins`` equal-count bins of elapsed time. Per bin: RMSE (m), skill over the
    bin's pairs together, and mean channel IoU (channels as in ``hydrology_metrics``). Also returns,
    per method, the median skill per pair on a grid of start time against elapsed time; the plot shows
    the grid of the first method.
    """
    if len(np.unique(pairs["dT"])) < 2:
        if show:
            print(f"skipped: every pair has the same elapsed time ({pairs['dT'][0]:,.0f} yr); "
                  "this needs a model trained on many elapsed times")
        return {}
    methods = pairs["methods"]
    areas = {name: np.stack([drainage_area(z) for z in pairs[name]]) for name in ("z_b", *methods)}
    threshold = float(np.quantile(areas["z_b"], quantile))
    sim = areas["z_b"] >= threshold
    sq = {m: ((pairs[m] - pairs["z_b"]) ** 2).mean(axis=(1, 2)) for m in methods}
    iou = {m: np.array([_iou(a, b) for a, b in zip(areas[m] >= threshold, sim)]) for m in methods}
    dT = pairs["dT"]
    edges = np.unique(np.quantile(dT, np.linspace(0, 1, bins + 1)))
    which = np.clip(np.searchsorted(edges, dT, side="right") - 1, 0, len(edges) - 2)
    groups = [which == b for b in range(len(edges) - 1)]
    result = {"edges": edges, "centers": np.array([dT[g].mean() for g in groups])}
    for m in methods:
        result[m] = {
            "rmse": np.array([np.sqrt(sq[m][g].mean()) for g in groups]),
            "skill": np.array([1 - sq[m][g].sum() / sq["no change"][g].sum() for g in groups]),
            "iou": np.array([iou[m][g].mean() for g in groups]),
        }
    # Grid of start time against elapsed time. End times past the last snapshot leave cells empty.
    start_edges = np.linspace(pairs["start_time"].min(), pairs["start_time"].max() + 1, 6)
    dT_edges = np.linspace(dT.min(), dT.max() + 1, 6)
    result["grid"] = {"start_edges": start_edges, "dT_edges": dT_edges}
    for m in model_methods(pairs):
        skill = per_pair_skill(pairs, m)
        grid = np.full((5, 5), np.nan)
        for i in range(5):
            for j in range(5):
                cell = ((pairs["start_time"] >= start_edges[i]) & (pairs["start_time"] < start_edges[i + 1])
                        & (dT >= dT_edges[j]) & (dT < dT_edges[j + 1]))
                if cell.any():
                    grid[i, j] = np.median(skill[cell])
        result["grid"][m] = grid
    if show:
        fig, axes = plt.subplots(1, 4, figsize=(19, 3.8), layout="constrained")
        for i, m in enumerate(methods):
            style = "s--" if m == "no change" else "o-"
            axes[0].plot(result["centers"], result[m]["rmse"], style, color=f"C{i}", label=m)
            axes[2].plot(result["centers"], result[m]["iou"], style, color=f"C{i}", label=m)
            if m != "no change":
                axes[1].plot(result["centers"], result[m]["skill"], "o-", color=f"C{i}", label=m)
        for ax, label in zip(axes[:3], ("RMSE (m)", "skill", f"channel IoU (top {1 - quantile:.0%} of cells)")):
            ax.set(xlabel="elapsed time (yr)", ylabel=label)
            ax.legend(fontsize=7)
        first = model_methods(pairs)[0]
        grid = result["grid"][first]
        im = axes[3].imshow(grid, origin="lower", aspect="auto", cmap="viridis",
                            extent=(dT_edges[0], dT_edges[-1], start_edges[0], start_edges[-1]))
        axes[3].set(xlabel="elapsed time (yr)", ylabel="start time (yr)", title=f"median skill per pair, {first}")
        for i in range(5):
            for j in range(5):
                if np.isfinite(grid[i, j]):
                    axes[3].text((dT_edges[j] + dT_edges[j + 1]) / 2, (start_edges[i] + start_edges[i + 1]) / 2,
                                 f"{grid[i, j]:.3f}", ha="center", va="center", fontsize=8, color="w")
        plt.colorbar(im, ax=axes[3])
        fig.suptitle(f"{len(dT)} pairs, {len(edges) - 1} bins of elapsed time")
        plt.show()
    return result


def fine_detail(spectrum: dict, name: str, wavelength: float = 4.0) -> float:
    """Mean of the method's power over the simulator's, at wavelengths up to ``wavelength`` cells."""
    fine = spectrum["wavelength"] <= wavelength
    return float(np.mean(spectrum[name][fine] / spectrum["simulator"][fine]))
