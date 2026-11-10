"""Metrics against the number of diffusion sampler steps."""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .metrics import error_metrics, fine_detail, hydrology_metrics, spectrum_comparison
from .pairs import predict_pairs


def step_sweep(checkpoint: Path, steps=(8, 16, 32, 64), regression: Path | None = None, split: str = "val",
               count: int = 256, samples: int = 4, seed: int = 0, show: bool = True,
               data_root: Path | None = None) -> dict:
    """Metrics of a diffusion checkpoint against the number of sampler steps.

    Every step count sees the same ``count`` pairs and the same noise, so only the step count differs.
    Per step count, for one sample and for the mean of ``samples``: RMSE (m), channel IoU, pits per
    terrain, and fine detail (power of the change at wavelengths up to 4 cells, over the simulator's;
    1 is right). Also the time per sample (ms), from the wall time of the whole prediction. A
    ``regression`` checkpoint adds its numbers as a reference.
    """
    common = dict(split=split, count=count, seed=seed, progress=False, data_root=data_root)
    predict_pairs(checkpoint, **{**common, "count": 1}, samples=1, steps=2)  # warm up the GPU
    result = {"steps": np.array(steps)}
    names = {"1 sample": "model (1 sample)", f"mean of {samples}": f"model (mean of {samples})"}
    rows = {label: {"rmse": [], "iou": [], "pits": [], "fine": []} for label in names}
    times = []
    for n in steps:
        start = time.perf_counter()
        pairs = predict_pairs(checkpoint, **common, samples=samples, steps=n)
        times.append(1000 * (time.perf_counter() - start) / (len(pairs["z_b"]) * samples))
        errors = error_metrics(pairs, show=False)
        hydro = hydrology_metrics(pairs, show=False)
        spectrum = spectrum_comparison(pairs, show=False)
        for label, method in names.items():
            rows[label]["rmse"].append(errors[method]["rmse"])
            rows[label]["iou"].append(hydro[method]["iou"])
            rows[label]["pits"].append(hydro[method]["pits"])
            rows[label]["fine"].append(fine_detail(spectrum, method))
    for label in rows:
        result[label] = {key: np.array(value) for key, value in rows[label].items()}
    result["ms_per_sample"] = np.array(times)
    result["simulator_pits"] = hydro["simulator_pits"]
    result["no change"] = {"rmse": errors["no change"]["rmse"], "iou": hydro["no change"]["iou"]}
    if regression is not None:
        pairs = predict_pairs(regression, **common)
        errors, hydro = error_metrics(pairs, show=False), hydrology_metrics(pairs, show=False)
        result["regression"] = {"rmse": errors["model"]["rmse"], "iou": hydro["model"]["iou"],
                                "pits": hydro["model"]["pits"],
                                "fine": fine_detail(spectrum_comparison(pairs, show=False), "model")}
    if show:
        print(f"{len(pairs['z_b'])} pairs, {samples} samples each; simulator: {result['simulator_pits']:.1f} pits")
        for label in rows:
            r = result[label]
            print(f"\ndiffusion, {label}")
            print(f"{'steps':>6} {'ms/sample':>10} {'RMSE (m)':>9} {'IoU':>6} {'pits':>6} {'fine':>5}")
            for i, n in enumerate(steps):
                print(f"{n:>6} {times[i]:>10.1f} {r['rmse'][i]:>9.3f} {r['iou'][i]:>6.3f} {r['pits'][i]:>6.1f}"
                      f" {r['fine'][i]:>5.2f}")
        if regression is not None:
            r = result["regression"]
            print(f"\nregression: RMSE {r['rmse']:.3f}, IoU {r['iou']:.3f}, pits {r['pits']:.1f}, fine {r['fine']:.2f}")
        panels = [("rmse", "RMSE (m)"), ("iou", "channel IoU"), ("pits", "pits per terrain"),
                  ("fine", "fine detail (1 = simulator)")]
        fig, axes = plt.subplots(1, 5, figsize=(20, 3.8), layout="constrained")
        for ax, (key, ylabel) in zip(axes, panels):
            for i, label in enumerate(rows):
                ax.plot(steps, result[label][key], "o-", color=f"C{i + 1}", label=f"diffusion, {label}")
            if regression is not None:
                ax.axhline(result["regression"][key], color="C0", ls="--", label="regression")
            if key == "pits":
                ax.axhline(result["simulator_pits"], color="k", ls=":", label="simulator")
            if key == "fine":
                ax.axhline(1, color="k", ls=":", label="simulator")
            if key == "iou":
                ax.axhline(result["no change"]["iou"], color="gray", ls=":", label="no change")
            ax.set_ylabel(ylabel)
        axes[4].plot(steps, times, "o-", color="k")
        axes[4].set_ylabel("ms per sample")
        for ax in axes:
            ax.set_xscale("log", base=2)
            ax.set_xticks(steps, [str(n) for n in steps])
            ax.set_xlabel("sampler steps")
        axes[0].legend(fontsize=8)
        fig.suptitle(f"{Path(checkpoint).stem}: metrics against sampler steps")
        plt.show()
    return result
