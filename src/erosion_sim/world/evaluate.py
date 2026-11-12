"""Evaluate the world models and the world machinery (infinite-plan step 10).

- ``level_pairs``: each level alone against the simulator, on validation pairs. The output has the
  ``pairs`` format of ``erosion_sim.evaluate``, so its metrics (``error_metrics``,
  ``spectrum_comparison``, ``hydrology_metrics``) apply directly.
- ``window_disagreement``: how much neighboring windows disagree where they overlap, before blending.
- ``edge_profile``: a level's change against distance from its own window edge.
- ``order_invariance``: whether query order and clearing the cache change results.
- ``chunked_vs_whole``: whether a region asked for chunk by chunk, each from a new world, equals one query.
- ``query_times``: time of first and repeat queries.
- ``simulator_times``: the simulator's time on the same terrain.
- ``speed_comparison``: the world's times against the simulator's.
- ``run_world``: a world on a dataset sample's terrain and erosion parameters. ``world.compare`` grades
  worlds against such samples.
- Plots: ``headline``, ``show_examples``, ``by_elapsed``, ``seam_summary``.

Functions return their results and print a summary unless ``show=False``.
"""

import time
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm

from ..data import Split
from ..diffusion import DEFAULT_STEPS, initial_noise, sample
from ..evaluate.metrics import error_metrics, hydrology_metrics
from ..evaluate.pairs import add_method, make_pairs, subset_pairs
from ..evaluate.speed import time_simulator
from ..render import hillshade
from .data import ErosionCoarse, ErosionCrops, block_mean, upsample
from .level import Level
from .world import World


def smooth_upsample(z: np.ndarray, block: int) -> np.ndarray:
    """Bilinear upsampling of coarse cells (..., h, w) to (..., h * block, w * block), by cell centers.

    The baselines without fine detail use this rather than repeating each value, which would add
    block edges that the simulator does not have.
    """
    t = torch.as_tensor(np.asarray(z, dtype=np.float64)).reshape(-1, 1, *z.shape[-2:])
    out = F.interpolate(t, scale_factor=block, mode="bilinear", align_corners=False)
    return out.reshape(*z.shape[:-2], z.shape[-2] * block, z.shape[-1] * block).numpy()


def _sample(model, x: torch.Tensor, cond: torch.Tensor, keys: list, steps: int, batch_size: int) -> torch.Tensor:
    """One diffusion sample per row of ``x``, from the noise for ``keys[i]``, on the model's device."""
    device = next(model.parameters()).device
    out = []
    for i in range(0, len(x), batch_size):
        rows = slice(i, i + batch_size)
        noise = initial_noise(keys[rows], x.shape[-2:], fixed_edges=model.fixed_edges)
        out.append(sample(model, x[rows].to(device), cond[rows].to(device), noise, steps).float().cpu())
    return torch.cat(out)


def level_pairs(coarse: tuple, fine: tuple, split: str = "val", count: int = 256, seed: int = 0,
                steps: int = DEFAULT_STEPS, batch_size: int = 64, data_root=None, show: bool = True) -> dict:
    """Each level against the simulator on ``count`` random pairs of ``split`` (no flips or rotations).

    ``coarse`` and ``fine`` are ``(model, scaling, config)`` from ``load_checkpoint``. Returns
    ``{"coarse": pairs, "fine": pairs}`` in the ``pairs`` format of ``erosion_sim.evaluate``, in meters:

    - coarse pairs (whole runs at coarse cells; for levels trained on interior crops, the run's interior
      ``margin`` cells from its edge, sampled as one window): ``"coarse model"``.
    - fine pairs (crops at 1 m). The fine model reads a coarse map of the change (4 m cells), which
      comes either from the simulator (its true change, block-averaged; only known in testing, and what
      training used) or from the coarse model (what the world uses):

      - ``"fine model, simulator's coarse"``: the fine model given the simulator's coarse map. The fine
        model alone, at its best.
      - ``"coarse + fine models"``: the coarse model's map, then the fine model. The world's pipeline.
      - ``"simulator's coarse only"``: the simulator's coarse map, smoothly upsampled to 1 m, no fine
        model.
      - ``"coarse model only"``: the coarse model's map, smoothly upsampled to 1 m, no fine model. The
        world without its fine level.

    Item ``i`` of both levels is the same (run, start, end) pair, so the coarse model's sample for a run
    is the one the fine model reads. Noise for pair ``p`` is keyed ``(seed, "coarse" or "fine", p)``.
    """
    (cm, cn, ccfg), (fm, fn, fcfg) = coarse, fine
    if (ccfg.runs, ccfg.lags, ccfg.block) != (fcfg.runs, fcfg.lags, fcfg.block):
        raise ValueError("the two levels must share runs, lags, and block, so their items are the same pairs")
    data = Split.open(fcfg.data_profile, split, data_root)
    params = data.params
    runs = data.select(**dict(fcfg.runs))
    block = fcfg.block
    # Levels trained away from the run's edge are tested there too: the coarse level on the run's whole
    # interior, the fine level on crops inside it.
    margin = fcfg.margin
    size = data.snapshots.shape[-1]
    crop = (size - 2 * margin) // block if ccfg.coarse_crop else None
    coarse_data = ErosionCoarse(data, cn, runs=runs, lags=ccfg.lags, block=block, augment=False, crop=crop,
                                margin=margin if crop else 0)
    fine_data = ErosionCrops(data, fn, runs=runs, lags=fcfg.lags, crop=fcfg.crop, block=block, seed=seed,
                             augment=False, margin=margin)
    offset = margin if crop else 0  # fine cell of the coarse sample's first cell
    picks = np.sort(np.random.default_rng(seed).choice(len(fine_data), min(count, len(fine_data)), replace=False))
    c_items = [coarse_data[int(p)] for p in tqdm(picks, desc="items", disable=not show)]
    f_items = [fine_data[int(p)] for p in picks]

    def stack(items, key):
        return torch.stack([item[key] for item in items])

    # Every item's K_ref and elapsed time, shaped to broadcast over heightmaps. Item i is the same pair at
    # both levels.
    K_ref = params["K_ref"][[item["run"] for item in f_items]][:, None, None]
    dT = np.array([data.times[item["end"]] - data.times[item["start"]] for item in f_items])[:, None, None]

    # The coarse level: whole runs.
    c_pred = _sample(cm, stack(c_items, "x"), stack(c_items, "cond"), [(seed, "coarse", int(p)) for p in picks],
                     steps, batch_size)[:, 0].double().numpy()
    c_dz = cn.change.from_model(c_pred, K_ref, dT)  # m, over the whole coarse sample
    c_za = stack(c_items, "z_a").double().numpy()
    run, start, end = (np.array([item[k] for item in f_items]) for k in ("run", "start", "end"))
    coarse_pairs = make_pairs(data, run, start, end, c_za, stack(c_items, "z_b").double().numpy())
    add_method(coarse_pairs, "coarse model", c_za + c_dz, "sample")

    # The fine level: crops, with the coarse change from the simulator or from the coarse model.
    c = fcfg.crop // block
    c_change = coarse_under(c_dz, f_items, offset, block, c)  # m, under each fine crop
    x_true = stack(f_items, "x")
    x_model = x_true.clone()
    x_model[:, 1] = torch.as_tensor(fn.change.to_model(upsample(c_change, block), K_ref, dT), dtype=torch.float32)
    keys = [(seed, "fine", int(p)) for p in picks]
    cond = stack(f_items, "cond")
    f_za, f_zb = stack(f_items, "z_a").double().numpy(), stack(f_items, "z_b").double().numpy()
    fine_pairs = make_pairs(data, run, start, end, f_za, f_zb)
    for name, x in [("fine model, simulator's coarse", x_true), ("coarse + fine models", x_model)]:
        y = _sample(fm, x, cond, keys, steps, batch_size)[:, 0].double().numpy()
        add_method(fine_pairs, name, f_za + fn.change.from_model(y, K_ref, dT), "sample")
    add_method(fine_pairs, "simulator's coarse only", f_za + smooth_upsample(block_mean(f_zb - f_za, block), block),
               "baseline")
    add_method(fine_pairs, "coarse model only", f_za + smooth_upsample(c_change, block), "baseline")
    return {"coarse": coarse_pairs, "fine": fine_pairs}


def coarse_under(coarse: np.ndarray, items: list, offset: int, block: int, size: int) -> np.ndarray:
    """The ``size`` × ``size`` piece of each coarse map under fine item ``i``'s crop.

    ``coarse[i]`` starts at fine cell (``offset``, ``offset``) of the run; items carry ``y0`` and ``x0``.
    """
    corners = [((it["y0"] - offset) // block, (it["x0"] - offset) // block) for it in items]
    return np.stack([coarse[i, y:y + size, x:x + size] for i, (y, x) in enumerate(corners)])


def window_disagreement(level: Level, y0: int = 0, x0: int = 0, windows: int = 6, show: bool = True) -> dict:
    """How much neighboring windows disagree where they overlap, before blending, after each phase.

    Takes the ``windows`` × ``windows`` grid of windows whose first is the one at cell (``y0``, ``x0``)
    (rounded down to the stride). For each phase, every window's own output is compared with its right
    and lower neighbors over their overlap (half a window).

    Returns, per phase, ``rms`` (m): the root mean square difference over all overlaps, and
    ``relative``: that over the RMS of the blended ``Δz`` there. Low means the windows already agree
    before the blend; high means the blend averages different answers. The last phase is the result.
    """
    s = level.stride
    i0, j0 = y0 // s, x0 // s
    ctxs = [(0, i0 + i, j0 + j) for i in range(windows) for j in range(windows)]
    result = []
    for phase, (tensor, outputs) in enumerate(zip(level.tensors, level.window_outputs(ctxs))):
        own = dict(zip(ctxs, outputs))
        diffs = []
        for _, a, b in ctxs:
            if (0, a, b + 1) in own:  # right neighbor: my right half against its left half
                diffs.append(own[(0, a, b)][:, s:] - own[(0, a, b + 1)][:, :s])
            if (0, a + 1, b) in own:  # lower neighbor
                diffs.append(own[(0, a, b)][s:] - own[(0, a + 1, b)][:s])
        rms = float(np.sqrt(np.mean([np.mean(d ** 2) for d in diffs])))
        blended = tensor[:, i0 * s:(i0 + windows) * s + s, j0 * s:(j0 + windows) * s + s]
        scale = float(np.sqrt(np.mean(level.to_meters((blended[0] / blended[1]).double().numpy()) ** 2)))
        result.append({"phase": phase, "rms": rms, "relative": rms / scale})
    if show:
        for r in result:
            print(f"{level.name} phase {r['phase']}: windows disagree by {r['rms']:.3f} m RMS "
                  f"({100 * r['relative']:.1f}% of the change)")
    return {"phases": result}


def edge_profile(level: Level, windows: int = 6, show: bool = True) -> dict:
    """Mean absolute change of a level's own window outputs against distance from the window edge.

    A model trained on whole simulator runs learns that the simulator's edges do not change, and puts
    that at every window edge. Inside the world, window edges are not boundaries, so a dip here shows up
    as seams. Returns ``distance`` (cells) and ``mean_abs_change`` (m).
    """
    ctxs = [(0, i, j) for i in range(windows) for j in range(windows)]
    own = level.window_outputs(ctxs)[-1]
    n = level.window
    i = np.arange(n)
    distance = np.minimum.reduce(np.meshgrid(np.minimum(i, n - 1 - i), np.minimum(i, n - 1 - i), indexing="ij"))
    d = np.arange(n // 2)
    profile = np.array([np.abs(own[:, distance == k]).mean() for k in d])
    if show:
        step = max(1, n // 16)
        print(f"{level.name} windows: mean |Δz| (m) at distance " + ", ".join(
            f"{k}: {profile[k]:.1f}" for k in d[::step]) + " cells from the edge")
    return {"distance": d, "mean_abs_change": profile}


def order_invariance(make_world: Callable[[], World], regions: int = 12, size: int = 96, spread: int = 2_000,
                     seed: int = 0, show: bool = True) -> dict:
    """Query random regions in two orders, on two fresh worlds, clearing one's cache halfway.

    ``make_world()`` builds a fresh world (empty cache). Returns ``max_difference`` (m) between the two
    answers for any region, and ``max_change`` (m) for scale. Overlapping windows are summed and batched
    in different orders, so expect float rounding, not zero.
    """
    rng = np.random.default_rng(seed)
    boxes = [(int(y), int(x), size, size) for y, x in rng.integers(-spread, spread, (regions, 2))]
    # Overlapping pairs, so some windows are shared between regions.
    boxes += [(y + size // 2, x + size // 3, size, size) for y, x, _, _ in boxes[: regions // 2]]
    a, b = make_world(), make_world()
    first = {box: a.get(*box) for box in boxes}
    second = {}
    for k, i in enumerate(rng.permutation(len(boxes))):
        second[boxes[i]] = b.get(*boxes[i])
        if k == len(boxes) // 2:
            b.clear_cache()
    diff = max(float(np.abs(first[box] - second[box]).max()) for box in boxes)
    change = max(float(np.abs(a.change(*box)).max()) for box in boxes)
    if show:
        print(f"{len(boxes)} regions of {size}²: largest difference {diff:.2e} m (largest change {change:.1f} m)")
    return {"max_difference": diff, "max_change": change, "regions": len(boxes)}


def chunked_vs_whole(make_world: Callable[[], World], y0: int, x0: int, size: int, chunk: int = 128,
                     show: bool = True) -> dict:
    """The ``size`` × ``size`` region at (``y0``, ``x0``) asked for chunk by chunk, each ``chunk`` × ``chunk``
    chunk from a new world (``make_world()``, empty cache), and tiled; against the same region from one
    world in one query. Each chunk is computed knowing nothing of the others, so the two should agree up
    to float rounding.

    Returns ``max_difference`` (m), the tiled heights ``chunked``, and ``grid``: the chunk edges in the
    region's cells, for drawing.
    """
    whole = make_world().get(y0, x0, size, size)
    chunked = np.empty_like(whole)
    for cy in range(0, size, chunk):
        for cx in range(0, size, chunk):
            h, w = min(chunk, size - cy), min(chunk, size - cx)
            chunked[cy:cy + h, cx:cx + w] = make_world().get(y0 + cy, x0 + cx, h, w)
    diff = float(np.abs(chunked - whole).max())
    if show:
        print(f"{size}² region in {chunk}² chunks, each from a new world: largest difference from one query "
              f"{diff:.1e} m")
    return {"max_difference": diff, "chunked": chunked, "grid": list(range(chunk, size, chunk))}


def query_times(make_world: Callable[[], World], sizes=(64, 128, 256, 512), show: bool = True) -> dict:
    """Seconds to query a square of each size on a fresh world, then again (warm cache).

    Each size is queried far from the others, so no windows are shared.
    """
    result = []
    for k, size in enumerate(sizes):
        world = make_world()
        y0 = 10_000 * (k + 1)
        times = []
        for _ in range(2):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t = time.perf_counter()
            world.get(y0, 0, size, size)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t)
        result.append({"size": size, "first": times[0], "repeat": times[1]})
        if show:
            print(f"{size:>5}²: first {times[0]:7.2f} s ({times[0] / size ** 2 * 1e6:6.1f} µs/cell), "
                  f"repeat {times[1]:.3f} s")
    return {"sizes": result}


def simulator_times(world: World, sizes=(64, 128, 256, 512), dTs=(10_000.0, 50_000.0), dt: float = 100.0,
                    show: bool = True) -> dict:
    """Seconds for the simulator to erode a square of each size of ``world``'s terrain, for each elapsed time.

    One run per size and elapsed time, on one CPU core, with ``world``'s erosion parameters. ``dt`` is the
    simulator's step (yr), as in datagen's ``Config.dt``. Large sizes take minutes.

    Returns ``runs``: one dict per run with ``size``, ``dT``, ``steps``, and ``seconds``.
    """
    p = {k: float(world.params[k][0]) for k in ("K", "D", "m", "U")}
    runs = []
    for size in sizes:
        z0 = world.terrain.heights(0, 0, size, size)
        for dT in dTs:
            steps = int(round(dT / dt))
            seconds = time_simulator(z0, **p, dt=dt, steps=steps)
            runs.append({"size": size, "dT": dT, "steps": steps, "seconds": seconds})
            if show:
                print(f"{size:>5}², dT {dT:>7.0f} yr ({steps} steps): {seconds:8.2f} s")
    return {"runs": runs}


def speed_comparison(worlds: dict[str, dict], simulator: dict | None, show: bool = True) -> dict:
    """Seconds for the world and the simulator to give the same square region.

    ``worlds`` maps a name to a :func:`query_times` result. ``simulator`` is :func:`simulator_times`.
    The simulator's time grows with the elapsed time (one step per 100 yr); the world's does not. The
    simulator runs on one CPU core, the world on the GPU.

    Returns ``rows``: one per size, with seconds for each world (first query and repeat) and for the
    simulator at each elapsed time.
    """
    sim = {}
    for r in (simulator or {}).get("runs", []):
        sim.setdefault(r["size"], {})[r["dT"]] = r["seconds"]
    dTs = sorted({dT for by_dT in sim.values() for dT in by_dT})
    sizes = sorted({r["size"] for t in worlds.values() for r in t["sizes"]} | set(sim))
    first = {name: {r["size"]: r["first"] for r in t["sizes"]} for name, t in worlds.items()}
    repeat = {name: {r["size"]: r["repeat"] for r in t["sizes"]} for name, t in worlds.items()}
    rows = [{"size": n, "simulator": sim.get(n, {}), "first": {k: v.get(n) for k, v in first.items()},
             "repeat": {k: v.get(n) for k, v in repeat.items()}} for n in sizes]
    if not show:
        return {"rows": rows}

    def cell(x):
        return f"{x:>9.2f}" if x is not None else f"{'-':>9}"

    names = list(worlds)
    head = "".join(f"{'sim ' + f'{dT / 1000:g}k yr':>14}" for dT in dTs)
    head += "".join(f"{name + ' (first)':>{max(14, len(name) + 10)}}" for name in names)
    print("Seconds to get one square region. Simulator: one CPU core. World: GPU, first query (empty cache).")
    print(f"{'size':>6}" + head)
    for r in rows:
        line = f"{r['size']:>5}²" + "".join(f"{cell(r['simulator'].get(dT)):>14}" for dT in dTs)
        line += "".join(f"{cell(r['first'][name]):>{max(14, len(name) + 10)}}" for name in names)
        print(line)
    if dTs and names:
        print("\nSpeedup (simulator seconds / world first-query seconds; above 1 means the world is faster):")
        for name in names:
            for dT in dTs:
                ratios = [f"{r['size']}²: {r['simulator'][dT] / r['first'][name]:.2g}×" for r in rows
                          if r["simulator"].get(dT) and r["first"][name]]
                print(f"  {name}, dT {dT / 1000:g}k yr: " + ", ".join(ratios))

    fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
    for k, dT in enumerate(dTs):
        xs = [n for n in sizes if dT in sim.get(n, {})]
        ax.plot(xs, [sim[n][dT] for n in xs], "o-", color="black", alpha=0.4 + 0.6 * k / max(len(dTs) - 1, 1),
                label=f"simulator, dT {dT / 1000:g}k yr (1 core)")
    for k, name in enumerate(names):
        xs = [n for n in sizes if first[name].get(n)]
        color = f"C{k}"
        ax.plot(xs, [first[name][n] for n in xs], "s-", color=color, label=f"world {name}, first query")
        ax.plot(xs, [repeat[name][n] for n in xs], "s:", color=color, alpha=0.6, label=f"world {name}, cached")
    ax.set(xscale="log", yscale="log", xlabel="region side (cells, 1 m each)", ylabel="seconds",
           title="Time to get a square region")
    ax.set_xticks(sizes, [str(n) for n in sizes])
    ax.minorticks_off()
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    plt.show()
    return {"rows": rows}


# ---------------------------------------------------------------------------------------------------
# Summaries and plots


def headline(pairs: dict, show: bool = True) -> dict:
    """One table per level: height error and river quality of every method.

    - ``rmse`` (m) and ``skill`` (``1 - MSE / MSE(no change)``) from ``evaluate.error_metrics``.
    - ``channel IoU``: overlap of the method's channels with the simulator's (top 5% of drainage area),
      within one cell (``iou_1`` of ``evaluate.hydrology_metrics``). 1 is a perfect match.
    - ``pits``: interior cells with no lower neighbor, per terrain. Each one stops the water, so rivers
      break there. The simulator's terrain has few.
    """
    errors = error_metrics(pairs, show=False)
    rivers = hydrology_metrics(pairs, show=False)
    result = {m: {"rmse": errors[m]["rmse"], "skill": errors[m]["skill"], "iou": rivers[m]["iou_1"],
                  "pits": rivers[m]["pits"]} for m in pairs["methods"]}
    result["simulator"] = {"pits": rivers["simulator_pits"]}
    if show:
        w = max(len(m) for m in pairs["methods"])
        print(f"{len(pairs['z_a'])} pairs. Simulator: mean |Δz| {np.abs(pairs['z_b'] - pairs['z_a']).mean():.2f} m, "
              f"{rivers['simulator_pits']:.1f} pits per terrain.\n")
        print(f"{'method':>{w}}  {'RMSE (m)':>8}  {'skill':>6}  {'channel IoU':>11}  {'pits':>6}")
        for m in pairs["methods"]:
            r = result[m]
            print(f"{m:>{w}}  {r['rmse']:8.3f}  {r['skill']:6.3f}  {r['iou']:11.3f}  {r['pits']:6.1f}")
    return result


def show_examples(pairs: dict, methods: list[str], examples=(0, 1, 2), title: str = "", grid=(),
                  simulator: str = "simulator") -> None:
    """For each example pair, two rows of maps.

    Top row: the starting terrain, then the change ``Δz`` (m) of the simulator and of each method, on
    one color scale. Bottom row: each method's error, method minus simulator (m), on its own scale.
    Blue is lowering, red is raising. ``grid`` draws lines at those cells (chunk edges, for example);
    ``simulator`` names the answer key in the titles.
    """
    columns = 2 + len(methods)
    fig, ax = plt.subplots(2 * len(examples), columns, figsize=(3.1 * columns + 1, 5.6 * len(examples)),
                           layout="constrained")
    for r, i in enumerate(examples):
        top, bottom = ax[2 * r], ax[2 * r + 1]
        true = pairs["z_b"][i] - pairs["z_a"][i]
        changes = {simulator: true, **{m: pairs[m][i] - pairs["z_a"][i] for m in methods}}
        lim = max(np.abs(c).max() for c in changes.values())
        top[0].imshow(hillshade(pairs["z_a"][i]))
        top[0].set_title(f"start\ndT {pairs['dT'][i]:.0f} yr", fontsize=9)
        for a, (name, c) in zip(top[1:], changes.items()):
            image = a.imshow(c, cmap="RdBu_r", vmin=-lim, vmax=lim)
            a.set_title(f"Δz, {name}", fontsize=9)
        fig.colorbar(image, ax=top[-1], label="Δz (m)", shrink=0.8)
        errors = {m: changes[m] - true for m in methods}
        elim = max(np.abs(e).max() for e in errors.values())
        for a, (name, e) in zip(bottom[2:], errors.items()):
            image = a.imshow(e, cmap="RdBu_r", vmin=-elim, vmax=elim)
            a.set_title(f"error, {name}\nRMSE {np.sqrt((e ** 2).mean()):.2f} m", fontsize=9)
        fig.colorbar(image, ax=bottom[-1], label="error (m)", shrink=0.8)
        for a in (*top, *bottom):
            a.set_xticks([]); a.set_yticks([])
            for e in grid:
                a.axhline(e - 0.5, color="k", lw=0.5, alpha=0.5); a.axvline(e - 0.5, color="k", lw=0.5, alpha=0.5)
        for a in bottom[:2]:
            a.set_axis_off()
    fig.suptitle(title)
    plt.show()


def by_elapsed(pairs: dict, methods: list[str], bins: int = 6, show: bool = True) -> dict:
    """RMSE and channel IoU against elapsed time, per method, with "no change" as the reference.

    Bins have equal numbers of pairs; each point is the bin's mean elapsed time.
    """
    order = np.argsort(pairs["dT"])
    groups = np.array_split(order, bins)
    result = {"dT": [float(pairs["dT"][g].mean()) for g in groups], "count": [len(g) for g in groups]}
    for m in ["no change", *methods]:
        result[m] = {"rmse": [], "iou": []}
    for g in groups:
        sub = subset_pairs(pairs, g) | {"methods": ["no change", *methods]}
        errors, rivers = error_metrics(sub, show=False), hydrology_metrics(sub, show=False)
        for m in sub["methods"]:
            result[m]["rmse"].append(errors[m]["rmse"])
            result[m]["iou"].append(rivers[m]["iou_1"])
    if show:
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.2), layout="constrained")
        for k, m in enumerate(["no change", *methods]):
            style = dict(color="k", ls="--", marker="s") if m == "no change" else dict(color=f"C{k}", marker="o")
            ax[0].plot(result["dT"], result[m]["rmse"], label=m, **style)
            ax[1].plot(result["dT"], result[m]["iou"], label=m, **style)
        ax[0].set_ylabel("RMSE of the end heights (m)")
        ax[1].set_ylabel("channel IoU, within 1 cell (1 = match)")
        ax[1].set_ylim(0, 1)
        for a in ax:
            a.set_xlabel(f"elapsed time (yr); {result['count'][0]} pairs per point")
            a.legend(fontsize=8)
        plt.show()
    return result


def seam_summary(seams: dict, model_rmse: dict, show: bool = True) -> dict:
    """Window disagreement at the end of sampling against each level's own error.

    ``seams`` is ``{"T = 1": {"coarse": window_disagreement(...), "fine": ...}, "T = 2": ...}``;
    ``model_rmse`` is ``{"coarse": m, "fine": m}``, each level's RMSE against the simulator. If
    neighboring windows disagree by more than the model errs, the blend averages different answers.
    """
    levels = ("coarse", "fine")
    result = {t: {lv: seams[t][lv]["phases"][-1]["rms"] for lv in levels} for t in seams}
    if show:
        fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
        x = np.arange(len(levels))
        width = 0.8 / (len(seams) + 1)
        for k, t in enumerate(seams):
            ax.bar(x + k * width, [result[t][lv] for lv in levels], width, label=f"windows disagree, {t}")
        ax.bar(x + len(seams) * width, [model_rmse[lv] for lv in levels], width, color="0.6",
               label="model error against the simulator (RMSE)")
        ax.set_xticks(x + width * len(seams) / 2, levels)
        ax.set_ylabel("m")
        ax.legend(fontsize=8)
        ax.set_title("Overlap disagreement between neighboring windows, before the final blend")
        plt.show()
    return result


# ---------------------------------------------------------------------------------------------------
# Worlds on dataset samples


def run_world(data: Split, i: int, dT: float, coarse: tuple, fine: tuple, **kwargs) -> World:
    """A world on sample ``i``'s terrain and erosion parameters. ``kwargs`` go to ``World``."""
    e = data.erosion(i)
    if e["U"]:
        raise ValueError(f"sample {i} has uplift; the world has none")
    terrain = data.terrain(i)
    return World(terrain.seed, dT, coarse, fine, K_ref=e["K_ref"], D=e["D"], m=e["m"], terrain=terrain, **kwargs)
