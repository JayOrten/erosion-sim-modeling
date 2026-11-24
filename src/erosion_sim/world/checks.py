"""Checks of the world machinery, as plain functions that return their numbers.

- ``window_disagreement``: how much neighboring windows of a level disagree where they overlap, before
  blending (seams).
- ``order_invariance``: whether query order and clearing the cache change results.
- ``chunked_vs_whole``: whether a region asked for chunk by chunk, each from a new world, equals one query.
- ``query_times``: time of first and repeat queries.
"""

import time
from typing import Callable

import numpy as np
import torch

from .level import Level
from .world import World


def window_disagreement(level: Level, y0: int = 0, x0: int = 0, windows: int = 6) -> dict:
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
    return {"phases": result}


def order_invariance(make_world: Callable[[], World], regions: int = 12, size: int = 96, spread: int = 2_000,
                     seed: int = 0) -> dict:
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
    return {"max_difference": diff, "max_change": change, "regions": len(boxes)}


def chunked_vs_whole(make_world: Callable[[], World], y0: int, x0: int, size: int, chunk: int = 128) -> dict:
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
    return {"max_difference": diff, "chunked": chunked, "grid": list(range(chunk, size, chunk))}


def query_times(make_world: Callable[[], World], sizes=(64, 128, 256, 512)) -> dict:
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
    return {"sizes": result}
