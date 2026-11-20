"""One level of the world, sampled in overlapping windows with InfiniteDiffusion (infinite-plan step 6).

A level covers the whole plane with windows of ``window`` × ``window`` cells at a stride of half a
window. Each window is a sample from the level's model. The sampler's schedule is cut into phases
(``sampling.phase_schedules``: EDM noise levels or flow matching times); after each phase, overlapping
windows are averaged, weighted by a tent that is 1 at the window center and near 0 at its edge. The next
phase continues from the average. A regression model has one phase: its windows are only averaged.
This follows ``annotated_infinite_panorama.py`` in terrain-diffusion, with ``infinite-tensor`` doing
the window bookkeeping, the sums, and the cache.

Each phase is an ``InfiniteTensor`` with 2 channels: the window's values times the tent, and the tent.
The library sums overlapping windows, so dividing channel 0 by channel 1 gives the weighted average.

Everything a window computes is a fixed function of the seed, its position, the parameters, and
``dT``: the starting terrain (``Terrain``), the erodibility map, the diffusion noise
(``noise.diffusion_noise``), and the model. So the result does not depend on query order or on what is
cached. A window's inputs are built as in training (``data.pairs.input_channels``).
"""

from typing import Callable

import numpy as np
import torch
from infinite_tensor import InfiniteTensor, MemoryTileStore, TensorWindow

from ..config import Config
from ..data.normalization import Scaling
from ..data.pairs import input_channels, upsample
from ..objectives.edm import DEFAULT_STEPS
from ..objectives.sampling import phase_schedules, run_phase, start_state, to_target
from .noise import diffusion_noise

# The split for T = 2, as a multiple of sigma_data (plan, "Why T = 2").
SPLIT = 5.0


def tent(size: int) -> torch.Tensor:
    """Blend weight of a window: 1 at the center, falling linearly to 0.001 at the edge, in each axis.

    As ``linear_weight_window`` in terrain-diffusion's ``world_pipeline.py``.
    """
    mid = (size - 1) / 2
    w = 1 - 0.999 * (torch.arange(size, dtype=torch.float64) - mid).abs() / mid
    return (w[:, None] * w[None, :]).float()


# datagen's reference drainage area (m^2) and exponent: the simulator uses K = K_ref * A_REF ** (M_REF - m).
A_REF, M_REF = 30.0, 0.4


def world_params(K_ref: float, D: float, m: float) -> dict:
    """Erosion parameters in the form ``ParameterScaling.scale`` reads: one-run arrays, run 0.

    ``K_ref`` is the erodibility at the reference exponent, as datagen draws it; the simulator's ``K``
    follows from it as in datagen's ``erodibility``. ``D`` is the hillslope diffusivity (m^2/yr).
    """
    K = K_ref * A_REF ** (M_REF - m)
    return {name: np.array([float(v)]) for name, v in dict(K=K, D=D, m=m, K_ref=K_ref).items()}


class Level:
    """One level of the world: its change ``Δz`` (m) at any region, sampled in windows.

    - ``model``, ``scaling``, ``cfg``: a trained level, from ``load_checkpoint``. ``cfg.window`` is the window
      size in cells (the stride is half of it), and ``cfg.inputs`` the extra input channels.
    - ``terrain``: ``terrain(y0, x0, h, w)`` gives the starting heights (m) on this level's cells.
    - ``log_map``: ``log_map(y0, x0, h, w)``, ``log10`` of the erodibility factors on this level's cells;
      needed if the model takes the map as input. None is a uniform map.
    - ``seed``, ``name``: pick the diffusion noise field; each level has its own ``name``.
    - ``params``, ``dT``: from ``world_params``, and the elapsed time in yr. The same everywhere.
    - ``splits``: noise levels where phases end, as multiples of ``sigma_data``. ``(SPLIT,)`` is T = 2;
      ``()`` is T = 1, where windows are only averaged at the end.
    - ``context``: optional ``context(y0, x0, h, w)`` giving the parent level's change in this model's
      units, shape (h, w) (``parent_context``).
    - ``batch_size``: windows that go through the model together. Batching changes float rounding
      slightly, so which windows share a batch can change results at that level.
    - ``bf16``: run the network in 16-bit autocast on the GPU (ignored on CPU), as in training. Faster;
      results differ from 32-bit at about 1e-3 relative.
    """

    def __init__(self, model, scaling: Scaling, cfg: Config, terrain: Callable, seed: int, name: str, params: dict,
                 dT: float, splits: tuple[float, ...] = (SPLIT,), steps: int = DEFAULT_STEPS,
                 context: Callable | None = None, log_map: Callable | None = None, batch_size: int = 16,
                 bf16: bool = False):
        window = cfg.window
        if window is None or window % 2:
            raise ValueError("a world level needs an even window, so the stride is half of it")
        if (cfg.context_cell is None) != (context is None):
            raise ValueError("a level gets its parent's change exactly when it was trained with context_cell")
        self.model, self.scaling, self.cfg, self.terrain, self.context = model, scaling, cfg, terrain, context
        self.log_map = log_map
        self.seed, self.name, self.window, self.stride = seed, name, window, window // 2
        self.device = next(model.parameters()).device
        self.bf16 = bf16 and self.device.type == "cuda"
        self.weight = tent(window)
        self.K_ref, self.dT = float(params["K_ref"][0]), float(dT)
        self.cond = torch.as_tensor(scaling.parameters.scale(params, [0], dT), dtype=torch.float32, device=self.device)
        self.phases = phase_schedules(model, steps, splits)
        self.store = MemoryTileStore()

        layout = TensorWindow(size=(2, window, window), stride=(2, self.stride, self.stride))
        shape = (2, None, None)
        tensors = [InfiniteTensor(shape, self._first_phase, layout, tile_store=self.store, tensor_id=f"{name}-phase0",
                                  batch_size=batch_size)]
        for i in range(1, len(self.phases)):
            tensors.append(InfiniteTensor(shape, self._later_phase(i), layout, args=(tensors[-1],),
                                          args_windows=(layout,), tile_store=self.store, tensor_id=f"{name}-phase{i}",
                                          batch_size=batch_size))
        self.tensors = tensors
        self.result = tensors[-1]

    def clear_cache(self) -> None:
        """Drop every cached window, in every phase. Later queries recompute them."""
        for tensor in self.tensors:
            tensor.clear_cache()

    def origin(self, ctx) -> tuple[int, int]:
        """Top-left cell of window ``ctx`` (its index in window space)."""
        return ctx[1] * self.stride, ctx[2] * self.stride

    def _inputs(self, ctxs: list) -> torch.Tensor:
        """The model inputs of windows ``ctxs``: starting terrain, plus any context channels.

        The terrain and the context are each read in one query over the area covering every window, then
        cut into windows. Windows overlap by half, so this computes each terrain cell once instead of up to
        four times, and the level the context comes from (the coarse level) computes its windows in large
        batches. Both are fixed functions of position, so the values are the same as per-window queries.
        """
        origins = [self.origin(ctx) for ctx in ctxs]
        n = self.window
        top, left = min(y for y, _ in origins), min(x for _, x in origins)
        h, w = max(y for y, _ in origins) + n - top, max(x for _, x in origins) + n - left
        terrain = np.asarray(self.terrain(top, left, h, w))
        context = None if self.context is None else np.asarray(self.context(top, left, h, w))
        log_map = None
        if "map" in self.cfg.inputs and self.log_map is not None:
            log_map = np.asarray(self.log_map(top, left, h, w))
        inputs = []
        for y0, x0 in origins:
            cut = (slice(y0 - top, y0 - top + n), slice(x0 - left, x0 - left + n))
            inputs.append(input_channels(terrain[cut], self.cfg.cell, self.cfg.inputs,
                                         None if log_map is None else log_map[cut],
                                         None if context is None else context[cut]))
        return torch.as_tensor(np.stack(inputs), dtype=torch.float32, device=self.device)

    def _run(self, ctxs: list, y: list[torch.Tensor], phase: int) -> list[torch.Tensor]:
        """Run phase ``phase`` of the schedule on windows ``ctxs``, each from its ``y`` (window, window).

        Returns each window's output for ``infinite-tensor``: (values × tent, tent).
        """
        x = self._inputs(ctxs)
        y = torch.stack(y)[:, None].to(self.device)
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.bf16):
            out = run_phase(self.model, x, self.cond.expand(len(ctxs), -1), y, self.phases[phase])
        out = out[:, 0].float().cpu()
        return [torch.stack([o * self.weight, self.weight]) for o in out]

    def _first_phase(self, ctxs: list) -> list[torch.Tensor]:
        noise = [start_state(self.model, torch.as_tensor(diffusion_noise(self.seed, self.name, *self.origin(ctx),
                                                                         self.window, self.window)), self.phases)
                 for ctx in ctxs]
        return self._run(ctxs, noise, 0)

    def _later_phase(self, phase: int) -> Callable:
        def f(ctxs: list, previous: list[torch.Tensor]) -> list[torch.Tensor]:
            return self._run(ctxs, [p[0] / p[1].clamp(min=1e-6) for p in previous], phase)
        return f

    def window_outputs(self, ctxs: list) -> list[np.ndarray]:
        """Each window's own output (m) at the end of each phase, before it is blended with its neighbors:
        one array (len(ctxs), window, window) per phase. For checking how much neighbors disagree.

        Every phase is recomputed for these windows; each later phase starts from the cached blend of the
        phase before, as in a normal query.
        """
        s, n = self.stride, self.window
        outputs, result = self._first_phase(ctxs), []
        for phase in range(len(self.tensors)):
            if phase:
                previous = [self.tensors[phase - 1][:, a * s:a * s + n, b * s:b * s + n] for _, a, b in ctxs]
                outputs = self._later_phase(phase)(ctxs, previous)
            result.append(np.stack([self.to_meters((o[0] / o[1]).double().numpy()) for o in outputs]))
        return result

    def change(self, y0: int, x0: int, h: int, w: int) -> np.ndarray:
        """``Δz`` (m) on cells ``y0 .. y0 + h - 1`` by ``x0 .. x0 + w - 1``, float64."""
        summed = self.result[:, y0:y0 + h, x0:x0 + w]
        return self.to_meters((summed[0] / summed[1].clamp(min=1e-6)).double().numpy())

    def to_meters(self, y: np.ndarray) -> np.ndarray:
        """A blended final state of this level's sampler to a change in m."""
        target = to_target(self.model, torch.as_tensor(y)).numpy()
        return self.scaling.change.from_model(target, self.K_ref, self.dT)

    def heights(self, y0: int, x0: int, h: int, w: int) -> np.ndarray:
        """Eroded heights (m): the starting terrain plus ``change``."""
        return self.terrain(y0, x0, h, w) + self.change(y0, x0, h, w)


def parent_context(parent: Level, child_scaling: Scaling, ratio: int) -> Callable:
    """A level's context: its parent's change over a region of the level's cells, as one input channel.

    The parent's cells are ``ratio`` of the level's cells wide, so cells ``y0 .. y0 + h - 1`` lie in parent
    cells ``y0 // ratio ..``; ``y0``, ``x0``, ``h``, and ``w`` must be multiples of ``ratio``. The parent's ``Δz``
    (m) is repeated to the level's cells and converted to the level's model units (``child_scaling``), as
    ``data.pairs.ErosionPairs`` does with the true change in training.
    """

    def context(y0: int, x0: int, h: int, w: int) -> np.ndarray:
        if y0 % ratio or x0 % ratio or h % ratio or w % ratio:
            raise ValueError(f"regions must be aligned to {ratio}-cell blocks")
        dz = parent.change(y0 // ratio, x0 // ratio, h // ratio, w // ratio)
        return child_scaling.change.to_model(upsample(dz, ratio), parent.K_ref, parent.dT)

    return context
