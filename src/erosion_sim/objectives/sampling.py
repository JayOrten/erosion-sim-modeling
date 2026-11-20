"""One way to get a prediction from any trained model, whatever its objective.

- Regression (``UNet`` or ``DiT`` alone): one network call.
- EDM (``edm.Denoiser``): Heun sampler, ``2 * steps - 1`` network calls.
- Flow matching (``flow.FlowModel``): Euler sampler, ``steps`` network calls.
- Consistency student (``consistency.ConsistencyModel``): ``steps`` network calls, 1 or 2 being the point.
- Residual (``residual.ResidualModel``): one regression call plus its generative model's sampler.

Code that predicts goes through ``generate`` instead of checking the model's type itself.

The world samples in phases, blending overlapping windows between them (``world.level``). For that,
``phase_schedules`` cuts a model's schedule at noise levels ``splits`` (multiples of ``sigma_data``),
``start_state`` and ``run_phase`` run it phase by phase, and ``to_target`` turns the final state into the
target's units. Running every phase in turn equals ``generate``.
"""

import torch

from . import consistency, edm, flow
from .residual import ResidualModel


def is_generative(model) -> bool:
    """Whether ``model`` samples from noise (EDM or flow matching) rather than predicting directly."""
    return isinstance(model, (edm.Denoiser, flow.FlowModel, ResidualModel))


@torch.no_grad()
def generate(model, x: torch.Tensor, cond: torch.Tensor, noise: torch.Tensor | None, steps: int) -> torch.Tensor:
    """The model's output for inputs ``x`` and conditioning ``cond``: a sample from ``noise`` for generative
    models, the direct prediction otherwise (``noise`` and ``steps`` are then ignored)."""
    if isinstance(model, ResidualModel):
        return model.base(x, cond).float() + generate(model.inner, x, cond, noise, steps)
    if isinstance(model, consistency.ConsistencyModel):
        return consistency.sample(model, x, cond, noise, steps)
    if isinstance(model, edm.Denoiser):
        return edm.sample(model, x, cond, noise, steps)
    if isinstance(model, flow.FlowModel):
        return flow.sample(model, x, cond, noise, steps)
    return model(x, cond)


def network_calls(model, steps: int) -> int:
    """Network calls for one prediction with ``steps`` sampler steps."""
    if isinstance(model, ResidualModel):
        return 1 + network_calls(model.inner, steps)
    if isinstance(model, consistency.ConsistencyModel):
        return steps
    if isinstance(model, edm.Denoiser):
        return 2 * steps - 1
    if isinstance(model, flow.FlowModel):
        return steps
    return 1


def phase_schedules(model, steps: int, splits: tuple[float, ...] = ()) -> list:
    """The model's sampler schedule cut into phases: EDM noise levels, flow matching times, or one
    phase with no schedule for regression. A consistency student's steps are each a phase, whatever the
    splits: ``[level, next level]`` per step. A residual model cannot be split."""
    if isinstance(model, ResidualModel):
        raise ValueError("a residual model's regression part needs its inputs at the end; it cannot run in phases")
    if isinstance(model, consistency.ConsistencyModel):
        levels = consistency.step_levels(model, steps)
        return [levels[i:i + 2] for i in range(steps)]
    if isinstance(model, edm.Denoiser):
        return edm.phase_sigmas(steps, model.sigma_data, splits)
    if isinstance(model, flow.FlowModel):
        return flow.phase_times(steps, splits)
    return [None]


def start_state(model, noise: torch.Tensor, schedules: list) -> torch.Tensor:
    """The state the first phase starts from: noise at the highest EDM level, the flow path's noise end, or
    nothing (zeros) for regression."""
    if isinstance(model, edm.Denoiser):
        return noise * schedules[0][0].item()
    if isinstance(model, flow.FlowModel):
        return noise
    return torch.zeros_like(noise)


@torch.no_grad()
def run_phase(model, x: torch.Tensor, cond: torch.Tensor, state: torch.Tensor, schedule) -> torch.Tensor:
    """One phase from ``state``. Regression ignores the state and predicts directly."""
    if isinstance(model, consistency.ConsistencyModel):
        s, s_next = schedule.tolist()
        return consistency.denoise_and_renoise(model, x, cond, state, s, s_next, consistency.renoise(state / s))
    if isinstance(model, edm.Denoiser):
        return edm.heun(model, x, cond, state, schedule)
    if isinstance(model, flow.FlowModel):
        return flow.euler(model, x, cond, state, schedule)
    return model(x, cond).float()


def to_target(model, state: torch.Tensor) -> torch.Tensor:
    """The final state in the target's units (flow matching works in units of ``sigma_data``)."""
    return state * model.sigma_data if isinstance(model, flow.FlowModel) else state

