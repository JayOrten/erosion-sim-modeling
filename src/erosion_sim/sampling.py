"""One way to get a prediction from any trained model, whatever its objective.

- Regression (``UNet`` or ``DiT`` alone): one network call.
- EDM (``diffusion.Denoiser``): Heun sampler, ``2 * steps - 1`` network calls.
- Flow matching (``flow.FlowModel``): Euler sampler, ``steps`` network calls.

Code that predicts goes through ``generate`` instead of checking the model's type itself.
"""

import torch

from . import diffusion, flow


def is_generative(model) -> bool:
    """Whether ``model`` samples from noise (EDM or flow matching) rather than predicting directly."""
    return isinstance(model, (diffusion.Denoiser, flow.FlowModel))


def generate(model, x: torch.Tensor, cond: torch.Tensor, noise: torch.Tensor | None, steps: int) -> torch.Tensor:
    """The model's output for inputs ``x`` and conditioning ``cond``: a sample from ``noise`` for generative
    models, the direct prediction otherwise (``noise`` and ``steps`` are then ignored)."""
    if isinstance(model, diffusion.Denoiser):
        return diffusion.sample(model, x, cond, noise, steps)
    if isinstance(model, flow.FlowModel):
        return flow.sample(model, x, cond, noise, steps)
    return model(x, cond)


def network_calls(model, steps: int) -> int:
    """Network calls for one prediction with ``steps`` sampler steps."""
    if isinstance(model, diffusion.Denoiser):
        return 2 * steps - 1
    if isinstance(model, flow.FlowModel):
        return steps
    return 1
