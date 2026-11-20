"""Consistency distillation (Song et al. 2023; experiments B13 and C9): a student that maps any noisy
target straight to the clean one, so it samples in 1 or 2 network calls, trained from an EDM teacher.

- ``ConsistencyModel`` is an EDM ``Denoiser`` with the consistency boundary condition: at the lowest noise
  level it returns its input exactly (``c_skip = 1``, ``c_out = 0``).
- Training: pick adjacent levels ``σ_n < σ_{n+1}`` on a Karras schedule of ``SCALES`` levels; noise the
  target to ``σ_{n+1}``; one Heun step of the teacher brings it to ``σ_n``. The student at ``σ_{n+1}`` must
  match the target network (the EMA of the student, without gradient) at ``σ_n``.
- Sampling: the student at the highest level gives the target in one call. With more steps, the result
  is noised again to the next lower level of the step schedule and denoised again. The extra noise is the
  current state over its level (pure noise at the first step), shifted by half the window and
  sign-flipped (``renoise``), so a sample stays a fixed function of its starting noise
  (``sampling.generate`` takes one noise field) and the world can run each step as a phase.
"""

import torch

from .edm import SIGMA_MIN, Denoiser, sigma_steps

SCALES = 18  # noise levels of the training schedule


class ConsistencyModel(Denoiser):
    """A ``Denoiser`` with consistency preconditioning:
    ``c_skip = σ_d² / ((σ - σ_min)² + σ_d²)`` and ``c_out = (σ - σ_min) σ_d / sqrt(σ² + σ_d²)``."""

    def forward(self, x: torch.Tensor, cond: torch.Tensor, noisy: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        noisy = self.edges(noisy)
        s = sigma.reshape(-1, 1, 1, 1).to(noisy.dtype)
        sd = self.sigma_data
        low = SIGMA_MIN * sd
        c_skip = sd**2 / ((s - low) ** 2 + sd**2)
        c_out = (s - low) * sd / (s**2 + sd**2).sqrt()
        c_in = 1 / (s**2 + sd**2).sqrt()
        net_cond = torch.cat([cond, self.noise_feature(sigma.to(cond.dtype))[:, None]], dim=1)
        out = self.net(torch.cat([x, c_in * noisy], dim=1), net_cond)
        return self.edges(c_skip * noisy + c_out * out)


def training_levels(sigma_data: float, device=None) -> torch.Tensor:
    """The ``SCALES`` noise levels of the training schedule, lowest first."""
    return sigma_steps(SCALES, sigma_data)[:-1].flip(0).to(device=device, dtype=torch.float32)


@torch.no_grad()
def teacher_step(teacher: Denoiser, x, cond, y: torch.Tensor, s_from: torch.Tensor, s_to: torch.Tensor) -> torch.Tensor:
    """One Heun step of the teacher's probability-flow ODE from ``s_from`` to ``s_to`` (per sample)."""
    a, b = s_from.reshape(-1, 1, 1, 1), s_to.reshape(-1, 1, 1, 1)
    d = (y - teacher(x, cond, y, s_from).float()) / a
    y_next = y + (b - a) * d
    d_next = (y_next - teacher(x, cond, y_next, s_to).float()) / b
    return y + (b - a) * 0.5 * (d + d_next)


def consistency_loss(student, target, teacher, x, cond, y, n: torch.Tensor, noise: torch.Tensor,
                     weight: torch.Tensor | None = None) -> torch.Tensor:
    """Per-sample squared difference between the student at level ``n + 1`` and the target network at
    level ``n`` after one teacher step (``n``: per-sample indices into ``training_levels``)."""
    levels = training_levels(student.sigma_data, y.device)
    s_hi, s_lo = levels[n + 1], levels[n]
    noisy = y + s_hi.reshape(-1, 1, 1, 1) * noise
    stepped = teacher_step(teacher, x, cond, noisy, s_hi, s_lo)
    with torch.no_grad():
        goal = target(x, cond, stepped, s_lo).float()
    error = (student(x, cond, noisy, s_hi).float() - goal).pow(2)
    return (error if weight is None else error * weight).mean(dim=(1, 2, 3))


def training_loss(student, target, teacher, x, cond, y, generator: torch.Generator,
                  weight: torch.Tensor | None = None) -> torch.Tensor:
    n = torch.randint(0, SCALES - 1, (len(y),), device=y.device, generator=generator)
    noise = torch.randn(y.shape, device=y.device, generator=generator)
    return consistency_loss(student, target, teacher, x, cond, y, n, noise, weight).mean()


def renoise(noise: torch.Tensor) -> torch.Tensor:
    """Unit noise for the next sampling step, from ``noise`` (the state over its level): shifted by half
    the field in both axes and sign-flipped. Deterministic, so a sample depends only on its starting noise."""
    h, w = noise.shape[-2:]
    return -torch.roll(noise, shifts=(h // 2, w // 2), dims=(-2, -1))


def step_levels(model: ConsistencyModel, steps: int) -> torch.Tensor:
    """The sampling levels, then 0: the first ``steps`` of ``steps + 1`` EDM levels from the highest down
    (``edm.sigma_steps``). The lowest level is left out, since the student returns its input there;
    two steps use the highest level and one partway down (about 5 ``sigma_data``)."""
    levels = sigma_steps(steps + 1, model.sigma_data)
    return torch.cat([levels[:steps], levels[-1:]])


@torch.no_grad()
def denoise_and_renoise(model: ConsistencyModel, x, cond, state: torch.Tensor, s: float, s_next: float,
                        noise: torch.Tensor) -> torch.Tensor:
    """The student's clean estimate from ``state`` at level ``s``, noised again to ``s_next`` (0: none)."""
    clean = model(x, cond, state, torch.full((len(state),), s, device=state.device)).float()
    if s_next <= 0:
        return clean
    low = SIGMA_MIN * model.sigma_data
    return clean + (max(s_next**2 - low**2, 0.0) ** 0.5) * noise


@torch.no_grad()
def sample(model: ConsistencyModel, x: torch.Tensor, cond: torch.Tensor, noise: torch.Tensor, steps: int) -> torch.Tensor:
    """A sample in ``steps`` network calls (1 or 2 is the point)."""
    noise = noise.to(device=x.device, dtype=torch.float32)
    levels = step_levels(model, steps).tolist()
    state = noise * levels[0]
    for i in range(steps):
        state = denoise_and_renoise(model, x, cond, state, levels[i], levels[i + 1], renoise(state / levels[i]))
    return state
