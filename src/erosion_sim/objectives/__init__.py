"""What a model is trained to do, and how it samples.

- ``edm``: the EDM denoiser, its loss, and the Heun sampler.
- ``flow``: flow matching: the velocity model, its loss, and the Euler sampler.
- ``consistency``: consistency distillation of an EDM teacher (1- and 2-step sampling).
- ``residual``: residual diffusion: a regression mean plus a sampled residual.
- ``sampling``: ``generate``, network call counts, and phased sampling, for any objective.

Regression has no module here: the backbone (``models``) predicts the target directly.
"""
