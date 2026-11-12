"""Infinite eroded terrain from a seed, with InfiniteDiffusion and a two-level hierarchy.

See docs/plans/infinite-plan.md. Modules:

- ``world``: ``World``, the entry point. ``World.load(seed, dT).get(y0, x0, h, w)``.
- ``level``: one level sampled in overlapping windows, and the coarse-to-fine context.
- ``terrain``: the starting terrain at any region.
- ``noise``: the diffusion noise at any region.
- ``data``: training items for the two levels.
- ``evaluate``: the levels against the simulator, seams, invariance, timing (notebooks/world/world.ipynb).
- ``compare``: model versions side by side on one benchmark of large simulator runs
  (notebooks/world/compare.ipynb).
- ``viewer``: a browser page to pan and zoom over a world (``python -m erosion_sim.world.viewer``).

Import from the modules, for example ``from erosion_sim.world.world import World``. This package does
not import them itself, since ``train`` imports ``world.data`` and ``world.world`` imports ``train``.
"""
