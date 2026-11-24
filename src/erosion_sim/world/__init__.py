"""Infinite eroded terrain from a seed, with InfiniteDiffusion and a stack of levels (track C).

See docs/plans/done/infinite-plan.md. Modules:

- ``world``: ``World``, the entry point: ``World.load(seed, dT, folders).get(y0, x0, h, w)``.
- ``level``: one level sampled in overlapping windows, and the context a level reads from its parent.
- ``noise``: the diffusion noise at any region.
- ``suite``: the world evaluation suite, written as a results folder.
- ``checks``: seams, query-order invariance, chunked against whole queries, timing.
- ``viewer``: a browser page to pan and zoom over a world (``python -m erosion_sim.world.viewer``).

A world level is trained like any erosion model, on windows (``data.pairs.ErosionPairs`` with ``window``).
Import from the modules, for example ``from erosion_sim.world.world import World``.
"""
