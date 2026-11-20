"""What the models train on.

- ``table``: run tables (``runs.npz``), and selecting runs by column.
- ``splits``: ``Split``, one split of a generated erosion dataset.
- ``pairs``: ``ErosionPairs``, the training items of tracks B and C, their input channels, and
  ``make_datasets``.
- ``normalization``: conversions to and from model units.
- ``hydrology``: D8 routing, drainage area, pit filling, pits, and the slope-area fit (Numba).
"""
