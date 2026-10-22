"""Water routing on a heightmap, in plain numpy.

The simulator (in datagen/) routes water over a pit-filled copy of the terrain with Landlab. This
module has no pit filling: water that reaches a pit stops there. On terrain without pits the two give
the same drainage area. Pits in a model's output therefore show up here as broken rivers.
"""

import numpy as np

# The 8 neighbors, as (row offset, column offset).
_OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def receivers(z: np.ndarray, cell_size: float = 1.0) -> np.ndarray:
    """Flat index of the node each node drains to: its steepest downhill neighbor (D8).

    Edge nodes, and nodes with no lower neighbor (pits), drain to themselves.
    """
    rows, cols = z.shape
    padded = np.pad(z, 1, constant_values=np.inf)
    ii, jj = np.mgrid[0:rows, 0:cols]
    best = np.zeros(z.shape)
    target = (ii * cols + jj).copy()
    for di, dj in _OFFSETS:
        neighbor = padded[1 + di:1 + di + rows, 1 + dj:1 + dj + cols]
        slope = (z - neighbor) / (cell_size * np.hypot(di, dj))
        better = slope > best
        best = np.where(better, slope, best)
        target = np.where(better, (ii + di) * cols + (jj + dj), target)
    edge = np.zeros(z.shape, dtype=bool)
    edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
    target[edge] = (ii * cols + jj)[edge]
    return target.ravel()


def drainage_area(z: np.ndarray, cell_size: float = 1.0) -> np.ndarray:
    """Area (m²) of land that drains through each node, including the node itself."""
    target = receivers(z, cell_size)
    area = np.full(z.size, cell_size * cell_size)
    for node in np.argsort(z, axis=None)[::-1]:  # highest first, so every node is final before it is passed on
        if target[node] != node:
            area[target[node]] += area[node]
    return area.reshape(z.shape)


def pits(z: np.ndarray) -> np.ndarray:
    """Boolean mask of interior nodes with no lower neighbor."""
    target = receivers(z).reshape(z.shape)
    rows, cols = z.shape
    own = np.arange(z.size).reshape(z.shape)
    mask = target == own
    mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = False
    return mask
