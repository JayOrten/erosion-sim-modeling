"""Procedural starting terrain: Perlin noise summed over octaves (fBm, fractal Brownian motion).

The noise comes from FastNoiseLite (https://github.com/Auburn/FastNoiseLite, Python bindings
``pyfastnoiselite``). It is a function of (seed, x, y): each lattice point's gradient comes from a hash
of the seed and the point, and the octaves are summed with a fixed scale. So any region can be computed
on its own, and overlapping regions agree.

FastNoiseLite computes in 32-bit floats. Coordinates up to about 10^5 cells give smooth noise; at
10^6 it starts to wobble, and at 10^7 it breaks into steps.
"""

import numpy as np
from pyfastnoiselite.pyfastnoiselite import FastNoiseLite, FractalType, NoiseType

# FastNoiseLite seeds are signed 32-bit integers.
SEED_LIMIT = 2**31


def fbm(
    shape: tuple[int, int],
    seed: int = 0,
    period: float = 128.0,
    octaves: int = 5,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
    origin: tuple[int, int] = (0, 0),
) -> np.ndarray:
    """Perlin fBm on the cells ``origin + (row, col)`` for ``shape`` rows and columns, in float64.

    The first octave has lattice cells ``period`` cells wide. Each later octave is ``lacunarity`` times
    finer and ``persistence`` times weaker. Values are in about [-1, 1]; a 128-cell window at period 32
    spans about 1.05.
    """
    noise = FastNoiseLite(seed=int(seed))
    noise.noise_type = NoiseType.NoiseType_Perlin
    noise.frequency = 1.0 / period
    noise.fractal_type = FractalType.FractalType_FBm
    noise.fractal_octaves = octaves
    noise.fractal_gain = persistence
    noise.fractal_lacunarity = lacunarity
    rows = origin[0] + np.arange(shape[0])
    cols = origin[1] + np.arange(shape[1])
    y, x = np.meshgrid(rows, cols, indexing="ij")
    coords = np.stack([x.ravel(), y.ravel()]).astype(np.float32)
    return noise.gen_from_coords(coords).reshape(shape).astype(np.float64)
