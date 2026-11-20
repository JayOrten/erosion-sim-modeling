"""Procedural starting terrain: noise summed over octaves (fractal noise), from FastNoiseLite.

The noise comes from FastNoiseLite (https://github.com/Auburn/FastNoiseLite, Python bindings
``pyfastnoiselite``). It is a function of (seed, x, y): each lattice point's value or gradient comes from
a hash of the seed and the point, and the octaves are summed with a fixed scale. So any region can be
computed on its own, and overlapping regions agree.

- ``Terrain``: one starting terrain, as a frozen dataclass of every setting. Datasets store these fields
  per sample, and the infinite world takes one.
- ``fractal_noise``: every option: noise type, fractal type, octaves, persistence, lacunarity, and
  domain warp.
- ``fbm``: Perlin fBm (fractal Brownian motion), the original starting terrain. The same as
  ``fractal_noise`` with its defaults.
- ``ErodibilityMap``: a factor on the erodibility ``K`` per cell (uniform, patches, or bands), also a
  function of world coordinates, so the world can compute it anywhere.

FastNoiseLite computes in 32-bit floats. Coordinates up to about 10^5 cells give smooth noise; at
10^6 it starts to wobble, and at 10^7 it breaks into steps.
"""

from dataclasses import dataclass, fields

import numpy as np
from pyfastnoiselite.pyfastnoiselite import FastNoiseLite, FractalType, NoiseType

# FastNoiseLite seeds are signed 32-bit integers.
SEED_LIMIT = 2**31


# Names for the noise and fractal types, as ``Terrain`` and ``fractal_noise`` take them.
NOISE_TYPES = ("perlin", "opensimplex2", "value_cubic")
FRACTAL_TYPES = ("fbm", "ridged")
_NOISE = {"perlin": NoiseType.NoiseType_Perlin, "opensimplex2": NoiseType.NoiseType_OpenSimplex2,
          "value_cubic": NoiseType.NoiseType_ValueCubic}
_FRACTAL = {"fbm": FractalType.FractalType_FBm, "ridged": FractalType.FractalType_Ridged}
# Domain warp: two OpenSimplex2 fBm fields of 3 octaves, at the terrain's period, shift the coordinates.
WARP_OCTAVES = 3


def fractal_noise(
    shape: tuple[int, int],
    seed: int = 0,
    period: float = 128.0,
    octaves: int = 5,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
    noise: str = "perlin",
    fractal: str = "fbm",
    warp: float = 0.0,
    origin: tuple[float, float] = (0, 0),
    step: float = 1,
) -> np.ndarray:
    """Fractal noise on the cells ``origin + step * (row, col)`` for ``shape`` rows and columns, in float64.

    - ``period``: the first octave's lattice cells are ``period`` cells wide. Each later octave is
      ``lacunarity`` times finer and ``persistence`` times weaker.
    - ``noise``: one of ``NOISE_TYPES``. Perlin and OpenSimplex2 are gradient noise (smooth hills);
      cubic value noise is blobbier.
    - ``fractal``: one of ``FRACTAL_TYPES``. "fbm" sums the octaves; "ridged" folds each octave at zero
      (sharp ridges, wide rounded valleys).
    - ``warp``: domain warp strength. Each cell is sampled at coordinates shifted by up to about
      ``warp * period`` cells, along two more noise fields (seeds ``seed + 1`` and ``seed + 2``). 0 is
      none. It bends ridges and valleys into less regular shapes.
    - ``step`` > 1 samples every ``step``-th cell, for quick previews far zoomed out. ``origin`` may then
      be fractional; Perlin noise is 0 on its lattice, so samples on a lattice multiple are all 0.

    Each kind has its own feature size and spread at the same ``period``. Perlin fBm values are in about
    [-1, 1], with standard deviation about 0.18 at 5 octaves and persistence 0.5.
    """
    if noise not in _NOISE or fractal not in _FRACTAL:
        raise ValueError(f"noise must be one of {NOISE_TYPES} and fractal one of {FRACTAL_TYPES}")
    rows = origin[0] + step * np.arange(shape[0])
    cols = origin[1] + step * np.arange(shape[1])
    y, x = np.meshgrid(rows, cols, indexing="ij")
    if warp:
        offsets = [fractal_noise(shape, (seed + k) % SEED_LIMIT, period, WARP_OCTAVES, 0.5, 2.0, "opensimplex2",
                                 origin=origin, step=step) for k in (1, 2)]
        y = y + warp * period * offsets[0]
        x = x + warp * period * offsets[1]
    field = FastNoiseLite(seed=int(seed))
    field.noise_type = _NOISE[noise]
    field.frequency = 1 / period
    field.fractal_type = _FRACTAL[fractal]
    field.fractal_octaves = int(octaves)
    field.fractal_gain = persistence
    field.fractal_lacunarity = lacunarity
    coords = np.stack([x.ravel(), y.ravel()]).astype(np.float32)
    return field.gen_from_coords(coords).reshape(shape).astype(np.float64)


def fbm(
    shape: tuple[int, int],
    seed: int = 0,
    period: float = 128.0,
    octaves: int = 5,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
    origin: tuple[int, int] = (0, 0),
    step: int = 1,
) -> np.ndarray:
    """Perlin fBm on the cells ``origin + step * (row, col)`` for ``shape`` rows and columns, in float64.

    ``step`` > 1 samples every ``step``-th cell, for quick previews far zoomed out. ``origin`` may then
    be fractional; Perlin noise is 0 on its lattice, so samples on a lattice multiple are all 0.

    The first octave has lattice cells ``period`` cells wide. Each later octave is ``lacunarity`` times
    finer and ``persistence`` times weaker. Values are in about [-1, 1]; a 128-cell window at period 32
    spans about 1.05.
    """
    return fractal_noise(shape, seed, period, octaves, persistence, lacunarity, origin=origin, step=step)


@dataclass(frozen=True)
class Terrain:
    """A starting terrain: heights in m at 1 m cells, ``height`` times ``fractal_noise``.

    The fields are ``fractal_noise``'s settings, plus ``height`` (m per unit of noise). It exists at every
    cell, so any region can be computed on its own. The defaults are the infinite world's terrain.
    """

    seed: int
    period: float = 32.0
    persistence: float = 0.5
    octaves: int = 5
    lacunarity: float = 2.0
    height: float = 64.0
    noise: str = "perlin"
    fractal: str = "fbm"
    warp: float = 0.0

    def heights(self, y0: float, x0: float, h: int, w: int, step: float = 1) -> np.ndarray:
        """Heights (m) on cells ``y0 .. y0 + h - 1`` by ``x0 .. x0 + w - 1``, in float64.

        ``step`` > 1 samples every ``step``-th cell from (``y0``, ``x0``), for previews.
        """
        return self.height * fractal_noise((h, w), self.seed, self.period, self.octaves, self.persistence,
                                           self.lacunarity, self.noise, self.fractal, self.warp, origin=(y0, x0),
                                           step=step)


# Terrain's fields, in order. Datasets store one array per field.
TERRAIN_FIELDS = tuple(f.name for f in fields(Terrain))


# ---- Erodibility maps ----

MAP_SHAPES = ("uniform", "patches", "bands")
PATCH_OCTAVES = 3
# Patches: the smooth step from hard to soft spans fBm values -PATCH_EDGE to PATCH_EDGE. The 3-octave
# fBm has a standard deviation of about 0.2, so most cells end near 0 or 1.
PATCH_EDGE = 0.04


def smoothstep(x: np.ndarray, low: float, high: float) -> np.ndarray:
    """0 below ``low``, 1 above ``high``, and the smooth cubic ``3t² - 2t³`` between."""
    t = np.clip((x - low) / (high - low), 0.0, 1.0)
    return t * t * (3 - 2 * t)


@dataclass(frozen=True)
class ErodibilityMap:
    """A factor on ``K`` per cell, ``contrast ** (s - 1/2)``, where ``s`` is 0 on the hardest rock and 1 on
    the softest. The softest rock erodes ``contrast`` times faster than the hardest, and the geometric
    mean is about 1, so ``K_ref`` stays the overall rate.

    - ``uniform``: ``s`` = 1/2 everywhere; every factor is 1.
    - ``patches``: Perlin fBm at ``period`` (3 octaves) through a smooth step, so most cells are near 0 or
      1. Stands for intrusions and mixed rock.
    - ``bands``: stripes ``band_width`` cells wide along ``band_angle`` degrees, alternating 0 and 1, with
      a smooth transition about ``band_width / 8`` wide. Each stripe is bent sideways by up to about one
      ``band_width`` by Perlin noise at ``bend_period`` (the terrain's period). Stands for tilted layers.

    ``seed`` seeds the noise. Like ``Terrain``, it exists at every cell.
    """

    shape: str = "uniform"
    contrast: float = 1.0
    period: float = 64.0
    band_width: float = 32.0
    band_angle: float = 0.0
    bend_period: float = 32.0
    seed: int = 0

    def __post_init__(self):
        if self.shape not in MAP_SHAPES:
            raise ValueError(f"shape must be one of {MAP_SHAPES}, got {self.shape!r}")
        if self.contrast < 1:
            raise ValueError("contrast is the softest rate over the hardest, at least 1")

    def softness(self, y0: float, x0: float, h: int, w: int) -> np.ndarray:
        """``s`` on cells ``y0 .. y0 + h - 1`` by ``x0 .. x0 + w - 1``: 0 hardest, 1 softest."""
        if self.shape == "uniform":
            return np.full((h, w), 0.5)
        if self.shape == "patches":
            n = fractal_noise((h, w), self.seed, self.period, PATCH_OCTAVES, origin=(y0, x0))
            return smoothstep(n, -PATCH_EDGE, PATCH_EDGE)
        y, x = np.meshgrid(y0 + np.arange(h), x0 + np.arange(w), indexing="ij")
        angle = np.deg2rad(self.band_angle)
        across = -x * np.sin(angle) + y * np.cos(angle)  # distance across the stripes
        # One octave of Perlin noise is within about ±0.7; scaled so the bend reaches about one band width.
        bend = fractal_noise((h, w), self.seed, self.bend_period, 1, origin=(y0, x0))
        across = across + self.band_width * np.clip(bend / 0.7, -1, 1)
        # sin(pi * across / band_width) changes sign every band width. Near a crossing it moves by
        # pi / band_width per cell, so a step over ±pi/16 is about band_width / 8 cells wide.
        return smoothstep(np.sin(np.pi * across / self.band_width), -np.pi / 16, np.pi / 16)

    def factors(self, y0: float, x0: float, h: int, w: int) -> np.ndarray:
        """The factor on ``K`` per cell, ``contrast ** (s - 1/2)``."""
        return self.contrast ** (self.softness(y0, x0, h, w) - 0.5)


# ErodibilityMap's fields, and the run-table columns that hold them.
MAP_COLUMNS = {"shape": "map_shape", "contrast": "map_contrast", "period": "map_period", "band_width": "band_width",
               "band_angle": "band_angle", "bend_period": "period", "seed": "map_seed"}


def map_from_columns(row: dict) -> ErodibilityMap:
    """The ``ErodibilityMap`` of one run, from its run-table row (a dict of scalars)."""
    return ErodibilityMap(**{field: np.asarray(row[column]).item() for field, column in MAP_COLUMNS.items()})

