# erosion-terrain

The starting terrain: fractal noise from FastNoiseLite, at any region of an unbounded grid. Perlin,
OpenSimplex2, or cubic value noise; fBm (fractal Brownian motion) or ridged; with optional domain warp.
Each kind is rescaled to Perlin fBm's feature size and spread, so `period` means the same for all. One copy, installed by both environments:

- `datagen/` uses it to make the training runs' starting terrain, and stores each run's `Terrain`
  fields in the dataset.
- The main package reads those fields back as a `Terrain` (`erosion_sim.data.Split.terrain`), and the
  infinite world takes one as its starting terrain.

So the world starts from exactly the terrain the models were trained on. It depends only on `numpy`
and `pyfastnoiselite`, and supports Python 3.10 and later, since datagen needs 3.10.

`Terrain` holds every setting of one starting terrain, with `heights(y0, x0, h, w)` for any region:

```python
from erosion_terrain import Terrain, fbm, fractal_noise

z = Terrain(seed=0, period=32.0).heights(0, 0, 128, 128)  # heights in m

z = 64.0 * fbm((128, 128), seed=0, period=32.0, origin=(0, 0))  # heights in m, Perlin fBm
z = 64.0 * fractal_noise((128, 128), seed=0, period=64.0, noise="opensimplex2", fractal="ridged", warp=0.3)
```
