# erosion-terrain

The starting terrain: Perlin fBm (fractal Brownian motion) from FastNoiseLite, at any region of an
unbounded grid. One copy, installed by both environments:

- `datagen/` uses it to make the training runs' starting terrain.
- The main package uses it for the infinite world's starting terrain (`erosion_sim.world.terrain`).

So the world starts from exactly the terrain the models were trained on. It depends only on `numpy`
and `pyfastnoiselite`, and supports Python 3.10 and later, since datagen needs 3.10.

```python
from erosion_terrain import fbm

z = 64.0 * fbm((128, 128), seed=0, period=32.0, origin=(0, 0))  # heights in m
```
