# erosion-sim-modeling

A research prototype for a neural surrogate of terrain erosion. A surrogate is a learned model that
approximates a slow simulator: it takes a terrain heightmap, erosion parameters, and an elapsed time,
and returns the eroded terrain without running the simulation. See `AGENTS.md` for the research
context.

There are two lines of work:

- **Single window.** One model call erodes one 128×128 terrain. Regression and diffusion models, with
  evaluation against the simulator. Plans: `docs/plans/unet-baseline-plan.md`,
  `docs/plans/diffusion-plan.md`.
- **Infinite world.** Eroded terrain for any region of an unbounded world, from a seed, with
  InfiniteDiffusion and a two-level hierarchy. Plan: `docs/plans/infinite-plan.md`. Not started.

## Layout

```
datagen/                 simulator and dataset generation (its own environment; see datagen/README.md)
terrain/                 erosion_terrain: the seeded starting terrain, shared by datagen and the world
src/erosion_sim/
  paths.py               where datasets, checkpoints, and animations live
  config.py              training profiles
  data.py                datasets, run selection, Normalizer (all scaling to and from model units)
  model.py               the U-Net
  diffusion.py           EDM denoiser, loss, sampler, seeded noise
  train.py               training and load_checkpoint
  predict.py             predictions, batches, and chains of calls over time
  hydrology.py           drainage area and pits
  render.py              hillshade and 3D renders
  evaluate/              metrics, chained calls, sampler sweeps, timing, and plots
  world/                 infinite world (docs/plans/infinite-plan.md): terrain, noise, level datasets
notebooks/
  chunk/                 dataset, train, evaluate: the single-window work
  world/                 infinite world notebooks (none yet)
docs/
  plans/                 numbered plans, one per line of work
  notes/                 measurements and findings
tests/                   uv run pytest
artifacts/               generated, gitignored
  data/<profile>/<split>/
  checkpoints/<objective>-<profile>.pt
  animations/
```

## Setup

```sh
uv sync
```

The simulator needs its own environment, because `richdem` needs Python 3.10 or older:

```sh
cd datagen && uv sync
```

## Use

1. Generate data: see `datagen/README.md`. Datasets go to `artifacts/data/`.
2. Train: `notebooks/chunk/train.ipynb`, or in Python:

   ```python
   from erosion_sim.config import get_config
   from erosion_sim.train import train

   train(get_config("fixed_dt", "diffusion"))  # saves artifacts/checkpoints/diffusion-fixed_dt.pt
   ```

3. Evaluate: `notebooks/chunk/evaluate.ipynb`. Every evaluation function is in `erosion_sim.evaluate`.

## Tests

```sh
uv run pytest
```

Tests use the `smoke` profile and tiny fake datasets, and run on CPU in seconds.
