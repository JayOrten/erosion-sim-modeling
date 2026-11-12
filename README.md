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
terrain/                 erosion_terrain: Terrain, the seeded starting terrain, shared by datagen and the world
src/erosion_sim/
  paths.py               where datasets, checkpoints, and animations live
  config.py              training profiles
  data.py                Split (one dataset split), training pairs
  normalization.py       every conversion to and from model units: terrain (fixed 64 m unit), erosion
                         parameters (to [0, 1]), height change (divided by its typical size)
  model.py               the U-Net
  diffusion.py           EDM denoiser, loss, sampler, seeded noise
  train.py               training and load_checkpoint
  predict.py             predictions, batches, and chains of calls over time
  hydrology.py           drainage area and pits
  render.py              hillshade and 3D renders
  evaluate/              metrics, chained calls, sampler sweeps, timing, and plots
  world/                 infinite world (docs/plans/infinite-plan.md): World, levels, noise,
                         level datasets, evaluation, version comparison, and a browser viewer
notebooks/
  datagen/               browse, demo, landlab-demo, parameter-ranges; kernel: datagen/.venv
  chunk/                 dataset, train, evaluate: the single-window work
  world/                 world.ipynb: evaluate one version of the infinite world in detail;
                         compare.ipynb: rank versions on world_master test samples;
                         data.ipynb, train.ipynb: generate any datagen profile, train any profiles
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

1. Generate data: `notebooks/world/data.ipynb`, for any datagen profile. Datasets go to
   `artifacts/data/<profile>/<split>/`. Open one with `Split.open(profile, split)`
   (`from erosion_sim.data import Split`): its snapshots, times, and for each sample its erosion
   parameters (`erosion(i)`) and starting terrain (`terrain(i)`).
2. Train: `notebooks/chunk/train.ipynb`, or in Python:

   ```python
   from erosion_sim.config import get_config
   from erosion_sim.train import train

   train(get_config("fixed_dt", "diffusion"))  # saves artifacts/checkpoints/diffusion-fixed_dt.pt
   ```

3. Evaluate: `notebooks/chunk/evaluate.ipynb`. Every evaluation function is in `erosion_sim.evaluate`.

For the infinite world, train the profiles `world_coarse` and `world_fine` (diffusion), then:

- `notebooks/world/world.ipynb` evaluates it.
- `notebooks/world/compare.ipynb` grades every version side by side on the same large simulator runs:
  test samples of the `world_master` dataset. Add a version with one line in its `VERSIONS`.
- `uv run python -m erosion_sim.world.viewer` serves a browser page to pan and zoom over it.
- In Python: `World.load(seed=1, dT=30_000).get(y0, x0, h, w)` returns eroded heights in m
  (`from erosion_sim.world.world import World`).

## Tests

```sh
uv run pytest
```

Tests use the `smoke` profile and tiny fake datasets, and run on CPU in seconds.
