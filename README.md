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
  config.py              training Config: task (erosion, dla), backbone, objective (regression, edm), data filters
  identity.py            model identity: config hash, model folder, finished check
  table.py               run tables (runs.npz): load, select runs by column
  data.py                Split (one dataset split), training pairs
  normalization.py       every conversion to and from model units: terrain (fixed 64 m unit), erosion
                         parameters (to [0, 1]), height change (divided by its typical size)
  model.py               the U-Net
  diffusion.py           EDM denoiser, loss, sampler, seeded noise
  train.py               training into a model folder, resume, load_checkpoint
  predict.py             predictions, batches, and chains of calls over time
  hydrology.py           drainage area and pits
  render.py              hillshade and 3D renders
  evaluate/              metrics, chained calls, sampler sweeps, timing, and plots
  dla/                   DLA (track A): simulator with sticking probability, datasets, training pairs,
                         sampling, metrics, evaluation suite, plots
  results.py             results folders: write_results, read_results
  experiments.py         experiment registry: one entry per experiment, its models and suite
  runner.py              train, evaluate, or submit a registry entry
  cli.py                 erosion-sim command: list, submit, train, evaluate, dla-generate
  world/                 infinite world (docs/plans/infinite-plan.md): World, levels, noise,
                         level datasets, evaluation, version comparison, and a browser viewer
notebooks/
  experiments/           one notebook per experiment, reading its results folder; summary.ipynb
  datagen/               browse, demo, landlab-demo, parameter-ranges; kernel: datagen/.venv
  chunk/                 dataset, train, evaluate: the single-window work
  world/                 world.ipynb: evaluate one version of the infinite world in detail;
                         compare.ipynb: rank versions on world_master test samples;
                         data.ipynb, train.ipynb: generate any datagen profile, train any profiles
slurm/                   Slurm batch scripts; submit from the repo root
logs/                    Slurm job output, gitignored
archive/                 notebooks and configs of past runs, kept for reference; not run or tested
docs/
  plans/                 numbered plans, one per line of work
  notes/                 measurements and findings
tests/                   uv run pytest
artifacts/               generated, gitignored; $EROSION_ARTIFACTS moves it
  data/<profile>/<split>/
  models/<profile>-<hash>/  one model: config.json, best.pt, last.pt, history.json, done.json
  results/<name>/        one evaluation: results.json, arrays.npz, examples.npz, manifest.json
  checkpoints/           models trained before identity.py, still read by world/
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

Everything generated goes under `artifacts/` in the repo, or under `$EROSION_ARTIFACTS` if it is set.
Both packages follow the same rule.

### Cluster

Compute nodes have no internet access, so everything is installed on the login node first.

1. On the login node, clone the repo and run both `uv sync` commands above. This also downloads the
   Python versions uv needs.
2. Set these in your shell profile on the cluster. Jobs inherit them from the shell that submits them.

   ```sh
   export EROSION_ARTIFACTS=/path/to/large/storage/erosion  # data, checkpoints, results
   export SBATCH_PARTITION=...                                # sbatch reads these two itself
   export SBATCH_ACCOUNT=...
   ```

3. From the repo root, run `sbatch slurm/check.sbatch`. It prints the GPU, torch, and CUDA versions,
   compiles a small kernel, and runs both test suites, all offline. Output goes to
   `logs/check-<job>.out`.

Jobs use the installed environments as they are. After changing dependencies, run `uv sync` on the
login node again.

Training one model:

```sh
EROSION_COMMIT=$(git rev-parse HEAD) sbatch slurm/train.sbatch <profile> [--set field=value ...]
```

The model goes to `$EROSION_ARTIFACTS/models/<profile>-<hash>/`, where the hash comes from the config
(`identity.py`). If a job dies, submitting it again resumes from the last evaluation. Rerunning a
finished model does nothing. With `EROSION_COMMIT` set, the job refuses
to run if the repo has moved or `src/` has uncommitted changes.

## Use

Every experiment in `docs/plans/experiments-plan.md` is an entry in `src/erosion_sim/experiments.py`:
its models (configs) and the evaluation suite that scores them.

1. Data. DLA: `uv run erosion-sim dla-generate dla_master`, locally (a few minutes), then copy
   `artifacts/data/dla_master/` to the cluster. Erosion: datagen jobs (implementation plan, step 8).
2. Commit, then on the login node: `erosion-sim submit A1`. It submits one training job per model that is
   not yet trained or queued, and one evaluation job that waits for them. `--dry-run` prints the
   `sbatch` commands only. `erosion-sim list` shows every experiment and the state of its models.
3. Read: `notebooks/experiments/A1.ipynb` loads `artifacts/results/A1/` and draws the tables and
   figures. `summary.ipynb` lists every result. Notebooks never train or sample.

Without Slurm, the same steps run directly: `erosion-sim train A1 base`, then
`erosion-sim evaluate A1` (`--allow-dirty` for a trial run with uncommitted changes).

The infinite world code (`world/`) still reads models trained before this setup; the implementation
plan, step 11, moves it over.

## Tests

```sh
uv run pytest
```

Tests use the `smoke` profile and tiny fake datasets, and run on CPU in seconds.
