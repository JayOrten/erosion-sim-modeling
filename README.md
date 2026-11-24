# erosion-sim-modeling

A research prototype for a neural surrogate of terrain erosion. A surrogate is a learned model that
approximates a slow simulator: it takes a terrain heightmap, erosion parameters, and an elapsed time,
and returns the eroded terrain without running the simulation. See `AGENTS.md` for the research
context.

The experiments are planned in `docs/plans/experiments-plan.md`, in three tracks:

- **Track A, DLA.** Diffusion-limited aggregation as a fast random process to test the methods on.
- **Track B, single window.** One model call erodes one closed 256² run.
- **Track C, infinite world.** Eroded terrain for any region of an unbounded world, from a stack of
  window models blended with InfiniteDiffusion.

`docs/plans/implementation-plan.md` is how the code was built for them.

## Layout

```
datagen/                 Landlab simulator and the erosion datasets (its own environment; datagen/README.md)
terrain/                 erosion_terrain: starting terrain and erodibility maps, from world coordinates
src/erosion_sim/
  paths.py               where data, models, and results live
  config.py              Config: task, backbone, objective, data filters, model and training settings
  seeding.py             random streams from public keys
  cli.py                 the erosion-sim command
  data/                  run tables, dataset splits, training pairs (ErosionPairs), normalization, hydrology
  models/                backbones: U-Net, DiT, FNO; build_backbone
  objectives/            EDM, flow matching, consistency distillation, residual diffusion; sampling for any
  training/              the training loop and resume; train a model by name; building and loading checkpoints
  records.py             log files for training and evaluation runs
  evaluation/            run a named evaluation on a model or world; results folders: write, read, combine
  erosion/               track B: erosion metrics, the erosion suite, baselines, plots, timing
  world/                 track C: levels, World, world suite, checks, viewer
  dla/                   track A: DLA simulator, datasets, sampling, metrics, suites, plots
configs/
  models/<name>.yaml     one model's training settings, every field listed
  evaluations/<name>.yaml  one evaluation: suite, test dataset, and every suite setting
  worlds/<name>.yaml     one world: its levels' models, coarsest first, and its sampler's steps and splits
notebooks/experiments/   one notebook per experiment: the commands it needs, then its analysis
slurm/                   batch scripts: check, train, evaluate, datagen, dla_generate
docs/plans/, docs/notes/ plans, and measurements and findings
archive/                 notebooks and configs of past runs, kept for reference; not run or tested
tests/                   uv run pytest
logs/                    Slurm job output, gitignored
artifacts/               generated, gitignored; local.json can move it
  data/<dataset>/<group>/<split>/   runs.npz, snapshots.npy (DLA: arrivals.npy), meta.json
  models/<model>/                   config.json, best.pt, last.pt, history.json, done.json, train.log
  results/<evaluation>/<predictor>/ results.json, arrays.npz, examples.npz, manifest.json, evaluate.log
```

## Setup

```sh
uv sync
cd datagen && uv sync   # the simulator's environment: richdem needs Python 3.10
```

Everything generated goes under `artifacts/` in the repo, for both packages. To put it somewhere else,
create `local.json` at the repo root (gitignored, one per machine):

```json
{"artifacts": "/path/to/large/storage/erosion"}
```

### Cluster

Compute nodes have no internet access, so everything is installed on the login node first.

1. On the login node, clone the repo and run both `uv sync` commands above.
2. From the repo root, `sbatch slurm/check.sbatch`. It prints the GPU, torch, and CUDA versions,
   compiles a small kernel, and runs both test suites, offline. Output: `logs/check-<job>.out`.

After changing dependencies, run `uv sync` on the login node again.

## Use

Models, evaluations, and experiments are separate:

- **Model**: a file in `configs/models/`. Trained by name into `artifacts/models/<model>/`.

- **Evaluation**: a file in `configs/evaluations/`. It fixes the suite, the test data, and the settings.
  It scores one *predictor* at a time: a trained model, or a world (`configs/worlds/`, a stack of models).
  Results go to `artifacts/results/<evaluation>/<predictor>/`. Predictors scored under the same
  evaluation can be compared.
- **Experiment**: a notebook in `notebooks/experiments/`. Its first cell lists the commands it needs;
  the rest loads results and analyzes them. The code knows nothing about experiments.

Every config file lists all of its settings; a missing or unknown field is an error. Each file is the
complete record of what it defines, and none depends on defaults in the code or on another file.

Commands, local and on the cluster (submit from the repo root):

```sh
uv run erosion-sim train <model>                     sbatch slurm/train.sbatch <model>
uv run erosion-sim evaluate <evaluation> <predictor> sbatch slurm/evaluate.sbatch <evaluation> <predictor>
uv run erosion-sim list                              # every model, world, and evaluation, with what is done
```

Data:

- DLA: `sbatch slurm/dla_generate.sbatch dla_master` on the cluster, or
  `uv run erosion-sim dla-generate dla_master` locally (a few minutes).
- Erosion: `erosion-datagen list` shows the groups. `sbatch slurm/datagen.sbatch <group>` generates one;
  submit it again if it hits the time limit, and it resumes. `erosion-datagen chunked main_512 test --chunk 256`
  adds the simulator-in-chunks baseline that `world_baselines` needs.

In a notebook, `load("<evaluation>", "<predictor>", ...)` (`erosion_sim.evaluation.results`) combines the
results of several predictors into one, for the plots.

Rules that keep the records consistent:

- **To try something new, add a file.** A model file may not change once the model has trained, and an
  evaluation file may not change once it has results: `train` and `evaluate` refuse, naming what changed.
  Copy the file under a new name instead. Fields that do not change training (`num_workers`,
  `log_every`, `compile`, `patience`) may change.
- Training resumes if a job dies; a finished model is skipped.
- `evaluate` will not replace existing results without `--force`.

### Records

Every dataset, model, and result says what made it. "Code version" below is the git commit and whether
`src/`, `datagen/`, or `terrain/` had uncommitted changes; it is recorded, never checked.

- **Data**, `data/<dataset>/<group>/<split>/`: `runs.npz` holds every run's parameters and seed (each run
  is a fixed function of them); `meta.json` holds the group settings, snapshot times, code version,
  package versions, and time taken.
- **Models**, `models/<model>/`: `config.json` (the full config and code version);
  `history.json` (loss curves, validation, seconds, and one entry per training session with its start
  step and code version, so a resume under changed code shows); `done.json` (final step, best validation
  loss, seconds); `train.log` (everything printed, every session).
- **Results**, `results/<evaluation>/<predictor>/`: the numbers, curves, and examples; `manifest.json`
  (the evaluation file's contents; the predictor, and a world's file; a copy of each model's config,
  code version, and final step and validation loss; the test data; the suite settings; the code version;
  the Slurm job; the time); `evaluate.log`.

Slurm jobs also write `logs/<job name>-<job>.out`.

To look at a world, `uv run python -m erosion_sim.world.viewer --models <coarsest folder> ... <finest>`.

## Tests

```sh
uv run pytest                 # main package
cd datagen && uv run pytest   # simulator and datasets
```

Tests use tiny fake datasets and smoke settings, and run on CPU in under a minute. The `smoke_*`
models and evaluations in `configs/` run every suite on datagen's smoke group and `dla_smoke`.
