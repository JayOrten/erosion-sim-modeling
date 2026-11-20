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
  training/              the training loop and resume; building and loading checkpoints
  experiments/           the registry of experiments, the runner (train, evaluate, submit), results folders
  erosion/               track B: erosion metrics, the erosion suite, baselines, plots, timing
  world/                 track C: levels, World, world suite, checks, viewer
  dla/                   track A: DLA simulator, datasets, sampling, metrics, suites, plots
notebooks/experiments/   one notebook per experiment, reading its results folder; summary.ipynb;
                         make_notebooks.py writes them from the registry
slurm/                   batch scripts: check, train, evaluate, datagen
docs/plans/, docs/notes/ plans, and measurements and findings
archive/                 notebooks and configs of past runs, kept for reference; not run or tested
tests/                   uv run pytest
logs/                    Slurm job output, gitignored
artifacts/               generated, gitignored; local.json can move it
  data/<dataset>/<group>/<split>/   runs.npz, snapshots.npy (DLA: arrivals.npy), meta.json
  models/<experiment>/<label>/      config.json, best.pt, last.pt, history.json, done.json
  results/<experiment>/             results.json, arrays.npz, examples.npz, manifest.json
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

Every experiment in the experiments plan is an entry in `src/erosion_sim/experiments/registry.py`: its models
(each a `Config` it trains, or a `Ref` to another experiment's model) and the suite that scores them.

1. **Data.**
   - DLA: `uv run erosion-sim dla-generate dla_master` locally (a few minutes), then copy
     `artifacts/data/dla_master/` to the cluster.
   - Erosion: `erosion-datagen list` shows the groups. `sbatch slurm/datagen.sbatch <group>` generates
     one; submit it again if it hits the time limit, and it resumes. For experiment C8,
     `erosion-datagen chunked main_512 test --chunk 256` adds the simulator-in-chunks baseline.
2. **Run.** On the login node, `erosion-sim submit <experiment>`. It submits a training job
   per model that is neither trained nor queued, and an evaluation job that waits for them.
   `--dry-run` prints the `sbatch` commands. `erosion-sim list` shows every experiment and its models.
3. **Read.** `notebooks/experiments/<experiment>.ipynb` reads `artifacts/results/<experiment>/` and
   draws its tables and figures; `summary.ipynb` lists every result. Notebooks never train or sample.

Without Slurm the same runs directly: `erosion-sim train <experiment> <model>`, then
`erosion-sim evaluate <experiment>`.

A model trains into `models/<experiment>/<label>/`. If a job dies, running it again resumes from the last
evaluation; a finished model is skipped. If an entry is edited after its model trained, train and evaluate
stop and name the changed fields. `evaluate` will not replace existing results without `--force`.

### Records

Every dataset, model, and result says what made it. "Code version" below is the git commit and whether
`src/`, `datagen/`, or `terrain/` had uncommitted changes; it is recorded, never checked.

- **Data**, `data/<dataset>/<group>/<split>/`: `runs.npz` holds every run's parameters and seed (each run
  is a fixed function of them); `meta.json` holds the group settings, snapshot times, code version,
  package versions, and time taken.
- **Models**, `models/<experiment>/<label>/`: `config.json` (the full config and code version);
  `history.json` (loss curves, validation, seconds, and one entry per training session with its start
  step and code version, so a resume under changed code shows); `done.json` (final step, best validation
  loss, seconds); `train.log` (everything printed, every session).
- **Results**, `results/<experiment>/`: the numbers, curves, and examples; `manifest.json` (a copy of each
  scored model's config, code version, and final step and validation loss; the test data; the suite
  settings; the code version; the Slurm job; the time); `evaluate.log`.

Slurm jobs also write `slurm-<job>.out` into the model or results folder.

To look at a world, `uv run python -m erosion_sim.world.viewer --models <coarsest folder> ... <finest>`.

## Tests

```sh
uv run pytest                 # main package
cd datagen && uv run pytest   # simulator and datasets
```

Tests use tiny fake datasets and smoke settings, and run on CPU in under a minute. The `smoke_*`
experiments in the registry run every suite end to end on datagen's smoke group and `dla_smoke`.
