# Plan: codebase for the experiments

## Goal

Rework the codebase so it supports every experiment in `experiments-plan.md`, and so each one can be
run on the cluster with its data, checkpoints, and results saved where they can be found and read
later.

- New capabilities the experiments need:
  - the DLA track (simulator, data, training pairs, metrics);
  - run tables with column selection;
  - every `erosion_master` group, with erodibility maps, out-of-range draws, and real terrain;
  - flow matching, DiT, and later FNO, residual diffusion, and distillation;
  - a world with any number of levels;
  - evaluation suites per track, and the baselines.
- Restructured code: one config with task, backbone, objective, and data selection; models named by
  config hash; one experiment registry; results written to disk. Old code moves to `archive/`.
- Data generation and training run as Slurm jobs, through command-line entry points.
- Notebooks only read saved results and plot them. One notebook per experiment.
- Every result can be traced to its configs, data, git commit, and Slurm job.

## Cluster setup

- **Training:** one node with 8 A100s. Each training run is its own Slurm job with one GPU, so up to 8
  run at once. The GPU time limit is about 24 hours.
- **Data generation:** CPU only, on one node. Each split runs as one job with a process pool, as
  `make_dataset` does today. Nothing is split across nodes.
- **No internet on compute nodes.** Environments are installed on the login node with `uv sync`. Jobs
  run with `uv run --no-sync`, so they never try to download.
- **Disk space** is not a constraint.

## Layout after the changes

```
src/erosion_sim/
  paths.py          artifacts root from EROSION_ARTIFACTS
  table.py          run table: load, select by column
  config.py         Config with task, backbone, objective, data selection; smoke profile only
  identity.py       model name from a hash of its config
  train.py          train(cfg), unchanged loop, new checkpoint location
  cli.py            erosion-sim: train, evaluate, submit
  experiments.py    registry: one entry per experiment, A1 to C9
  results.py        write and read artifacts/results/<experiment>/
  dla/              simulate.py, data.py, metrics.py, suite.py
  evaluate/         erosion metrics, plus suite.py
  world/            infinite world, plus suite.py
datagen/src/erosion_datagen/
  cli.py            erosion-datagen: generate
slurm/
  datagen.sbatch, train.sbatch, evaluate.sbatch
notebooks/experiments/
  A1.ipynb ... C9.ipynb, summary.ipynb
archive/
  old notebooks, plans, and code
```

Artifacts, under `$EROSION_ARTIFACTS` (the repo's `artifacts/` when unset):

```
data/<dataset>/<group>/<split>/    runs.npz, snapshots.npy (or arrivals.npy for DLA), meta.json
models/<name>-<hash>/              best.pt, last.pt, config.json, history.json, slurm-<job>.out
results/<experiment>/              results.json, arrays.npz, examples.npz, manifest.json, slurm-<job>.out
```

## Steps

### 0. Archive the old code

Move the notebooks and configs of past runs into `archive/`, for reference. Nothing there is run or
tested.

- Notebooks: `notebooks/chunk/` and `notebooks/world/` move to `archive/notebooks/`.
  `notebooks/datagen/` stays.
- Configs: copies of both `config.py` files, with every past profile, go to `archive/configs/`. The
  live configs keep only `smoke`.
- Tests that need the removed profiles or past datasets are deleted.

Check-in: the moved files, and both test suites passing.

### 1. Paths and the cluster environment

Make every path come from one root folder, and confirm the environments run on a compute node offline.

- `EROSION_ARTIFACTS` sets the artifacts root. `paths.py` reads it. `datagen/config.py` stops
  computing its own `ARTIFACTS_DIR` and uses the same variable.
- `slurm/` scripts: a header with `--gres=gpu:1` (or CPU only), `--cpus-per-task`, `--mem`, and
  `--time`. Partition and account come from the `SBATCH_PARTITION` and `SBATCH_ACCOUNT` environment
  variables, which `sbatch` reads, so the scripts hold nothing cluster-specific.
- A setup section in `README.md`: clone, `uv sync` in the repo and in `datagen/` on the login node, and
  set `EROSION_ARTIFACTS`.
- A first job, `slurm/check.sbatch`: runs `uv run --no-sync pytest` on a GPU node, and prints the GPU,
  torch, and CUDA versions.

Check-in: the check job's output from the cluster.

### 2. Run tables

Store each dataset's per-run settings as one table, and select runs by column.

- `runs.npz` replaces `params.npz`. One array per column, one row per run. Columns are exactly the
  tables in `experiments-plan.md`. Text columns (`noise`, `fractal`, `map_shape`) are stored as
  strings.
- `table.py`:
  - `load(dataset, group, split)`: the columns, plus a memory map of the snapshots, opened lazily.
  - `select(table, **filters)`: run indices matching every filter. A filter is a value
    (`fractal="fbm"`), a set of values, or a range (`index=(0, 2667)`).
- `Split` and `ErosionPairs` read through `table.py`. The selection in each experiment entry (step 6)
  is a dict of these filters.
- `make_dataset` keeps its format and its resume logic. It writes `runs.npz` instead of `params.npz`.
- One snapshot file per split, not shards. Generation runs on one node, and a memory map reads only the
  runs it touches, so a 66 GB file opens as cheaply as a small one. Shards would only pay off if
  generation used several nodes at once.

Check-in: a smoke dataset in the new format, and `select` on it with each filter type.

### 3. DLA simulator and data

Build the DLA simulator with a sticking probability, and generate `dla_master`.

- `dla/simulate.py`: copied from `../seeded-pa-vae/src/dla/simulate.py` and adapted.
  - Sticking probability `p`: a walker next to the cluster sticks with probability `p`, otherwise it
    keeps walking.
  - Output: one 128 × 128 int16 image per run, holding each cell's arrival step, −1 where empty.
  - Continuation: start from a stored arrival image instead of a single seed particle.
  - Seeding: SHA-256 of `{"dataset", "split", "index"}` as canonical JSON, seeding PCG64.
- `numba` added to the main environment.
- Measure cluster radius and time per run. Result: 128² holds 600 particles, not 1,000.
- `dla/data.py`: the `dla_master` splits from `experiments-plan.md`, including the 20 shared starts for
  test. Written as `arrivals.npy` plus `runs.npz` and `meta.json`.
- `erosion-sim dla-generate <profile> [split ...]`, run locally. The DLA data takes minutes, so it is
  not generated on the cluster; it is copied there.
- Training pairs: `DLAPairs` draws `t` and `x` per item. Input is the occupancy at `t`, conditioning is
  `log10 p` and `log10 x`, target is the occupancy at `t + x`.

Check-in: the measurements, a grid of rendered clusters per `p`, and `dla_master` generated on the
cluster.

### 4. Model identity, training command, Slurm training

Train any model from the command line, as one Slurm job, with a checkpoint location that cannot
collide.

- `Config` gains:
  - `task`: `erosion` or `dla`.
  - `backbone`: `unet` for now. `dit` and `fno` arrive in step 7 and later.
  - `objective`: `regression` or `edm` for now. `flow_matching` arrives in step 7.
  - `data`: the selection filters from step 2.
  - `conditioning`: the list of conditioning names, with `U` dropped for erosion.
  - `extra_inputs`: empty for now. Later the erodibility map and drainage area.
- `identity.py`: a model's name is its experiment-facing label plus the first 8 hex digits of the
  SHA-256 of its config as canonical JSON. Fields that do not change the trained weights
  (`num_workers`, `log_every`) are left out of the hash. Two experiments that ask for the same model
  get the same folder, so the plan's reuse rule needs no bookkeeping.
- `train.py`: same loop. Saves `best.pt`, `last.pt`, `config.json`, and `history.json` in the model
  folder. Nothing plots during training.
- `erosion-sim train <experiment> <model>`: looks up the config in the registry and trains it,
  resuming from `last.pt` if present. Exits at once if the model is already finished.
- `slurm/train.sbatch`: one GPU. It asks Slurm for a signal 10 minutes before the time limit. On that
  signal, training saves `last.pt` and exits, and the job requeues itself. The next start resumes, so
  a run longer than 24 hours just continues. A requeued job keeps its job ID and appends to its log.
- Each job checks at start that the repo is at the commit it was submitted with (passed in an
  environment variable), and fails with a clear message if not. Edit freely, then commit and resubmit.

Check-in: a smoke DLA model trained on the cluster, killed by a short time limit, and resumed to the
same result as an uninterrupted run.

### 5. Results and the DLA suite

Score a trained DLA model on every DLA metric, and save the result to disk.

- `results.py`. One folder per experiment holds:
  - `results.json`: scalars, keyed by model and test condition.
  - `arrays.npz`: curves and binned values.
  - `examples.npz`: a few inputs, samples, and simulator outputs for figures.
  - `manifest.json`: the model names and hashes, the test selection, the git commit (refused if the
    working tree has uncommitted changes to `src/`), the Slurm job ID, and the time.
- `dla/metrics.py`:
  - Particle count.
  - Connectivity: the fraction of samples forming one connected cluster.
  - Containment: the fraction of start cells kept.
  - Fractal dimension and radius of gyration, as distributions against the simulator's.
  - Copied and adapted from `../seeded-pa-vae/src/dla/evaluate.py` where it already has them.
- `dla/suite.py`: runs every metric for one experiment's models and test selection, and writes the
  results folder. Plot functions read the folder. They never recompute.
- `erosion-sim evaluate <experiment>` runs the track's suite. `slurm/evaluate.sbatch` runs it on one
  GPU.

Check-in: the DLA suite on the smoke model, its results folder, and the plots drawn from it.

### 6. Registry, launcher, notebooks

Run an experiment with one command, and view it in one notebook.

- `experiments.py`: one entry per experiment, written to match its entry in `experiments-plan.md`:
  - name, track, phase;
  - models: label and config for each;
  - test selection and suite settings (sampler steps, `p` grid, `x` values, and so on);
  - for track C, world settings.
- `erosion-sim submit <experiment>`:
  - submits one `train.sbatch` job for each model not yet finished;
  - submits `evaluate.sbatch` with a dependency on those jobs (`--dependency=afterok`);
  - prints the job IDs. `--dry-run` prints the jobs without submitting.
  - Skips finished models. A model already queued or running, for example from another experiment, is
    found by job name (the model hash) with `squeue`, and waited on instead of trained twice.
  - Refuses to run if `src/` has uncommitted changes.
- A failed training job leaves evaluation pending. Running `submit` again resubmits only what is
  missing.
- `notebooks/experiments/<name>.ipynb`: loads the results folder, shows the tables and plots, and
  holds the written conclusion. No training or simulation inside.
- `notebooks/experiments/summary.ipynb`: every `results.json` in one table.
- A smoke entry for each track, used by the tests and for checking the cluster setup.

Check-in: A1 at smoke size submitted from the login node, trained, scored, and shown in its notebook.
Then A1 for real, which also gives the timed run that sets the phase 1 budget (`experiments-plan.md`,
step 4).

### 7. Flow matching and DiT

Add the second objective and the second backbone, which A2, A3, B2, and B3 need.

- Flow matching: straight path between data and noise, logit-normal time, Euler sampler. Same
  interface as the EDM denoiser.
- DiT: patch embedding, 2D RoPE, AdaLN conditioning. Sizes S and L from `experiments-plan.md`.
- Both work for DLA and erosion, and for any input channel count.
- Smoke tests for every task, backbone, and objective combination.

Check-in: the smoke tests, and the parameter counts against the plan's.

Phase 1 track A (A1 to A6) can run after this step.

### 8. Erosion data settings

Change `datagen` so it makes every group of `erosion_master`.

- Profiles for each group: `main_256` to `main_2048`, `ood_*`, `long_*`, `real_dem_*`.
- Draw rules from the plan: all columns independent, the 12 noise combinations uniform, the map split
  2/3 uniform, 1/6 patches, 1/6 bands.
- Erodibility maps:
  - patches and bands in `erosion_terrain`, computed from the stored settings at any world
    coordinate;
  - passed to the simulator as `K × map`.
- `ood`: out-of-range draws, one parameter per run, recorded in `ood_param`.
- `real_dem`: download and pick tiles on the login node, rescale, and store them. A run takes a stored
  start terrain instead of noise.
- `erosion-datagen generate <group> <split>` and `slurm/datagen.sbatch`, one job per group and split.
  A job that hits its time limit is resubmitted and resumes.
- Measure feature size for the 12 noise combinations, and set track C's window sizes and edge margin.
- Time a few 2048² runs, and size `main_2048` from the measured cost.

Check-in: each group generated at smoke size, a few runs of each rendered, and the cost estimate redone
from a short timed run.

### 9. Generate the erosion data

The user submits the datagen jobs, in the batches listed under Cost in `experiments-plan.md`.

Check-in: every split opens, run counts and column values match the plan, disk use matches the
estimate.

### 10. Erosion suite

Score a single-window model on every erosion metric, and add the baselines.

- `evaluate/suite.py`: runs the existing metrics (height change error, skill, spectrum, fine detail,
  hydrology) for one experiment, and writes the results folder. Results are broken down by noise
  combination, map shape, parameter bins, and elapsed time, as the experiments ask.
- New: the slope-area fit.
- Baselines: no change, and the analytical stream power method (Tzathas et al. 2024).
- Existing plot functions change to read the results folder.

Check-in: the suite on a smoke erosion model, its results folder, and its plots.

Phase 1 track B (B1 to B5) can run after this step.

### 11. World suite and baselines

Score the infinite world, at any number of levels, and add the chunked baselines.

- The world generalized from two levels to any number, each 4 times coarser than the one below.
- Training from interior crops with an edge margin, for any level.
- `world/suite.py`: per-level and world metrics, seams, and consistency, written as a results folder.
- Baselines: independent closed chunks (simulator, analytical method, track B model), and overlapping
  analytical chunks blended with tent weights.

Check-in: the suite on a smoke two-level and three-level world.

Phase 2 track C (C1 to C5) can run after this step.

### 12. Phase 2 and 3 additions

Built when the experiment that needs them comes up, each with smoke tests first:

- Erodibility map as an input channel (B6).
- Drainage area input and drainage-weighted loss, if kept (B7).
- Residual diffusion (B8).
- FNO (optional arm of B3).
- Consistency distillation (B13, C9).
- Out-of-range, long, and real terrain scoring (B10, C7).
- Training to convergence with three seeds (phase 3): `seed` is a config field, so each seed is its own
  model folder.

## Open questions

- Where notebooks run: on the cluster through Jupyter, or locally after copying
  `$EROSION_ARTIFACTS/results/` with rsync. Results are small either way. Locally, the notebooks never
  need the data.
- Requeue: whether users may run `scontrol requeue` on their own jobs. The fallback is a job that
  resubmits itself with `sbatch`. Needed by step 4.
- CPU allocation for datagen: whole node or not, and how many cores. Sets `--cpus-per-task`.
- Whether the commit check is too strict. The alternative is copying the source into each job at
  submit time.
- Time limit for CPU jobs. Datagen resumes after a limit, so it only changes how often jobs are
  resubmitted.
