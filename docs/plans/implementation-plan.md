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
  experiment and label; one experiment registry; results written to disk. Old code moves to `archive/`.
- Data generation and training run as Slurm jobs, through command-line entry points.
- Notebooks only read saved results and plot them. One notebook per experiment.
- Every result can be traced to its configs, data, and Slurm job.

## Cluster setup

- **Training:** one node with 8 A100s. Each training run is its own Slurm job with one GPU, so up to 8
  run at once. The GPU time limit is about 24 hours.
- **Data generation:** CPU only, on one node. Each split runs as one job with a process pool, as
  `make_dataset` does today. Nothing is split across nodes.
- **No internet on compute nodes.** Environments are installed on the login node with `uv sync`. Jobs
  run with `uv run --no-sync`, so they never try to download.
- **Disk space** is not a constraint.

## Layout after the changes

The steps below name modules by the flat paths they were built at. After step 12 the package was
regrouped into the folders below; the README lists every module.

```
src/erosion_sim/
  paths.py, config.py, seeding.py, cli.py
  data/             table.py, splits.py, pairs.py, normalization.py, hydrology.py
  models/           unet.py, dit.py, fno.py, layers.py; build_backbone
  objectives/       edm.py (was diffusion.py), flow.py, consistency.py, residual.py, sampling.py
  training/         train.py, checkpoints.py
  experiments/      registry.py (was experiments.py), runner.py, results.py
  erosion/          track B (was evaluate/): metrics.py, suite.py, plots.py, speed.py, baselines.py
  world/            track C
  dla/              track A
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
models/<experiment>/<label>/       best.pt, last.pt, config.json, history.json, done.json, slurm-<job>.out
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

### 4. Model folders, training command, Slurm training

Train any model from the command line, as one Slurm job, with a checkpoint location that cannot
collide.

- `Config` gains:
  - `task`: `erosion` or `dla`. DLA training uses `DLAPairs`, `repeats` items per run, and the target's
    standard deviation as `sigma_data`.
  - `backbone`: `unet` for now. `dit` and `fno` arrive in step 7 and later.
  - `objective`: `regression` or `edm` (renamed from `diffusion`). `flow_matching` arrives in step 7.
  - `data`: the selection filters from step 2, replacing `runs`.
  - The erosion conditioning list (dropping `U`) and extra input channels move to step 10 and step 12,
    where the erosion recipe changes.
- Model folders: `models/<experiment>/<label>/`. `config.json` records the full config, so a model always says what it was trained with. (A first version named folders by a hash
  of the config; it was dropped for plain names and explicit reuse, step 7.)
- `train.py`: same loop. Saves `config.json`, `best.pt`, `last.pt`,
  `history.json`, and `done.json` in the model folder. Resumes from `last.pt` whenever it exists.
  Nothing plots during training.
- `erosion-sim train <profile> [--set field=value ...]`: trains one model, resuming if interrupted, and
  does nothing if it is finished. Step 6 adds lookup by experiment and model name.
- `slurm/train.sbatch`: one GPU, 24 hours. No requeue logic: runs are expected to finish within the
  limit. If one dies, submitting it again resumes from `last.pt`.
- Old checkpoints (`objective = "diffusion"`, `runs`) no longer load. They needed retraining anyway.

Check-in: a smoke DLA model trained with the job script, interrupted, and resumed to the same result as
an uninterrupted run. Done locally; the cluster run waits until the cluster is available.

### 5. Results and the DLA suite

Score a trained DLA model on every DLA metric, and save the result to disk.

- `results.py`. One folder per experiment holds:
  - `results.json`: scalars, keyed by model, test condition, and metric.
  - `arrays.npz`: curves and per-sample values, keyed `<model>/<condition>/<metric>`.
  - `examples.npz`: a few start clusters, samples, and simulator outputs for figures.
  - `manifest.json`: the model folders, the test data, the conditions, the suite settings,
    the Slurm job ID, and the time.
- `dla/sample.py`: `grow` adds `x` particles to start clusters with a model, decoding at 0.5. The noise
  key per sample is `(seed, condition, index)`, the same for every model.
- `dla/metrics.py`, one value per cluster, compared as distributions (model and simulator mean and
  standard deviation, and the 1-Wasserstein distance):
  - particle count;
  - connectivity: the fraction of cells joined to the seed;
  - start kept: the fraction of start cells still occupied;
  - radius of gyration;
  - mass dimension, from cells within radius `r` of the seed;
  - radial density profile, as a curve.
- `dla/suite.py`: every metric for some models on every test condition (each fixed `p` from the seed
  particle, each shared start), plus seconds per sample, written as a results folder.
- `dla/plots.py`: metric per condition, example clusters, radial density. They read the folder only.
- `erosion-sim evaluate <profile>` runs the suite on one finished model. `slurm/evaluate.sbatch` runs
  it on one GPU. Step 6 switches it to experiment names.
- Not yet in the suite: branch and tree statistics, where growth lands, agreement across steps
  (A6), and the step sweep (A4). They are added with their experiments in step 6.

Check-in: the DLA suite on the smoke model, its results folder, and the plots drawn from it.

### 6. Registry, launcher, notebooks

Run an experiment with one command, and view it in one notebook.

- `experiments.py`: one entry per experiment: name, title, question, models (label -> `Config`), suite,
  and suite settings. Registered: A1, A5, A6, and two smoke experiments. A5 and A6 name A1's model as
  "the best so far"; the entry changes when A2 to A4 pick another. A2 to A4 are registered in step 7.
- `dla/suite.py` gains the agreement suite for A6: containment between jump sizes from one start and one
  noise field, and chaining (1, 2, 4 jumps) against the simulator. Jumps are fractions of the particles
  left after the start, so the same settings work on the smoke data.
- `runner.py`: `train_model`, `evaluate`, and `submit`. `submit`:
  - submits one `train.sbatch` job per model not yet finished, named `train-<experiment>-<label>`,
    logging to the model folder;
  - waits on a model already queued under that name, from any experiment, instead of training it twice;
  - submits `evaluate.sbatch` with `--dependency=afterok:<jobs>`, logging to the results folder, unless
    the results exist;
  - `--dry-run` prints the commands.
- A failed training job leaves evaluation pending. Running `submit` again resubmits only what is
  missing.
- `erosion-sim list | submit | train <experiment> <model> | evaluate <experiment>`.
- `notebooks/experiments/<name>.ipynb`: loads the results folder, shows the tables, plots, and training
  curves, and holds the written conclusion. `summary.ipynb`: every results folder in one list.

Check-in: the smoke experiments trained and evaluated through the command line, `submit` checked with a
fake Slurm, and every notebook run against the smoke results. Then A1 for real, which also gives the
timed run that sets the phase 1 budget (`experiments-plan.md`, step 4).

### 7. Flow matching and DiT

Add the second objective and the second backbone, which A2, A3, B2, and B3 need.

- `flow.py`: flow matching. Straight path `z_s = (1 − s) z + s ε` between the target (divided by
  `sigma_data`) and noise; the network predicts the velocity `ε − z`; `s` logit-normal in training;
  Euler sampler, one network call per step. `FlowModel` has the same inputs as the EDM `Denoiser`.
- `dit.py`: DiT with patch embedding, 2D RoPE, and AdaLN-Zero conditioning, with the same interface as
  `UNet`. `model.build_backbone` picks either; both wrappers take `backbone=`.
  - Sizes: S is 8 blocks of width 240, 4 heads (8.6 M, against the U-Net's 8.4 M); L is 12 blocks of
    width 384, 8 heads (32.5 M, against 32 M). The plan's first estimates (7 M and 30 M at widths 256
    and 448) left out the AdaLN layers.
- `sampling.py`: `generate` and `network_calls` for any objective. Prediction code calls it instead of
  checking the model's type.
- `Config`: `objective` adds `flow_matching`; `backbone` adds `dit`, with `patch`, `dit_depth`,
  `dit_width`, `dit_heads`.
- Smoke tests for every task, backbone, and objective combination (12), plus the RoPE offset property,
  the Euler sampler on Gaussian data, and the flow loss at the exact velocity.
- Registered A2 (objective), A3 (backbone), and A4 (sampler steps), with notebooks. The DLA suite's
  `step_sweep` scores each model at each step count (entries `<label>@<steps>`, with network calls), and
  `plot_steps` draws distance to the simulator against network calls.
- `DLA_BEST` in `experiments.py` names the model A5 and A6 use. Set it after A2 to A4 are read.

- Model folders by name instead of by config hash. An experiment's model is either a `Config` it trains
  into `models/<experiment>/<label>/`, or `Ref(experiment, label)` to another experiment's model, so
  reuse is written in the registry (A2's EDM arm is `Ref("A1", "base")`). Before a stored model is used
  or resumed, its `config.json` is compared with the registry; if a training setting differs, the run
  stops and names the fields. Adding a `Config` field no longer affects trained models.

Check-in: the smoke tests, and the parameter counts against the plan's.

Phase 1 track A (A1 to A6) can run after this step.

### 8. Erosion data settings (built)

- `Group` per entry in `datagen/config.py`: `main_256` to `main_2048` (bands 2 and 3 scaled × 2 and × 4),
  `ood_*`, `long_*`, and two smoke groups. `real_dem_*` is deferred: it needs elevation tiles
  downloaded and chosen first, and only B10 and C7 use it.
- Draws: every column independent, from the run's own stream (SHA-256 of dataset, group, split, index),
  so a run does not depend on its split's size. Replaces the old Latin Hypercube draw.
- Erodibility maps: `erosion_terrain.ErodibilityMap` (uniform, patches, bands), computed from world
  coordinates; the simulator uses `K × map` per cell. Columns `map_shape`, `map_contrast`, `map_period`,
  `band_width`, `band_angle`, `map_seed`.
- `ood`: `index % 3` picks the parameter out of range, recorded in `ood_param`.
- `erosion-datagen list | generate | chunked` and `slurm/datagen.sbatch`.
- Measurements (`docs/notes/datagen-measurements.md`): feature size varies 0.4 to 2.7 × the period
  across noise types (not corrected); fixed edges change drainage across a whole run (margin fixed at
  32 cells); one process does 4.8 M cell-steps per second at 256² and 1.6 M at 2048², about 300
  CPU-hours for everything.

### 9. Generate the erosion data

Not done: the user submits the datagen jobs. Time a short run on a cluster node first; the laptop's
scaling across processes was poor.

### 10. Erosion suite (built)

- `evaluate/metrics.py`: per-pair metrics (height error, erosion ratio, channel IoU, drainage area
  error, pits per km², slope-area exponent and its error against the simulator's, change spectrum), so
  every breakdown aggregates the same arrays.
- `evaluate/suite.py`: the same pairs for every model; entries for baselines, models, step sweeps, pit
  filling, and chained calls; breakdowns by noise setup, fractal, map shape, `ood_param`, `K_ref`,
  `D/K_ref`, `m`, map contrast, and elapsed time. `evaluate/plots.py` reads the folder.
- `hydrology.py` in Numba, with priority-flood pit filling and the slope-area fit.
- `baselines.py`: analytical stream power (n = 1, fixed drainage network, no hillslope diffusion), and
  the chunked and blended wrappers.
- The old `pairs`, `rollouts`, `sweeps`, and `predict` modules are gone.

### 11. World and world suite (built)

- `World` takes any stack of levels, coarsest first, each a window model (`cell`, `window`,
  `context_cell`); `check_stack` validates it. Levels use `sampling`'s phases, so EDM, flow matching,
  regression, and consistency students all work as levels.
- Level training data is `data.ErosionPairs` with `window` (the old level datasets are gone).
- `world/suite.py`: world benchmark on run interiors, levels alone with true context, baselines
  (analytical whole, in chunks, blended; a track B model in chunks; the simulator in chunks from
  `erosion-datagen chunked`), error near chunk edges, seams, and query time. Per-world sampler
  overrides for C3.
- `world/checks.py` keeps seams, order invariance, chunked-against-whole, and timing. The old
  `compare` and `evaluate` modules are gone; the viewer takes model folders.

### 12. Phase 2 and 3 additions (built)

- Map and drainage input channels (`Config.inputs`), drainage-weighted loss (`drainage_weight`),
  coarse-context noise (`context_noise`), nested data subsets (`max_runs`).
- Residual diffusion (`residual.py`, `Config.residual_of`), consistency distillation
  (`consistency.py`, `Config.teacher`), FNO (`fno.py`).
- Generalization scoring: `ood` and `long` test groups through the suite settings, chained calls.
- Every B and C experiment is registered, with a notebook (`notebooks/experiments/make_notebooks.py`).

### Built differently from the experiments plan, or not built

- B1 and C1 are split into `B1_pipeline_check` / `B1` and `C1_pipeline_check` / `C1`, since each stage
  is scored on its own filter. B10 and C7 are split by test set.
- The slope-area error is against the simulator's exponent on the same pair, not against `m`: on 50,000
  yr runs the simulator's own exponent ranges about -0.4 to 0.8 for `m` 0.3 to 0.6.
- Channel cells are the top 5% of the simulator's drainage area per pair, not over all pairs.
- C5 band 3's 64 m level uses a 16-cell window: the plan's 64 cells (4,096 m) does not fit a 2048² run.
  Band 2's 16 m level covers the whole 1024² run, so it has no edge margin.
- Not built: C5's shared stack over bands 1 to 3 (it needs a dataset over several groups); chained world
  calls in C7; B9 and C6 ablations (added once phase 2 is read); `real_dem`.
- Consistency students need an EDM teacher; if B2 picks flow matching, B13 and C9 need another
  distillation method.

## Open questions

- Where notebooks run: on the cluster through Jupyter, or locally after copying
  `$EROSION_ARTIFACTS/results/` with rsync. Results are small either way. Locally, the notebooks never
  need the data.
- CPU allocation for datagen: whole node or not, and how many cores. Sets `--cpus-per-task`.
- Time limit for CPU jobs. Datagen resumes after a limit, so it only changes how often jobs are
  resubmitted.
