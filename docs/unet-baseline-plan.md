# Plan: U-Net baseline surrogate for Landlab erosion

Status: draft. Decisions marked **Open** are not settled.

## Goal

Train a U-Net to predict eroded terrain from a starting heightmap, the erosion parameters, and an
elapsed time. A U-Net is a convolutional encoder-decoder with skip connections between matching
resolutions. It is the most common network for image-to-image tasks on a grid.

The purpose is a working baseline on one chunk. We want to learn:

- whether the U-Net can learn the erosion mapping at all;
- whether it learns a continuous notion of elapsed time;
- what breaks, and what it costs in data and compute.

This is not meant to be the final model. It is the reference that later models must beat.

## Deliverables

The end product is two notebooks to work in, backed by code in `src/erosion_sim/`:

- `notebooks/train.ipynb`: pick a profile and a subset of the dataset, train a model, watch the loss,
  and save a checkpoint.
- `notebooks/evaluate.ipynb`: load a checkpoint, run the baselines and every metric below, and show
  predictions next to the simulator.

The notebooks stay thin. They set options and call functions from the package, so the same code runs
from tests and scripts.

Also produced along the way: one large dataset, built once (see Data).

## The simulator

The simulator is the model from `datagen/notebooks/landlab-demo.ipynb`. Each step runs three Landlab
components in order:

1. `FlowAccumulator` routes water downhill (D8: each node drains to its steepest of 8 neighbors) and
   computes drainage area. Water is routed over a copy of the terrain whose pits are filled with
   `richdem` (see `docs/simulator-timing.md`).
2. `FastscapeEroder` cuts channels with the stream power law, `dz/dt = -K A^m S^n`.
3. `LinearDiffuser` smooths hillslopes, `dz/dt = D ∇²z`.

Before these, uplift raises every interior node by `U·dt`. All four grid edges are open boundaries
with fixed elevation.

Four properties of this simulator shape the plan.

**It is deterministic.** The same terrain and parameters always give the same result. So a model
trained with MSE (mean squared error) has one correct answer to learn, not an average of many. A
generative model such as flow matching is only needed if we add randomness to the simulator (for
example random storms). This baseline uses plain regression.

**Some parameter changes cancel out.** `K`, `D`, `U`, and `T` are separate inputs, and the model
takes each one. But the simulator's output only depends on three combinations of them:

- `K·T`: how much total river cutting happens.
- `D/K`: how strong smoothing is compared to cutting.
- `U/K`: how fast uplift is compared to cutting.

For example, `K = 1e-3, D = 1e-3, T = 10,000 yr` and `K = 2e-3, D = 2e-3, T = 5,000 yr` give the
same terrain. Doubling every rate and halving the time cancel out. This follows from substituting
`τ = K t` into the equation. It is exact for the equation and approximate for the simulator, because
the simulator uses fixed time steps.

This has two uses:

- It is a free test. The model should give nearly the same output for two inputs that differ only by
  such a rescaling. Nothing in training forces this.
- It helps data design. Varying `K` and `T` together partly covers the same outcomes twice. That is
  fine, but it matters when choosing sampling ranges.

**With `n = 1`, height scale drops out.** Both erosion terms are then linear in `z`, and flow
directions do not change when the terrain is scaled. So eroding `h·z0` with uplift `h·U` gives exactly
`h` times the result of eroding `z0` with uplift `U`. We can normalize every terrain by its relief
(max minus min), divide `U` by the same relief, and lose nothing. This is only true for `n = 1`.

**The state is just the heightmap.** Flow is recomputed every step from `z`, so there is no hidden
state. Any two snapshots from one run, `(z at t1, z at t2)`, form a valid training pair with elapsed
time `t2 - t1`.

## Data

**Generated once, up front.** One large, varied dataset is built before any training. Every training
stage uses a subset of it. Regenerating is expensive, so everything the dataset depends on is settled
first: the simulator settings (step 1) and the parameters and their ranges (step 2).

**What varies.** Each run draws its own values. Draft list, ranges set in step 2:

| Parameter | What it is | Sampling |
|---|---|---|
| `seed` | starting terrain | one per run |
| `period` | size of the largest terrain features, in cells | log-uniform, 32 to 256 |
| `persistence` | how strong fine detail is compared to coarse | uniform, 0.35 to 0.65 |
| `K_ref` | erodibility at `m = 0.4` | log-uniform, 1e-4 to 5e-4 |
| `D` | hillslope diffusivity | `D / K_ref` log-uniform, 0.3 to 3 |
| `m` | drainage-area exponent (channel concavity) | uniform, 0.3 to 0.6 |
| `U` | uplift rate | 0 for a third of runs; otherwise `U / K_ref` log-uniform, 0.1 to 1.5 |

The simulator uses `K = K_ref · A_ref^(0.4 − m)` with `A_ref = 30` m². This keeps the overall amount
of erosion about the same across the `m` range, so `m` changes channel shape rather than rate.
Ranges were checked in `datagen/notebooks/parameter-ranges.ipynb`.

Fixed for every run: grid 128×128, cell size 1 m, `n = 1`, `octaves = 5`, `lacunarity = 2`, terrain
height 64 m, step size `dt`, pits filled with `richdem` for routing, all edges open.

Continuous parameters are drawn with Latin Hypercube Sampling (LHS). LHS splits each parameter's range
into equal slices and puts exactly one run in each slice, so every range is covered evenly. This is
the design used by the postfire debris-flow surrogate listed in `AGENTS.md`.

**Time.** Every run has the same length, between 100 and 500 steps (settled in step 1), with a
snapshot every few steps. Training picks pairs of snapshots, so any elapsed time up to the run length
is available. `T` therefore does not need its own sampling.

**What one sample is.** A starting heightmap `z_a`, a later heightmap `z_b` from the same run,
elapsed time `ΔT = t_b - t_a`, and the run's parameters. Pairs are built at training time, not on
disk. One run gives many pairs.

**Subsets.** Each training stage picks runs by parameter value. For example, the first stage can use
only runs with `U = 0` and `K`, `D` in a narrow band. There is no separate block of runs at fixed
`K` and `D`.

**Storage.** At 128×128 in float32, one snapshot is 64 KB. A run of 500 steps with a snapshot every
10 steps is 51 snapshots, about 3.3 MB. 5,000 runs is about 16 GB. The disk has 819 GB free. Snapshots
go in one memory-mapped `.npy` file per split in `artifacts/`, so training reads them without loading
everything into memory. Parameters and snapshot times go in a small file next to it.

**Splits.** Split by run. Test terrains are never seen in training.

**Seeding.** Follow `../seeded-pa-vae`: hash a canonical JSON record (profile, split, index) with
SHA-256 and seed NumPy's PCG64 from it. One stream for terrain, one for parameters.

**Where it runs.** Generation lives in `datagen/`, a separate uv project with its own environment
(Python 3.10, numpy 1, Landlab 2.9.2, `richdem`). The main project only reads its output files. See
`datagen/README.md`.

**Speed.** One 500-step run takes about 1.9 s on one core, with `OMP_NUM_THREADS=1`. See
`docs/simulator-timing.md`.

**Parallelism.** `multiprocessing`, one run per task, across the 32 CPU cores. 10,000 runs take
about 10 minutes.

## Model

**Input.** One channel: the starting heightmap, normalized to [0, 1] by its own min and relief.

**Output.** One channel: the change in height `z_b - z_a`, divided by the same relief. Predicting
the change (a residual) rather than the full terrain is standard, because the change is small and the
starting terrain is already known. Edge nodes are forced to zero change, matching the simulator.

**Conditioning.** Each scalar (`ΔT`, `K`, `D`, and the others that vary) is log-transformed where its
range spans orders of magnitude. Each goes through sinusoidal features and a small MLP (multilayer
perceptron). The resulting vector modulates each U-Net block with FiLM (feature-wise linear
modulation): a learned per-channel scale and shift. This is the same idea as AdaLN (adaptive layer
normalization), applied to a conv net.

**Architecture.** A standard U-Net: four resolution levels, two residual conv blocks per level,
GroupNorm, SiLU activations, channels about 32-64-128-256. A few million parameters. Attention only
at the lowest resolution, if at all.

**Loss.** MSE on the normalized change. See Open decision 1.

## Baselines

Every result is compared against:

- **The simulator.** Ground truth, and the speed reference.
- **No change.** Predict `z_b = z_a`. Any useful model must beat this.
- **Diffusion only.** Apply linear diffusion analytically (a Gaussian blur) for the elapsed time.
  Cheap, ignores rivers. Shows how much of the result is just smoothing.
- **Coarse simulator** (optional). Run the simulator at half resolution and upsample. This is the
  fast non-learned competitor a reviewer would ask about.

## Evaluation

Error metrics:

- RMSE (root mean squared error) of the final heightmap and of the change.
- Skill against the no-change baseline: `1 - MSE_model / MSE_no_change`. 0 means no better than doing
  nothing. 1 is perfect.
- All of the above broken down by `ΔT` and by each parameter, to find where the model fails.

Time tests:

- **Erosion amount against time.** Total absolute height change as a function of `ΔT`, model and
  simulator on the same plot. The simplest single test of time conditioning.
- **Held-out times.** Leave some `ΔT` values out of training. Test inside the range (interpolation)
  and beyond it (extrapolation).
- **Composition.** Compare one step of `2ΔT` against two chained steps of `ΔT`. Because the model is
  deterministic, this can be compared pixel by pixel, and both against the simulator.
- **Zero time.** `ΔT = 0` should return the input.
- **Rescaling.** Multiply `K`, `D`, and `U` by a factor and divide `ΔT` by the same factor. The output
  should barely change.

Hydrology tests (run `FlowAccumulator` on the predicted terrain):

- **Channel overlap.** Mark channels as nodes with log10 drainage area above a threshold. Report IoU
  (intersection over union) between model and simulator channel masks.
- **Pits.** Count interior local minima, in the model's output and the simulator's. The simulator
  does not route pits, so its output may have some too. Compare the two counts, not against zero.
- **Power spectrum.** Radially averaged power spectrum of the change. Shows whether fine detail is lost.

Speed: model milliseconds per sample (GPU, batched) against simulator milliseconds per run.

## Code layout

```
datagen/         separate uv project, Python 3.10: simulator and dataset generation
  src/erosion_datagen/
    terrain.py   fbm terrain
    render.py    3D rendering
    config.py    Config dataclass and PROFILES (smoke, small, full)
    simulate.py  run one simulation; build and save a dataset
  scripts/generate.py      generate a profile from the command line
  notebooks/   demo, parameter ranges, Landlab demo
src/erosion_sim/
  data.py        PyTorch Dataset: load snapshots, select runs, build pairs, normalize
  model.py       U-Net with FiLM conditioning
  train.py       training loop, checkpoints to artifacts/unet-<profile>.pt
  evaluate.py    baselines and metrics; return results, plot by default
tests/           one test file per module, all on the smoke profile
notebooks/
  dataset.ipynb            step 5: look at the generated data
  train.ipynb              deliverable
  evaluate.ipynb           deliverable
```

`data.py` and `model.py` are additions to the role list in `AGENTS.md`. They keep the PyTorch code
out of `simulate.py` and `train.py`.

## Steps

### 1. Measure the simulator

Time one run at 128×128 with pit routing on. Profile it to see how the time splits between flow
routing, pit routing, incision, diffusion, and Python overhead. Estimate the cost of the full dataset
on 32 cores.

Done. Results in `docs/simulator-timing.md`. Pit routing took 99% of the time. Turning it off left
undrained basins, so pits are now filled with `richdem` in a separate environment, `datagen/`.

### 2. Choose the parameters and ranges

A notebook that sweeps each parameter on its own, holding the others at defaults, and shows the
eroded terrain. A range is good if its low end still changes the terrain visibly within a run and its
high end does not wear everything flat. Then check a few corners of the combined range, where several
parameters are at their extremes together.

Also settle the run length and snapshot spacing here, together with the `K` range, since the result
depends on `K·T`.

Check-in: the final parameter table. This is the last chance to change it before generating.

### 3. Add the config

Write `config.py` with a frozen `Config` and a `PROFILES` dict. `smoke` uses a 32×32 grid, a handful
of runs, and one epoch, and finishes in seconds on CPU. The parameter ranges from step 2 go here.

Check-in: review the fields.

### 4. Write the dataset generator, and generate

Write `simulate.py`. `run(z0, params, dt, steps, snapshot_every)` returns snapshots and times.
`make_dataset(cfg, split)` draws parameters with LHS, runs in parallel, and writes the snapshot file.
Add tests for determinism (same seed, same output) and for fixed edges.

Generate the smoke dataset, then a small one, then the full one.

Check-in: after the small one, before starting the full one.

### 5. Look at the data

A notebook that shows starting and final terrain across the parameter ranges, and how large the
change is at each `ΔT`. This decides normalization and loss weighting.

Check-in: decide normalization and loss weighting.

### 6. Add PyTorch and the data loader

Add `torch` with uv. Write `data.py`: load the snapshot file, select runs by parameter, sample pairs,
normalize, return tensors. Test shapes and that normalization round-trips.

### 7. Write the U-Net

Write `model.py`. Test output shape, and that zero conditioning still runs.

Check-in: review the architecture and parameter count.

### 8. Write training and the training notebook, and overfit a tiny set

Write `train.py`: AdamW, cosine learning rate schedule, validation each epoch, checkpoint on best
validation loss. Write `notebooks/train.ipynb` on top of it. First train on 10 examples until the
loss is near zero. If it cannot memorize 10 examples, the pipeline is broken.

Check-in: overfit result.

### 9. Train at one fixed time

Train on pairs with a single `ΔT`, on a narrow subset of runs. Confirms the U-Net can learn the
mapping before time conditioning is involved.

Check-in: skill score, and pictures against the simulator.

### 10. Train with variable time

Train on all `ΔT` values in the subset, with some held out.

Check-in: loss curves.

### 11. Write the evaluation and the evaluation notebook

Write `evaluate.py` with the baselines and every metric above. Write `notebooks/evaluate.ipynb` on
top of it.

Check-in: results.

### 12. Widen the subset

Train on more of the dataset: varying `K` and `D`, then uplift, then everything. Evaluate each.

Check-in: results per stage.

### 13. Write up

Record results, failures, and costs in `docs/`. List what the next model should address.

## Decisions

### Settled

- **Grid size.** 128×128.
- **Pit handling.** Fill pits with `richdem`, then route with `FlowAccumulator`. Generation runs in
  its own environment, `datagen/`. See `docs/simulator-timing.md`.
- **Training order.** Time only first, on a narrow subset. Then widen.
- **Dataset.** One large, varied dataset, generated once. Vary `K`, `D`, terrain `period` and
  `persistence`, and anything else worth varying. Train on subsets.
- **Varied parameters.** Terrain `seed`, `period`, `persistence`; `K`, `D`, `m`, and `U`. Exact
  ranges are set in step 2.
- **Uplift.** In the dataset, with about a third of runs at `U = 0`. The first models train on the
  `U = 0` runs only.
- **`n` stays at 1.** `n ≠ 1` breaks the height-scale property, so terrain height would also have to
  vary. It also makes `FastscapeEroder` slower, because it then solves a nonlinear equation each step.
- **No fixed-parameter block.** The first stage uses a narrow band of the varied runs.
- **Left out of this dataset.** Spatially varying `K` (it adds a second input channel), and which
  edges are open (it matters later, for chunks).
- **Dataset size.** As large as is useful. At about 1.9 s per run, 10,000 runs take about 10 minutes.
- **Hardware.** RTX 4080 Laptop GPU (12 GB), CUDA build of PyTorch. Smoke profile and tests run on CPU.

### Open

1. **Loss weighting across time.** Short elapsed times give tiny changes. Long ones give large
   changes. Plain MSE adds up squared errors, so the large changes dominate it. The model may then
   fit long times well and short times badly. Options: plain MSE; scale each target by its typical
   size at that `ΔT`; weight each example by its `ΔT`. Suggested: plain MSE first, then look at the
   error per `ΔT`. Settle at step 5. This does not affect data generation.

2. **Run length and snapshot spacing.** Between 100 and 500 steps. Settle at step 2.

3. **Parameter ranges.** Drafted in step 2; see the table under Data. Confirm before generating.
