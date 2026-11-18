# Plan: experiments for the paper

Status: draft, for discussion. Nothing built yet.

## Goal

Run the experiments for a paper on the infinite eroded world, aimed at SIGGRAPH.

- Generate all data once, as two datasets: `dla_master` and `erosion_master`.
- Make every experiment one named entry that trains and evaluates with one command.
- Run the experiments in three phases. Each phase can change the next.

This plan replaces step 9 of `infinite-plan.md`. The old `full`, `world`, and `world_master` datasets
were made before the review branch. They are deleted, not reused.

## Claims

Each experiment supports one or more of these. An experiment that supports none is cut.

1. **Accuracy.** The surrogate matches the simulator across parameters and elapsed times.
2. **Hydrological correctness.** Rivers and drainage networks come out right: connected channels, few
   false pits, and the slope-area relation the simulator's parameters imply.
3. **Consistency across chunks.** The infinite world has no seams, and it matches one large simulator
   run better than independent closed chunks do.
4. **Speed.** Much faster than the simulator. Fast enough to explore interactively.
5. **Generalization.** It handles terrain types, scales, parameters, and real terrain it was not
   trained on.
6. **Random processes.** The same recipe learns a random process, not only a deterministic one: its
   samples match the process's statistics, and samples at different times agree with each other.

## Overview

Three tracks, run in this order:

- **Track A: DLA (diffusion-limited aggregation).** A simple random process. Tests whether the recipe
  learns a distribution, not only a single answer. Claims 1, 6.
- **Track B: single window.** One model call takes a whole simulated erosion domain and returns it
  eroded. The same task as the simulator, so the comparison is direct. Claims 1, 2, 4, 5.
- **Track C: infinite world.** The InfiniteDiffusion pipeline with two or more levels, scored on large
  simulator runs at three feature scales. Claim 3, plus 1, 2, 4, 5 at world scale.

Model choices (objective, backbone, extra inputs, losses) are settled in track B, where they are cheap
to compare. Track C reuses that recipe.

**Phases**

| phase | experiments            | output                                                    |
| ----- | ---------------------- | --------------------------------------------------------- |
| 1     | A1 to A6, then B1 to B5 | DLA results; objective and backbone; training budget per run |
| 2     | B6 to B8, then C1 to C5 | the final recipe for both tracks                          |
| 3     | B9 to B13, C6 to C9   | the paper's tables and figures                            |

**How experiments are written.** Each track has a **base recipe**: the full list of model, objective,
sampler, and training settings. Every experiment entry then lists:

- **Question:** what it answers.
- **Varies:** each variable and the values it takes.
- **Fixed:** anything held fixed that differs from the base recipe. Everything else is the base recipe.
- **Models trained:** how many training runs.
- **Train / Scored on:** dataset, split, filter, and run count.
- **Reports:** the numbers and figures it produces.
- **Decides:** what the result changes later in the plan.
- **Claims:** which claims it supports.

Conventions:

- Val is always the same filter as train, on the val split.
- Normalization (`normalization.py`) is fit on the train runs.
- Every snapshot pair of every selected run is used, unless the entry says otherwise.
- **Training budget.** Phase 1 and 2 models train for a fixed budget of optimizer steps, set in step 4
  from a timed run, the same for every model in one experiment. Each keeps its best-val checkpoint, so
  models on small data are scored before they overfit.
- **Reuse.** When one arm of an experiment is a model trained earlier with the same settings, it is
  reused, not retrained. Phase 3 models train to convergence:
  validation every 1,000 steps, stop after 10 checks without improvement, keep the best checkpoint.
- **Seeds.** One training seed per model in phases 1 and 2. Three in phase 3 for the main numbers;
  ablations use one.
- Values marked *(current)* are what the code does today. Every other value is a proposal to review.

## Compute

- **Data:** this machine's CPU. 512² runs reach about 27 M cell-steps per second with 16 workers,
  1024² runs about 21 M (`infinite-plan.md` step 9a).
- **Training:** this machine's GPU, an RTX 4080 Laptop with 12 GB. An A100 cluster is available, at
  the cost of Slurm (the cluster's job scheduler) setup and data transfer. Use it only if phase 1
  shows the local GPU is too slow.
- **Disk:** 361 GB free on `/home`. The new data takes about 263 GB. The old datasets (90 GB) are
  deleted first.

---

## Part 1: DLA (track A)

### DLA data: `dla_master`

**The process.** Particles arrive one at a time, wander randomly, and stick when they touch the
cluster. One time step is one particle. A run starts from a lone seed particle at the grid center.

**Simulator.** Copied from `../seeded-pa-vae/src/dla/simulate.py` (Numba, square grid). Walkers start
on a circle just outside the cluster and jump across empty circles. One change: a **sticking
probability** `p`. A walker next to the cluster sticks with probability `p`, and otherwise keeps
walking. Below 1, walkers reach deeper before sticking, so clusters get denser and their fractal
dimension rises from about 1.71 toward 2.

**Stored output.** One 128 × 128 image per run. Each occupied cell holds its arrival step (0 for the
seed particle). Empty cells hold −1. This is only a compact way to store the run: the cluster at any
step `t` is every cell with arrival step below `t`. Every run is grown to 600 particles.

**Training pair.** A start step `t` and a count `x`, with `t + x` ≤ 600. Drawn at random during
training, so a run gives as many pairs as needed.

- Input: the occupancy at step `t` (1 occupied, 0 empty), plus `p` and `x` as conditioning.
- Target: the occupancy at step `t + x`, the cluster after `x` more particles.

`x` is the jump size, as the elapsed time `dT` is in erosion.

**Columns.** One row per run.

| column     | values                        | meaning                                                         |
| ---------- | ----------------------------- | --------------------------------------------------------------- |
| `split`    | `train`, `val`, `test`        |                                                                 |
| `index`    | 0, 1, 2, ... within the split | position in the split                                           |
| `seed`     | integer                       | SHA-256 of `{"dataset", "split", "index"}` as canonical JSON; seeds the run's PCG64 stream |
| `p`        | 0.05 to 1                     | sticking probability                                            |
| `start_id` | 0 to 19, or empty             | which shared start cluster the run continues (test only)        |

**Splits.**

| split   | runs    | `p`                                       | starts from                              |
| ------- | ------- | ----------------------------------------- | ---------------------------------------- |
| `train` | 200,000 | 0.05 to 1, log-uniform                    | the seed particle                        |
| `val`   | 2,000   | 0.05 to 1, log-uniform                    | the seed particle                        |
| `test`  | 1,000   | 0.05, 0.1, 0.2, 0.5, 1: 200 runs each     | the seed particle (`start_id` empty)     |
| `test`  | 1,000   | 0.1 for `start_id` 0 to 9, 1 for 10 to 19 | a shared start: 20 starts × 50 runs      |

Why the test split is built this way. DLA is random, so a model sample cannot match one simulator run
cell for cell. The metrics compare distributions instead: many model samples against many simulator
runs under the same conditions.

- **Fixed `p` values.** Each value has 200 simulator runs to compare against.
- **Shared starts.** For each `start_id`, one cluster is grown to 200 particles, then continued 50
  times with different seeds to 600. Each run stores its whole arrival image, start cells included.
  The model gets the same start 50 times, and its spread is compared with the simulator's.

Grid size and particle count are measured. A cluster must stay within radius 60 of the center to fit
the 128² grid. At 1,000 particles, 11% of clusters at `p` = 1 went past that. At 600, the largest of
10,000 clusters at `p` = 1 had radius 58. Cost: about 7 ms per run, a few CPU-minutes for the whole
dataset. Generated locally, not on the cluster.

### DLA evaluation

Each metric compares the distribution over model samples with the distribution over simulator runs.

- **Fractal dimension.** The number of particles within radius `r` grows like `r^D`. `D` should match
  the simulator's at the same `p`: about 1.71 at `p` = 1, higher at low `p`.
- **Radius of gyration** (root mean square distance of particles from the cluster's center of mass)
  against particle count. Grows like `N^(1/D)`.
- **Particle count.** The output has exactly `t + x` particles, and keeps every start particle.
- **Connectivity.** One piece, with no floating particles.
- **Branch and tree statistics.** Branch lengths and depths, from `../seeded-pa-vae/src/dla/evaluate.py`.
- **Where growth lands.** New particles land mostly on the outer branch tips.
- **Agreement across steps.** From one start and one noise field, sample `x1 < x2` independently. The
  cluster after `x1` should be contained in the cluster after `x2`. Also, one jump of `x` and two
  chained jumps of `x / 2` should give the same distribution.
- **Speed.** A small check, not an experiment or a claim: time per sample of the final model beside
  the simulator's, at a few `x`.

### DLA base recipe

The erosion code's U-Net and EDM sampler, on DLA images. Nothing here is built yet.

**Model**

- U-Net: the erosion code's `UNet` class, channels (32, 64, 128, 256), 1 residual block per level,
  self-attention at the bottleneck, embedding size 256. About 8.4 M parameters at 128².
- Input: 1 channel, the occupancy at step `t`.
- Conditioning vector, each scaled to about [0, 1]: `log10 p`, `log10 x`.
- Output: 1 channel, the occupancy at step `t + x`.
- No fixed edges: every cell can change.

**Target and decoding**

- Target: the occupancy at step `t + x`: 1 occupied, 0 empty.
- Decoding a sample: a cell is occupied if its value is above 0.5. The particle count metric checks
  that this gives `t + x` cells.

**Training pairs**

- `t` uniform over 1 to 540.
- `x` log-uniform over 1 to `600 − t`.
- One fresh (`t`, `x`) per item, so every epoch sees new pairs.

**Objective and sampler**

- EDM: noise levels drawn with ln σ ~ Normal(−1.2 + ln 2, 1.2), in units of `sigma_data`; loss weight
  `(σ² + σ_data²) / (σ σ_data)²` *(current, from the erosion code)*. `sigma_data` is the measured
  standard deviation of the target.
- Sampler: EDM's deterministic Heun sampler, 32 steps (63 network calls), σ from 160 to 0.004 times
  `sigma_data`, spacing ρ = 7 *(current)*.

**Training**

- AdamW, learning rate 3e-4, weight decay 1e-4, batch 32 *(current)*.
- 500 warmup steps, then cosine decay to 0 *(current)*.
- EMA (exponential moving average) of the weights with decay 0.999, used for validation and saved
  *(current)*.
- bf16 autocast and `torch.compile` *(current)*.
- Budget: the phase 1 budget, set in step 4.

### DLA experiments

Track A also serves as the first real test of the experiment runner and evaluation suite, since it
trains fast.

**A1. Base.**

- Question: does the base recipe learn DLA well enough to build on?
- Varies: nothing.
- Models trained: 1.
- Train: `dla_master` train, all 200,000.
- Scored on: test, all 2,000.
- Reports: every DLA metric, at each test `p` and from each shared start, beside the simulator's.
- Decides: if clusters come out broken (disconnected, wrong count), fix the decoding or the target's scaling before
  A2.
- Claims: 1, 6.

**A2. Objective.**

- Question: does flow matching beat EDM on a random target?
- Varies: objective ∈ {EDM, flow matching}.
  - Flow matching: straight path `y_s = (1 − s) y + s ε`; the network predicts the velocity `ε − y`;
    `s` drawn from a logit-normal distribution (mean 0, std 1); Euler sampler, 32 steps (32 network
    calls).
- Models trained: 1 (flow matching). The EDM arm is A1's model.
- Data: as A1.
- Reports: every DLA metric for both, at 32 steps.
- Decides: the objective for the rest of track A, and a first read for track B.
- Claims: 1, 6.

**A3. Backbone.**

- Question: does a transformer beat the U-Net on DLA?
- Varies: backbone ∈ {U-Net, DiT}, as in B3.
  - DiT (Diffusion Transformer): patch size 4 (32 × 32 = 1,024 tokens at 128²), 2D RoPE (rotary
    position embeddings), AdaLN conditioning (layer norm scaled and shifted by the conditioning), 8
    layers, width 240, 4 heads. About 8.6 M parameters, near the U-Net's 8.4 M.
- Fixed: objective from A2.
- Models trained: 1 (DiT). The U-Net arm is A2's winning model.
- Data: as A1.
- Reports: every DLA metric for both; training time per step.
- Claims: 1, 6.

**A4. Sampler steps.**

- Question: how few sampler steps are enough?
- Varies: steps ∈ {4, 8, 16, 32, 64}.
- Models: A2's two models. No training.
- Scored on: as A1.
- Reports: each metric against network calls per sample, and against wall time.
- Decides: the step count for A5 and A6.
- Claims: 1.

**A5. Sticking probability.**

- Question: does the model follow `p`, out to the ends of its range?
- Varies: `p` ∈ {0.05, 0.1, 0.2, 0.5, 1} (the test grid).
- Model: the best of A1 to A4. No training.
- Scored on: test, `start_id` empty (1,000), 200 samples per `p` from the seed particle.
- Reports: fractal dimension and radius of gyration against `p`, model beside simulator, as mean and
  spread.
- Claims: 1.

**A6. Agreement across steps.**

- Question: are samples at different `x` from one start consistent with each other?
- Varies:
  - Containment: `x` ∈ {25, 50, 100, 200, 400} from each shared start (`t` = 200), all with the same
    starting noise. Check every pair `x1 < x2`.
  - Chaining: reach `t` = 600 from `t` = 200 as 1 jump of 400, 2 jumps of 200, 4 jumps of 100.
- Model: the best of A1 to A4. No training.
- Scored on: test, `start_id` set (1,000): 50 samples per start, against the simulator's 50
  continuations. Containment also from 100 clusters in test, `start_id` empty, at `t` = 200.
- Reports: fraction of cells from the `x1` sample missing in the `x2` sample; distribution metrics of
  each chaining scheme against the simulator.
- Claims: 6.

---

## Part 2: Erosion (tracks B and C)

### Erosion data: `erosion_master`

One table. One row is one simulator run. A **group** is a block of rows made with the same generator
settings (grid size, ranges, run counts). Experiments select rows by group, split, and filter.

Old names, for reading `infinite-plan.md`: `master_closed` → `main_256`, `master` → `main_512`,
`master_large` → `main_1024`, `master_huge` → `main_2048`, `_closed` test sets → `_256` groups.

#### One run

1. Build a starting terrain on a square grid of 1 m cells: fractal noise from FastNoiseLite
   (`erosion_terrain.fractal_noise`), or a real elevation tile (`real_dem` groups only).
2. Build an erodibility map: a factor per cell on `K`.
3. Run the Landlab simulator in steps of `dt` = 100 yr. Each step: route water with D8 (each cell
   drains to its steepest of 8 neighbors) over the pit-filled surface; cut channels with the stream
   power law (`FastscapeEroder`, `n` = 1, erodibility `K × map`); diffuse hillslopes
   (`LinearDiffuser`, diffusivity `D`). No uplift.
4. All four edges stay fixed at their starting height. Track B models see the whole grid, edges
   included. Track C models see only windows cropped away from the edges.
5. Save the heights every `snapshot_every` steps, starting with the start terrain.

**Training pair.** Two snapshots `(i, j)` of one run with `i < j`. Input: snapshot `i`. Target:
snapshot `j`. Elapsed time: `times[j] − times[i]`. A run with `S` snapshots gives `S × (S − 1) / 2`
pairs.

#### Columns

**Identity**

| column  | values                             | meaning                                         |
| ------- | ---------------------------------- | ----------------------------------------------- |
| `group` | see the group tables               | which block of rows the run belongs to          |
| `split` | `train`, `val`, `test`             | test-only groups have only `test`               |
| `index` | 0, 1, 2, ... within (group, split) | position in the split                           |
| `seed`  | integer                            | SHA-256 of `{"dataset", "group", "split", "index"}` as canonical JSON; seeds the run's PCG64 streams |

**Starting terrain** (empty for `real_dem` runs, which have `dem_tile` instead: the tile's ID)

| column        | range                                   | distribution                  |
| ------------- | --------------------------------------- | ----------------------------- |
| `period`      | per group                               | log-uniform                   |
| `noise`       | `perlin`, `opensimplex2`, `value_cubic` | see "How settings are drawn"  |
| `fractal`     | `fbm`, `ridged`                         | see "How settings are drawn"  |
| `warp`        | 0 (off), or 0.1 to 0.5                  | see "How settings are drawn"; uniform when on |
| `octaves`     | 4 to 7, both ends included              | uniform integer               |
| `persistence` | 0.35 to 0.65                            | uniform                       |
| `lacunarity`  | 1.8 to 2.4                              | uniform                       |
| `height`      | per group, m per unit of noise          | log-uniform                   |

`noise`, `fractal`, and warp on or off are the three discrete noise choices: 3 × 2 × 2 = 12
combinations. fBm is fractional Brownian motion: octaves summed. Ridged folds each octave at zero,
which gives sharp ridges and wide valleys.

**Erosion parameters**

| column      | range                                                                 | distribution |
| ----------- | --------------------------------------------------------------------- | ------------ |
| `K_ref`     | 1e-4 to 5e-4                                                          | log-uniform  |
| `D / K_ref` | 0.3 to 3 (stored as `D`)                                              | log-uniform  |
| `m`         | 0.3 to 0.6                                                            | uniform      |
| `K`         | derived: `K_ref × A_ref^(m_ref − m)`, `m_ref` = 0.4, `A_ref` = 30 m²  | —            |
| `U`         | 0                                                                     | —            |
| `ood_param` | `K_ref`, `D_over_K`, `m`, or empty                                    | `ood` groups only: which parameter is out of range |

**Erodibility map.** The simulator uses `K × map`. The map is computed from world coordinates and
these columns, so the world can compute it anywhere. It is never stored as an array. `D` is
uniform in every run.

| column         | values                                           | distribution                    |
| -------------- | ------------------------------------------------ | ------------------------------- |
| `map_shape`    | `uniform`, `patches`, `bands`                    | see "How settings are drawn"    |
| `map_contrast` | 1 to 10: softest rock's rate ÷ hardest rock's    | log-uniform; 1 for uniform maps |
| `map_period`   | patches only, per group                          | log-uniform                     |
| `band_width`   | bands only, per group                            | log-uniform                     |
| `band_angle`   | bands only, 0 to 180 degrees                     | uniform                         |

How a map is computed:

- Each shape gives a field `s` from 0 (hardest) to 1 (softest). The map is `map_contrast ^ (s − 1/2)`.
  So the softest cell erodes `map_contrast` times faster than the hardest, and the geometric mean is
  about 1, which keeps `K_ref` as the overall rate.
- **Uniform:** `s` = 1/2 everywhere, so the map is 1.
- **Patches:** Perlin fBm at `map_period`, 3 octaves, through a smooth step so most cells are near 0
  or 1. Stands for intrusions and mixed rock.
- **Bands:** stripes `band_width` cells wide along `band_angle`, alternating 0 and 1, with a smooth
  transition `band_width / 8` wide. Each stripe is bent sideways by Perlin noise at the terrain's
  period, by up to one `band_width`. Stands for tilted rock layers.

**Stored output**

| field       | shape                                       | type    |
| ----------- | ------------------------------------------- | ------- |
| `snapshots` | (runs, snapshots, grid, grid), heights in m | float32 |
| `times`     | (snapshots,), years since the start         | float   |

#### How settings are drawn

In the main groups, every column is drawn independently from its distribution:

- `noise`, `fractal`, and warp: each uniform, so all 12 combinations are equally likely.
- `map_shape`: 2/3 uniform, 1/6 patches, 1/6 bands.
- Every other column from its range.

Most runs have a uniform map, so most of the data varies the noise without erodibility maps.

| rows with                                  | expected share         |
| ------------------------------------------ | ---------------------- |
| `map_shape = uniform`                      | 2/3 ≈ 66.7%            |
| `map_shape = patches`                      | 1/6 ≈ 16.7%            |
| `map_shape = bands`                        | 1/6 ≈ 16.7%            |
| `fractal = ridged`                         | 1/2 = 50%              |
| Perlin, fBm, no warp, uniform map          | 1/12 × 2/3 ≈ 5.6%      |

The last row is the **pipeline check subset**: one fixed noise setup, no maps. It is used to verify
that things work (B1, C1) before training on varied noise.

#### Main groups

Trained on and tested on. Each is one scale band. All run 500 steps (50,000 yr).

| group       | grid  | period     | height     | `map_period` | `band_width` | snapshot every | snapshots | pairs per run | runs (train / val / test) |
| ----------- | ----- | ---------- | ---------- | ------------ | ------------ | -------------- | --------- | ------------- | ------------------------- |
| `main_256`  | 256²  | 32 to 64   | 32 to 128  | 32 to 128    | 8 to 64      | 2,500 yr       | 21        | 210           | 4,000 / 400 / 400         |
| `main_512`  | 512²  | 32 to 64   | 32 to 128  | 32 to 128    | 8 to 64      | 2,500 yr       | 21        | 210           | 3,000 / 300 / 300         |
| `main_1024` | 1024² | 64 to 128  | 64 to 256  | 64 to 256    | 16 to 128    | 5,000 yr       | 11        | 55            | 1,200 / 150 / 150         |
| `main_2048` | 2048² | 128 to 256 | 128 to 512 | 128 to 512   | 32 to 256    | 5,000 yr       | 11        | 55            | 300 / 50 / 50             |

Period, `map_period`, and `band_width` are in cells; height in m per unit of noise.

- `main_256` is track B's data. `main_512`, `main_1024`, and `main_2048` are track C's bands 1, 2,
  and 3.
- In bands 2 and 3, height and map sizes are band 1's ranges × 2 and × 4. Slopes and map shapes
  relative to the terrain then stay the same as features grow.
- Bands 2 and 3 keep fewer snapshots, to save disk.
- Track C's top level sees about 4 periods per window. At period 256 that is about 1,024 cells, so
  band 3 needs 2048² to leave room for crops away from the edges.

Expected rows per subset. Real counts vary by a few percent.

| group       | split | all   | uniform map | Perlin, fBm, no warp, uniform map | `patches` | `bands` | `ridged` | `fbm` |
| ----------- | ----- | ----- | ----------- | --------------------------------- | --------- | ------- | -------- | ----- |
| `main_256`  | train | 4,000 | 2,667       | 222                               | 667       | 667     | 2,000    | 2,000 |
| `main_256`  | val   | 400   | 267         | 22                                | 67        | 67      | 200      | 200   |
| `main_256`  | test  | 400   | 267         | 22                                | 67        | 67      | 200      | 200   |
| `main_512`  | train | 3,000 | 2,000       | 167                               | 500       | 500     | 1,500    | 1,500 |
| `main_512`  | val   | 300   | 200         | 17                                | 50        | 50      | 150      | 150   |
| `main_512`  | test  | 300   | 200         | 17                                | 50        | 50      | 150      | 150   |
| `main_1024` | train | 1,200 | 800         | 67                                | 200       | 200     | 600      | 600   |
| `main_1024` | val   | 150   | 100         | 8                                 | 25        | 25      | 75       | 75    |
| `main_1024` | test  | 150   | 100         | 8                                 | 25        | 25      | 75       | 75    |
| `main_2048` | train | 300   | 200         | 17                                | 50        | 50      | 150      | 150   |
| `main_2048` | val   | 50    | 33          | 3                                 | 8         | 8       | 25       | 25    |
| `main_2048` | test  | 50    | 33          | 3                                 | 8         | 8       | 25       | 25    |

#### Test-only groups

Never trained on. Each exists at 256² (track B) and 512² (track C), with the period range 32 to 64.
All rows are `split = test`. Each starts from Perlin, fBm, no warp, a uniform map, and in-range
parameters, and changes one thing, so a score drop points at that one thing.

| group                          | runs per size | what changes                          | steps / snapshot every / snapshots |
| ------------------------------ | ------------- | ------------------------------------- | ---------------------------------- |
| `ood_256`, `ood_512`           | 200           | one erosion parameter out of range    | 500 / 2,500 yr / 21                |
| `long_256`, `long_512`         | 200           | 150,000 yr, three times the training  | 1,500 / 5,000 yr / 31              |
| `real_dem_256`, `real_dem_512` | 64            | real elevation as the starting terrain | 500 / 2,500 yr / 21               |

- **`ood`.** `index % 3` picks the out-of-range parameter: 0 → `K_ref` 5e-4 to 1e-3 (log-uniform);
  1 → `D / K_ref` 3 to 6 (log-uniform); 2 → `m` 0.6 to 0.7 (uniform). All above the training range.
- **`real_dem`.** Tiles from Copernicus GLO-30 (a free 30 m global elevation model, on AWS open data),
  from varied regions. Each tile is resampled so its dominant feature size, measured as in step 1,
  falls in 32 to 64 cells. Heights are shifted to mean 0 and scaled so their standard deviation
  matches a Perlin fBm noise terrain with `height` from 32 to 128. Real terrain is already eroded,
  so this is a hard test.

#### Cost

Times assume 27 M cell-steps per second at 256² and 512², and 21 M at 1024² and 2048². Only 512² and
1024² are measured.

| group                          | runs  | CPU time  | disk        |
| ------------------------------ | ----- | --------- | ----------- |
| `main_256`                     | 4,800 | ~1.6 h    | ~26 GB      |
| `main_512`                     | 3,600 | ~4.9 h    | ~79 GB      |
| `main_1024`                    | 1,500 | ~10.4 h   | ~69 GB      |
| `main_2048`                    | 400   | ~11 h     | ~74 GB      |
| `ood_256`, `ood_512`           | 400   | ~0.3 h    | ~6 GB       |
| `long_256`, `long_512`         | 400   | ~1 h      | ~8 GB       |
| `real_dem_256`, `real_dem_512` | 128   | ~0.1 h    | ~2 GB       |
| **total**                      |       | **~30 h** | **~263 GB** |

Two or three overnight batches: everything except `main_1024` and `main_2048` first, then those two.
If CPU time is the bottleneck, the cluster's CPUs could run `main_2048`.

### Erosion evaluation

Every experiment runs its track's evaluation suite and writes the same results file. Metrics are fixed
before phase 1, so results stay comparable across phases.

**Metrics on a terrain**, used by both tracks on whatever region is scored:

- **Height error:** RMSE (root mean square error) of `Δz` in m, and skill against the no-change
  baseline.
- **Detail:** radial power spectrum against the simulator's (`spectrum_comparison`, `fine_detail`).
- **Hydrology:** channel overlap (IoU, intersection over union, of high drainage area cells), pits per
  km², drainage area distribution (`hydrology_metrics`).
- **Slope-area fit.** The stream power law predicts that in steady channels, slope scales with
  drainage area to the power `−m` (Flint's law). Fit the exponent on the output and compare it to the
  `m` the model was given. New.
- **Sample spread**, for sampled models (`sample_spread`).
- **Error by parameter and elapsed time** (`error_by_condition`, `error_by_elapsed`).

**What gets scored**

- **Track B:** whole test runs. The model sees the whole run, and every cell is scored.
- **Track C, per level:** each level on interior windows of test runs, with the true coarse change as
  context for the fine level. Separates the levels' own errors from the pipeline's.
- **Track C, world benchmark:** run the world over the start terrain of each test run, and score it
  against that run's snapshots on the interior. The edge margin and window sizes per band are set in
  step 1.
- **Track C, seams:** disagreement between neighboring windows before blending (`window_disagreement`).

**Speed**

- Track B: time per 256² window, batched and unbatched, on the 4080.
- Track C: time per 512² query, cold and warm cache.
- The simulator's time for the same region and elapsed time.

**Baselines**

- **No change:** the start terrain.
- **Regression:** trained in B2.
- **Analytical stream power** (Tzathas et al. 2024). A fast non-learned method that solves the same
  equations exactly, on a bounded domain. Built in phase 2, before track C.
- **Track C only:** every bounded-domain method, made infinite the only ways it can be.
  - *Independent chunks, stitched:* the simulator, the analytical method, and the track B model, each
    on separate closed chunks. Each chunk's fixed edges become false outlets.
  - *Overlapping chunks, blended:* the analytical method on chunks at half-chunk stride, averaged with
    tent weights as in InfiniteDiffusion. Hides seams, but each chunk still decides its own rivers, so
    where neighbors disagree the average gives two half-depth rivers. Isolates what blending partway
    through sampling adds over blending finished results.

### Track B base recipe

The existing single-window diffusion model (`ErosionPairs`, `diffusion.py`), at 256².

**Model**

- U-Net: channels (32, 64, 128, 256), 1 residual block per level, self-attention at the bottleneck,
  embedding size 256 *(current)*. About 8.4 M parameters.
- Input: 1 channel, the start heights, normalized *(current)*.
- Conditioning vector, each scaled to about [0, 1]: `log10 dT`, `log10 K`, `log10 D`, `m`
  *(current, minus `U`)*. `U` is dropped, since it is 0 in every run.
- Output: 1 channel, the normalized height change.
- Fixed edges: the output is set to 0 on the edge cells, as the simulator holds them *(current)*.

**Target**

- The height change `z_j − z_i` in m, divided by `typical_change(K_ref, dT)`, a power law fit on the
  train runs *(current)*. So every pair's target is about size 1.

**Objective and sampler** *(current)*

- EDM: ln σ ~ Normal(−1.2 + ln 2, 1.2), in units of `sigma_data`; loss weight
  `(σ² + σ_data²) / (σ σ_data)²`. `sigma_data` is the measured RMS of the target.
- Sampler: EDM's deterministic Heun sampler, 32 steps (63 network calls), σ from 160 to 0.004 times
  `sigma_data`, spacing ρ = 7.

**Training** *(current)*

- AdamW, learning rate 3e-4, weight decay 1e-4, batch 32.
- 500 warmup steps, then cosine decay to 0.
- EMA of the weights with decay 0.999.
- bf16 autocast and `torch.compile`.
- Budget: the phase 1 budget, set in step 4.

### Track B experiments: single window

All track B data is whole runs from `main_256` or a `_256` test-only group.

**B1. Base.**

- Question: does the current model work on the new data, first on one noise setup, then on varied
  noise?
- Varies: training data, in two stages:
  - Pipeline check: Perlin, fBm, no warp, uniform map. If this fails, fix it before going on.
  - Base: uniform map, all noise settings. The reference for every phase 1 experiment.
- Models trained: 2.
- Train, pipeline check: `main_256` train, Perlin, fBm, no warp, uniform map (222).
- Train, base: `main_256` train, uniform map (2,667).
- Scored on: the same filter on `main_256` test (22 and 267).
- Reports: every terrain metric; error by elapsed time and by parameter. The base is also reported per
  noise setting.
- Decides: the reference for every phase 1 experiment.
- Claims: 1, 2, 4.

**B2. Objective.**

- Question: what does sampling buy over regression, and does the diffusion formulation matter?
- Varies: objective ∈ {regression, EDM, flow matching}.
  - Regression: one forward pass, MSE (mean squared error) on the normalized change *(current
    option)*.
  - Flow matching: as in A2. Euler sampler, 32 steps.
- Models trained: 2 (regression, flow matching). The EDM arm is B1's base model.
- Data: as B1.
- Reports: every terrain metric for each, at 32 steps. Regression shows what sampling buys in detail
  and channel sharpness.
- Decides: the objective for everything after.
- Claims: 1, 2, 4.

**B3. Backbone.**

- Question: does attention over the whole window help rivers, since drainage depends on distant
  terrain?
- Varies: backbone and size ∈ {U-Net S, U-Net L, DiT S, DiT L}.
  - U-Net S: the base, 8.4 M parameters.
  - U-Net L: channels (64, 128, 256, 512), about 32 M parameters.
  - DiT S: patch size 8 (32 × 32 = 1,024 tokens at 256²), 2D RoPE, AdaLN conditioning, 8 layers,
    width 240, 4 heads. About 8.6 M parameters.
  - DiT L: as DiT S, 12 layers, width 384, 8 heads. About 32.5 M parameters.
  - Widths are set so each DiT matches its U-Net's parameter count. The AdaLN layers (6 × width² per
    block) are a third of a DiT's parameters.
  - Optional: FNO (Fourier Neural Operator) as a regression model, 4 layers, width 64, 32 modes, since
    the closest prior work on landscape evolution used it.
- Position encoding must be relative (RoPE), so the same model can run as a world level, where a
  window must behave the same anywhere.
- Fixed: objective from B2. Same steps and batch for all, so compute differs by model; training FLOPs
  (floating point operations) and wall time are reported.
- Models trained: 3 (U-Net L, DiT S, DiT L), 4 with FNO. The U-Net S arm is B2's winning model.
- Data: as B1.
- Reports: every terrain metric; parameters, training FLOPs, wall time, sampling time.
- Decides: the backbone and size for everything after.
- Claims: 1, 2, 4.

**B4. Data scaling.**

- Question: does more data still help? Decides how to spend phase 2.
- Varies: training runs ∈ {333, 667, 1,333, 2,667}: the first `n` uniform-map runs of `main_256`
  train by `index` (nested). 2,667 is all of them.
- Fixed: recipe from B2 and B3.
- Models trained: 3 (333, 667, 1,333). The 2,667 point is B3's winning model.
- Scored on: as B1.
- Reports: each metric against training runs, on a log axis.
- Decides: whether phase 2 should add data or model capacity.
- Claims: 1.

**B5. Sampler steps.**

- Question: how few steps are enough?
- Varies: steps ∈ {4, 8, 16, 32, 64}.
- Models: B2's EDM and flow matching models. No training.
- Scored on: as B1.
- Reports: each metric against network calls, and against wall time. Regression as a one-call point.
- Decides: the step count for everything after.
- Claims: 4.

**B6. Erodibility maps.**

- Question: does the model need to see the map, and do hard bands produce the ridges and steps the
  simulator makes?
- Varies: map input ∈ {on, off}.
  - On: `log10(map)` as a second input channel, divided by `log10(10)` so it lies in [−0.5, 0.5].
  - Off: the model sees only `K_ref`, as in the base.
- Models trained: 2.
- Train: `main_256` train, all 4,000.
- Scored on: `main_256` test with `map_shape` = `patches` or `bands` (150), by shape and by
  `map_contrast` in 4 log-spaced bins over 1 to 10. Uniform-map test runs (250) as a check that nothing
  got worse.
- Reports: each metric per shape and contrast bin; rendered examples of bands.
- Decides: whether the map is an input from here on.
- Claims: 1, 2.

**B7. Hydrology inputs and losses.**

- Question: which change makes rivers most correct? The set to run is picked after phase 1 shows how
  far the rivers are off.
- Varies, one at a time against the base:
  - Drainage area input: `log10` of the start terrain's drainage area (D8 over the pit-filled
    surface), normalized, as an extra input channel. In track C it depends on terrain outside the
    window, so a world level can only use a bounded neighborhood.
  - Drainage-weighted loss: per-cell weight `(A / A_mean)^β`, normalized to mean 1 per window, with
    β ∈ {0.25, 0.5}.
  - Pit filling: fill pits in the output as post-processing (priority flood). No training. Applied to
    the base and to each model above. If it closes most of the gap, the paper should say so.
- Models trained: 3 (input; loss at β = 0.25; loss at β = 0.5), plus 1 combining the two best.
- Data: as B1.
- Reports: hydrology metrics and slope-area fit for each, with and without pit filling.
- Decides: which hydrology changes go into the final recipe.
- Claims: 2.

**B8. Residual diffusion (optional).**

- Question: does splitting the change into a mean and a residual help? Worth running if B2 shows
  regression is accurate at large scales but blurry.
- Setup: B2's regression model predicts the mean change. A diffusion model samples the residual
  (true change minus the regression output), normalized by the residual's own RMS. The two outputs are
  added.
- Varies: diffusion steps ∈ {4, 8, 16, 32}.
- Models trained: 1 (the residual model; the regression model is reused).
- Data: as B1.
- Reports: each metric against network calls, beside B5's curve.
- Decides: whether the final recipe is residual.
- Claims: 1, 4.

**B9. Final single-window model.**

- Question: the headline track B numbers.
- Recipe: B2 and B3's choices, plus every phase 2 change that helped (B6, B7, B8).
- Training: to convergence (phase 3 rule). Three seeds.
- Ablations, one seed each: each phase 2 change removed, one at a time.
- Models trained: 3, plus 1 per ablation.
- Train: `main_256` train, all 4,000.
- Scored on: `main_256` test, all 400.
- Reports: every terrain metric, mean and spread over seeds; the ablation table.
- Claims: 1, 2, 4.

**B10. Generalization.**

- Question: what does the model handle that it was not trained on?
- Held-out fractal type:
  - Models trained: 1, B9's recipe.
  - Train: `main_256` train, `fractal = fbm` (2,000).
  - Scored on: `main_256` test, `fractal = ridged` (200), against B9 on the same runs.
- Test-only groups, with B9's model, no training:
  - `ood_256`: parameters out of range. Reported per `ood_param`.
  - `long_256`: elapsed times beyond training. As one call (`dT` up to 150,000 yr) and as chained
    calls: 3 × 50,000, 6 × 25,000, 15 × 10,000 yr.
  - `real_dem_256`: real terrain.
- Reports: every terrain metric per test, beside B9's score on `main_256` test.
- Claims: 5.

**B11. Parameters and time scales.**

- Question: where in parameter space is the model good or bad?
- Varies: nothing; results are binned.
  - `K_ref`, `D / K_ref`, `m`, `map_contrast`: 4 bins each, evenly spaced on each one's sampling scale.
  - Elapsed time: every value, 2,500 to 50,000 yr.
- Model: B9's. No training.
- Scored on: `main_256` test, all 400.
- Reports: figures of error and hydrology against each binned variable.
- Claims: 1, 2.

**B12. Baselines.**

- Question: how does the final model compare with simpler methods?
- Compared: B9's model; regression trained with B9's recipe and data; analytical stream power
  (Tzathas et al. 2024) with each run's `K`, `m`, and map; no change.
- Models trained: 1 (the regression model).
- Scored on: `main_256` test, all 400.
- Reports: every terrain metric per method.
- Claims: 1, 2.

**B13. Speed.**

- Question: how fast can the final model be, and what does speed cost in accuracy?
- Varies:
  - Sampler steps ∈ {B5's choice, 8, 4}.
  - Consistency distillation: a student trained from B9's model to go from noise to the result in
    1 or 2 steps.
  - Batch ∈ {1, 16}.
- Models trained: 1 (the student), on B9's data.
- Scored on: `main_256` test, all 400, timed. Model on the 4080; simulator and analytical method on
  1 core and on 16 workers, for the same 256² region and elapsed time.
- Reports: a latency table against accuracy, beside the simulator and the analytical method.
- Claims: 4.

### Track C base recipe

The existing two-level world (`world/`), with the track B recipe from phase 1 for each level.

**Levels**

- Coarse level: 4 m cells, the 4 × 4 block means of the terrain and its change *(current)*. Window 64
  coarse cells (256 m, about 4 periods at period 64).
- Fine level: 1 m cells. Window 128 cells (about 2 periods). A second input channel holds the coarse
  level's change, upsampled to 1 m *(current)*.
- Window sizes follow `infinite-plan.md`: coarse about 4 periods, fine about 2. Step 1's feature size
  measurement confirms or changes them.
- Each level: the track B backbone, objective, and conditioning from phase 1. No fixed edges: a window
  edge is not a boundary *(current)*.

**Training data**

- Windows cropped from run interiors, at least the edge margin away from the fixed edges. The margin
  is set in step 1.
- Each training item: one window at a random interior position, randomly flipped *(current)*.
- The fine level reads the true coarse change from the same run.

**Sampling**

- Windows at a stride of half a window, blended with tent weights *(current)*.
- T = 2 blend phases, with the split at 5 × `sigma_data` *(current)*.
- 32 sampler steps per level *(current)*, or track B's choice once B5 is done.

**Training:** as the track B base, per level.

### Track C experiments: infinite world

**Scale is a main axis**: every main result is measured at three bands.

| band | period (cells) | group       |
| ---- | -------------- | ----------- |
| 1    | 32 to 64       | `main_512`  |
| 2    | 64 to 128      | `main_1024` |
| 3    | 128 to 256     | `main_2048` |

**C1. Base world.**

- Question: does the two-level world work at band 1 with the phase 1 recipe?
- Varies: training data, in two stages, as in B1:
  - Pipeline check: Perlin, fBm, no warp, uniform map.
  - Base: uniform map, all noise settings. The reference for every track C experiment.
- Models trained: 4 (coarse and fine, per stage).
- Train, pipeline check: `main_512` train, Perlin, fBm, no warp, uniform map (167), interior windows.
- Train, base: `main_512` train, uniform map (2,000), interior windows.
- Scored on: the same filter on `main_512` test (17 and 200). Per level and world benchmark.
- Reports: every terrain metric per level and for the world; seams.
- Decides: the reference for every track C experiment.
- Claims: 1, 2, 3, 4.

**C2. Edges.**

- Question: does training on whole closed runs teach the model fixed edges that the world does not
  have?
- Varies: training windows ∈ {whole closed runs, interior crops}.
  - Closed runs: `main_256` runs. Each run is exactly one coarse window (64 coarse cells). Fine windows
    are cropped anywhere, edges included.
  - Interior crops: C1's base.
- Models trained: 2 (coarse and fine, closed-run arm). The interior-crop arm is C1's base models.
- Train, closed runs: `main_256` train, uniform map (2,667).
- Train, interior crops: as C1's base.
- Scored on: world benchmark, `main_512` test, uniform map (200).
- Reports: every terrain metric; error against distance from the nearest training-window edge position.
- Decides: whether track B's closed runs can train world levels.
- Claims: 3.

**C3. Sampler settings.**

- Question: which sampler settings give the best world?
- Varies, one at a time from C1's settings:
  - Steps ∈ {8, 16, 32, 64}.
  - Blend phases T ∈ {1, 2, 3}.
  - Split for T = 2 ∈ {2, 5, 10} × `sigma_data`.
  - Splits for T = 3 ∈ {(10, 2), (20, 5)} × `sigma_data`.
- Models: C1's. No training.
- Scored on: as C1.
- Reports: every metric and seams per setting; sampling time.
- Decides: the sampler settings for everything after.
- Claims: 3, 4.

**C4. Coarse conditioning noise.**

- Question: does the fine level do better when trained not to trust its coarse input exactly? At
  generation, the fine level reads the coarse model's output, which has errors. In training it reads
  the true coarse change, which has none.
- Varies: Gaussian noise added to the fine level's coarse channel in training, with standard deviation
  ∈ {0, 0.05, 0.1, 0.2, 0.4} × the coarse change's RMS.
- Fixed: the coarse level is C1's.
- Models trained: 4 fine levels (0 is C1's).
- Data: as C1.
- Reports: world metrics against noise level; fine-level metrics with true and with sampled coarse
  input.
- Decides: the noise level for everything after.
- Claims: 1, 3.

**C5. Scale.**

- Question: how many levels and what windows each band needs, and whether quality holds as features
  grow.
- Each level's cells are 4 times coarser than the one below (1, 4, 16 m). Each level keeps the window
  sizes of C1 (64 cells at the top, 128 at the fine level).
- Varies:
  - Band 2: levels ∈ {2 levels with a 128-cell coarse window, 3 levels (1, 4, 16 m)}.
  - Band 3: levels ∈ {3 levels, 4 levels (1, 4, 16, 64 m)}.
  - Stacks: one stack per band, against one stack trained on bands 1 to 3 together.
- Models trained: one per level per configuration, about 14.
- Train, band 2: `main_1024` train, uniform map (800).
- Train, band 3: `main_2048` train, uniform map (200).
- Train, shared stack: the union of the above with C1's data.
- Scored on: world benchmark, each band's test, uniform map (200, 100, 33).
- Reports: every metric, seams, and consistency per band and configuration.
- Decides: the level setup per band for C6.
- Claims: 1, 3, 5.

**C6. Final world.**

- Question: the headline track C numbers.
- Recipe: phase 2's world recipe, with the track B additions that worked (erodibility maps,
  hydrology), at every band, with C5's level setup.
- Training: to convergence (phase 3 rule). Three seeds.
- Ablations, band 1, one seed each: T = 1; one level fewer; each track B addition removed.
- Train: each band's train, all: `main_512` (3,000), `main_1024` (1,200), `main_2048` (300).
- Scored on: world benchmark, each band's test, all (300, 150, 50).
- Reports: every metric and seams per band, mean and spread over seeds; the ablation table.
- Claims: 1, 2, 3, 4.

**C7. Generalization at world scale.**

- Question: do B10's results hold in the world?
- Model: C6's band 1 world. No training.
- Scored on: `ood_512`, `long_512` (as in B10, one call and chained), `real_dem_512`.
  World benchmark on each.
- Reports: every metric per test, beside C6's band 1 score.
- Claims: 5.

**C8. Baselines.**

- Question: does the world beat every way of making a bounded method infinite? The central comparison
  of the paper.
- Compared:
  - C6's world.
  - Independent chunks, stitched: the simulator, the analytical method, and B9's model, each on
    separate closed 256² chunks.
  - Overlapping chunks, blended: the analytical method on 256² chunks at a 128-cell stride, blended
    with tent weights.
- Models trained: none.
- Scored on: each band's test, all (300, 150, 50), against the whole simulator run. Scored twice:
  within 16 cells of a chunk edge, and over the whole interior.
- Reports: every metric per method and band. With chunk size fixed and features growing, each basin
  crosses more chunk edges, so the chunked baselines should fall further behind at bands 2 and 3.
- Claims: 3.

**C9. Speed and demo.**

- Question: is the world fast enough to explore interactively?
- Varies: query size ∈ {512², 2048²}; cache ∈ {cold, warm}; sampler ∈ {C3's choice, B13's speedups}.
- Models: C6's band 1 world, and a distilled student per level made with B13's method.
- Models trained: 1 per level (the students).
- Reports: latency per query, beside the simulator's time for the same region. An interactive demo in
  the viewer.
- Claims: 4.

---

## Decisions to review

Choices this plan makes that were not settled before.

**Data**

1. **Period** is drawn log-uniform within the group's range, as in the current `datagen`.
2. **Bands 2 and 3 scale height and map sizes** with period (× 2, × 4). The alternative keeps band 1's
   ranges, which makes slopes gentler and maps relatively finer as features grow.
3. **No forced draws.** The pipeline check subset (Perlin, fBm, no warp, uniform map) is the 5.6% that
   comes out naturally: 222 train runs at 256², 167 at 512².
4. **The map split** is 2/3 uniform, 1/6 patches, 1/6 bands, so most data varies only the noise.
5. **Map formulas:** `map_contrast ^ (s − 1/2)`, patch noise with 3 octaves, band transition
   `band_width / 8`, band bend up to one `band_width`. Settle in step 1 by looking at rendered maps.
6. **Test-only groups use Perlin, fBm, no warp, and a uniform map**, except for the one thing each
   tests.
7. **`ood` goes above the range only**, one parameter per run, a third of the runs each.
8. **`real_dem` heights** are scaled to match the standard deviation of a Perlin fBm noise terrain.
9. **The DLA test split uses fixed `p` values and shared starts**, because the distribution metrics
   need many simulator runs at one `p` and from one start.

**Experiments**

10. **B6 trains on all 4,000 runs**, since only a third have non-uniform maps. Only 67 test runs per
    map shape, so per-contrast results will be coarse.
11. **The held-out fractal type test is track B only.** C7 does not retrain a world without ridged terrain.
12. **C2's arms differ in data volume:** 2,667 runs of 256² against 2,000 runs of 512² interiors. Not
    equalized.
13. **Band 3 has only 33 uniform-map test runs.** C5's band 3 numbers will be noisy. C6 uses all 50.

## Steps

### 1. Data settings

Change `datagen` so it can make both datasets.

- The DLA simulator, copied from `../seeded-pa-vae`, with sticking probability, arrival-step storage,
  and continuation from a stored start cluster. Measure cluster sizes and simulator time, and settle
  the DLA settings from them.
- `dla_master`, and every group of `erosion_master`, as profiles with the columns above.
- The map split: 2/3 uniform, 1/6 patches, 1/6 bands.
- Erodibility maps: patches and bands, drawn and stored as settings, computed at any world coordinate
  by `erosion_terrain` (so the world can use them), and passed to the simulator as `K × map`.
- Out-of-range draws for `ood`.
- `real_dem`: download, pick, and rescale tiles. A run takes a stored start terrain instead of noise.
- Measure feature size per noise setting (the 12 combinations), and decide track C's window sizes and edge margin from it.
- Time a few 2048² runs, and size `main_2048` from the measured cost.

Check-in: each profile generated at smoke size, a few runs of each rendered, and the cost estimate
redone from a short timed run.

### 2. Generate the data

Run the profiles through `notebooks/world/data.ipynb`, in the batches listed under Cost. The user runs
it.

Check-in: every split opens, run counts and settings match, disk use matches the estimate.

### 3. Evaluation suites

One function per track that runs every metric for one experiment and writes a results file. Adds the
DLA metrics, the slope-area fit, and the closed-chunk baselines.

Check-in: each suite on smoke checkpoints, and its output file.

### 4. Experiment runner

- Each experiment is an entry in one registry: its name, its track, its data selection, the configs
  of its models, and for track C its world settings.
- One command trains whatever checkpoints are missing, runs the track's suite, and saves the results
  under `artifacts/results/<experiment>/`, with the config and the git commit.
- A notebook collects every results file into one table and the standard plots.
- Measure the time of one training run, and set the phase 1 budget from it.

Check-in: a smoke experiment of each track end to end, and the table.

### 5. Model options for phase 1

- DLA training pairs: occupancy at `t`, `p`, and `x` in; occupancy at `t + x` out.
- Backbone choice in `Config`: U-Net or DiT.
- Objective: flow matching, beside regression and EDM.

Check-in: smoke tests for each combination.

### 6. Phase 1

Run A1 to A6, then B1 to B5.

Check-in: after track A, read its results together before starting track B. After B5, pick the recipe
and revise phase 2.

### 7. Phase 2

Run B6 to B8, then C1 to C5. Build the model inputs these need (erodibility map, drainage area) as
they come up. Build the analytical method and the chunked baselines before C1, and the world with any
number of levels before C5.

Check-in: after B8, and after C5.

### 8. Phase 3

Run B9 to B13 and C6 to C9.

Check-in: the tables and figures for the paper.

## Open questions

- Track C's window sizes and edge margin depend on step 1's feature size measurement.
- Run counts for bands 2 and 3 are a first guess. If the top level is short of data, more runs are the
  fix, not larger domains.
