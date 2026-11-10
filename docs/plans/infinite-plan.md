# Plan: infinite eroded terrain with InfiniteDiffusion and a two-level hierarchy

Status: draft. No steps started.

## Goal

Query any rectangle of an infinite eroded world:

```python
world.get(y0, x0, h, w)  # world holds seed, params, dT, and the models
```

- The seed defines the uneroded starting terrain everywhere, as Perlin noise.
- The models return that terrain eroded for the given parameters and elapsed time `dT`.
- The result for a region depends only on the seed, the region, the parameters, and `dT`. Query order
  and cache state do not change it.
- The cost of a query depends only on the region's size, not on where it is.

This plan gets the machinery working end to end. Quality and speed come later.

## Sources

- **Paper.** Goslin, "InfiniteDiffusion: Bridging Learned Fidelity and Procedural Utility for
  Open-World Terrain Generation", SIGGRAPH 2026, arXiv:2512.08309. Cited below as "paper §n".
- **Code.** https://github.com/xandergos/terrain-diffusion. Cited below by file:
  - `annotated_infinite_panorama.py`: a short, commented demo of InfiniteDiffusion with the
    `infinite-tensor` library, on Stable Diffusion. The closest template for this plan.
  - `terrain_diffusion/inference/world_pipeline.py`: the full terrain pipeline. Coordinate noise
    (`gaussian_noise_patch`), tent weights (`linear_weight_window`), and the stages (`_build_*_stage`).
- **MultiDiffusion.** Bar-Tal et al., "MultiDiffusion: Fusing Diffusion Paths for Controlled Image
  Generation", 2023. InfiniteDiffusion extends it.
- **Library.** `infinite-tensor` (https://github.com/xandergos/infinite-tensor, on PyPI). Lazy,
  cached tensors with infinite dimensions, built from sliding windows (paper Appendix C).

## Background

### Two kinds of noise

The world pipeline uses two unrelated kinds of noise. Keep them apart.

- **Terrain noise.** Perlin fBm (fractal Brownian motion: Perlin noise summed over octaves). This is the
  uneroded starting terrain `z_a`, the model's input. It is smooth and has hills.
- **Diffusion noise.** Unit Gaussian noise, independent at every cell. A diffusion sample starts from
  it, and the sampler turns it into a predicted `Δz`. Today `diffusion.initial_noise` makes one field
  per window from a key. A different noise field gives a different sample.
  - The key leaves out `dT`, so every elapsed time starts from the same noise. Only the `dT`
    conditioning changes between them, and the terrain changes smoothly over time instead of
    jumping. Direct mode in `generate.batched_rollout` already does this, with key `(seed, 0)`.
  - The world keeps this: its noise depends on (seed, level, y, x), never on `dT` (step 2).

### Windows and resolutions

The models only ever see fixed-size windows. A query of any size is answered by running the model on
every window that touches it and combining the results.

- **Window size is fixed per level.** The fine model always sees 64×64 cells. The coarse model always
  sees 32×32 cells.
- **Levels differ in cell size.** A fine cell is 1 m. A coarse cell is the mean of 4×4 fine cells, so
  it is 4 m. The coarse window has fewer cells but covers more ground: 128 m against 64 m.
- **The coarse level gives context.** A fine window cannot see beyond its 64 m. The coarse result
  covers the fine window and its surroundings, at lower resolution, and the fine model reads it as an
  extra input channel.

### How InfiniteDiffusion works

1. **Diffusion noise is tied to world coordinates** (paper §3, `tiled_gaussian_noise` in the demo).
   The noise at a cell is a function of (seed, y, x), not of the window. Two overlapping windows then
   start from the same noise where they overlap, so they tend to agree there.
2. **Overlapping windows, averaged** (paper §3.1, Eq. 1). Windows cover the plane at a stride of half a
   window. Each window is sampled on its own. Where windows overlap, the results are averaged, weighted
   by a tent that is 1 at the window center and near 0 at its edge (`linear_kernel` in the demo). The
   tent hides each window's edge, where the model sees the least context.
3. **Blending only a few times** (paper §3.4). MultiDiffusion blends after every sampler step, so
   neighbors keep agreeing as the sample forms. On an infinite canvas, every blend needs a wider region
   from the step before, so the work grows with each blend. InfiniteDiffusion splits the sampler's
   schedule into T phases and blends only at the end of each phase. See "Why T = 2" below.
4. **Lazy evaluation with a cache** (paper §3.3, Algorithm 1). A query runs only the windows that touch
   it. Their inputs come from queries to the phase before, recursively. Results are cached. Since
   everything is a fixed function of the seed and coordinates, evicting and recomputing a window gives
   the same values.

The paper's terrain stack (a latent model and decoder, Laplacian encoding, consistency distillation,
AutoGuidance) is not part of InfiniteDiffusion, and this plan does not use it. Only the hierarchy is
used (paper §5.3): a coarse level gives each fine window context from beyond its edges.

### Why T = 2

Each window is one sampler run. Diffusion samplers form large structure early, at high noise, and fine
detail late, at low noise.

- **T = 1.** Each window runs the whole schedule alone, and only the finished results are averaged.
  Neighbors may decide on different large structure, for example a river in two slightly different
  places. Averaging then gives two half-depth rivers or a visible seam.
- **T = 2.** The schedule is split at a medium noise level. Each window runs the high-noise half.
  The partly denoised windows are blended, so neighbors share one large structure. Each window then
  runs the low-noise half from the blended state, and the results are blended again. The detail is
  formed on top of a structure that already agrees.

Evidence: on Stable Diffusion panoramas, FID (Fréchet Inception Distance, lower is better) is 7.20 for
T = 1, 5.83 for T = 2, and 5.75 for full MultiDiffusion (paper Table 1). On terrain it is 19.78 for
T = 1 and 14.78 for T = 2 (paper Table 2), and channels fail more often at T = 1 (paper Fig. 8).

Each level is its own windowed sampler, so each level has its own two phases. That gives four stages:
coarse phase 1, coarse phase 2, fine phase 1, fine phase 2.

How the split works depends on the sampler:

- The demo uses an ordinary multi-step sampler (DDIM, denoising diffusion implicit models). It cuts the
  schedule at a noise level, blends the partly denoised states, and continues the same schedule. No
  new noise is added.
- The terrain pipeline uses consistency models, which jump straight to a clean result. It blends clean
  results, adds fresh coordinate noise at a medium level, and denoises again.

Our sampler is the deterministic Heun solver over a fixed schedule, like the demo's. So we follow the
demo: one noise field per level, the schedule cut at one noise level, no re-noising. The paper's T = 2
split is at DDIM step 800 of 1000 (paper Appendix D.1), which is about σ = 5 in EDM units for data
with unit standard deviation. Our `σ_data` is 1.06, so start near σ = 5.

## Decisions

- **Use `infinite-tensor` for windows, blending, and caching.** It does the window bookkeeping, sums
  overlapping window outputs, and caches them. Follow `annotated_infinite_panorama.py`:
  - Each window returns `C + 1` channels: its values times the tent weight, plus the weight. Dividing
    the summed values by the summed weight gives the weighted average.
  - Each phase is an `InfiniteTensor` whose window function reads the phase before through `args`.
- **Level sizes follow the starting terrain's largest period.** fBm has no structure above its largest
  period, and ridges about one period apart divide drainage basins. So the coarse window spans about 4
  periods, and a level coarser than that would see only noise. Each world fixes its period, and window
  sizes are set from it:
  - Fine window: about 2 periods.
  - Coarse window: about 4 periods.
  - Longer elapsed times spread influence farther, so they may need more context. Deferred.
- **Two levels, both EDM diffusion models with the existing U-Net.** First pass, with a world period of
  32 cells, so the existing 128² runs can train both levels:
  - Coarse: 4 m cells. Predicts the coarse `Δz` from the coarse `z_a`. Windows of 32 coarse cells
    (128 m, 4 periods), stride 16.
  - Fine: 1 m cells. Predicts the fine `Δz` from the fine `z_a` and the coarse `Δz`, upsampled and
    added as an input channel. Windows of 64 cells (2 periods), stride 32. The paper's decoder takes
    its coarse input the same way, concatenated as channels (paper §5.3, `_decoder_inference`).
- **Half-window strides**, as in the paper (§3.2) and the demo's latent stage.
- **T = 2 at both levels**, split at about σ = 5. See "Why T = 2".
- **Parameters and `dT` are the same everywhere in a world** and are given to both levels.
- **Diffusion noise is shared across `dT`**, as in the current direct mode.
- **Data: existing 128² runs first.** Step 9 generates large-domain data only once the machinery works.
  Scaling up comes later.
- **Deferred:**
  - base level and outlets (where water leaves an infinite world);
  - how much context drainage needs;
  - adding noise to the coarse conditioning to cover coarse-model errors (the paper does this,
    `cond_snr` in `world_pipeline.py`);
  - more than two levels;
  - consistency distillation;
  - coherence across `dT` queries.

## Steps

### 1. Starting terrain at any coordinate

Make the Perlin fBm starting terrain a function of (seed, y, x), so any region can be computed alone.

Why: the world's input terrain has to exist everywhere. Perlin noise is defined at every coordinate,
but `datagen/src/erosion_datagen/terrain.py` does not compute it that way. Two things tie it to one
array:

- Perlin noise places a random direction at each point of a square grid and interpolates between
  them. `perlin` draws those directions for the whole array at once from one random generator. A
  different array size or position draws different directions. Standard Perlin noise (Perlin 2002)
  gets each direction from a hash of the grid point instead. Use that.
- `fbm` rescales each array to [0, 1] by its own min and max. Drop the rescale and use a fixed
  amplitude, chosen so terrain heights match the training data (about 64 m of relief).

The coarse input is the 4×4 block mean of the fine input.

Check-in: a test that two overlapping queries agree. A render of a large region next to training
terrain.

### 2. Diffusion noise at any coordinate

Make the diffusion noise a function of (seed, level, y, x).

Why: so that overlapping windows start from the same noise where they overlap (see "How
InfiniteDiffusion works", point 1). Today each window gets its own field from a key, so two windows
covering the same cell start from different noise.

How: Gaussian noise cannot be computed per cell from a formula the way Perlin noise can. The demo and
the paper split the plane into fixed tiles and seed one random generator per tile from (seed, tile).
A query reads whichever tiles it touches (`tiled_gaussian_noise` in the demo, `gaussian_noise_patch`
in `world_pipeline.py`). Here each tile's generator is seeded from SHA-256 of a JSON record (seed,
level, tile row, tile column) into NumPy's PCG64, as `AGENTS.md` asks.

Check-in: a test that two overlapping queries agree.

### 3. One scale for every window

Change how terrain is scaled into and out of the model, so the scale no longer depends on the window.

What the normalizer is: the network works on numbers of a similar size, not on meters. `Normalizer`
in `data.py`, with `ErosionPairs` and `generate.predict_batch`, converts to and from model units, in
training and in generation:

- Input: `(z_a - min) / relief`, using the window's own min and relief.
- Target: `Δz / (relief · target_scale)`. `target_scale` depends on `K_ref · dT`, so short and long
  elapsed times give targets of similar size.
- Conditioning includes `U / relief`.

Why it must change: the blend in "Why T = 2" averages the partly denoised states of overlapping
windows. Those states are in model units. If one window has 40 m of relief and its neighbor 60 m, the
same `Δz` in meters is a different number in each window. The noise level differs in meters too.
Averaging them mixes two different scales.

Change:

- Input: `(z_a - window mean) / H`, where `H` is fixed (64 m, the dataset's starting relief).
  Subtracting each window's own mean is fine, since raising or lowering the whole terrain does not
  change `Δz`. Only the scale must be shared.
- Target: `Δz / (H · target_scale)`. `target_scale` already depends only on parameters and `dT`, which
  are global.
- Conditioning: `U / H`.
- Edges. The U-Net, the denoiser, and `initial_noise` force edge cells to zero change, because the
  simulator holds its edges fixed. Inside the world a window edge is not a boundary. Make this a flag,
  off for world models, so existing checkpoints load and behave as before.

The models are trained with the same scaling they are used with, so steps 4 and 5 use this mode.

Check-in: tests pass. Existing checkpoints evaluate the same as before.

### 4. Training datasets for the two levels

Build one dataset per level from the existing 128² runs.

- `ErosionCrops`, fine level: random 64² crops of the runs, every elapsed time, random flips and 90°
  rotations. A second input channel holds the run's true coarse `Δz` (4×4 block mean of `z_b - z_a`),
  upsampled to the crop.
  - Why crops: world windows are pieces of a larger terrain, not whole simulated domains. The paper
    trains every model on random crops so the model does not learn absolute position (paper §5.3,
    Appendix F).
  - Why flips and rotations: the world has no preferred direction (paper Appendix F).
  - Why the true coarse `Δz`: the fine model learns to read the coarse channel as context. At generation
    the coarse model supplies it instead. Its errors are not covered yet (see Deferred).
- `ErosionCoarse`, coarse level: whole runs averaged to 32², with flips and rotations.
- Both use the scaling from step 3.
- Both use only runs whose terrain period is near the world's: 48 cells or less keeps 1,560 of the
  8,000 training runs (`select_runs(params, period=(32, 48))`). See Decisions for why the period
  matters.

Check-in: plots of a few items from each. The fine target, block-averaged, matches the coarse target.

### 5. Train both levels

Add profiles for the two world models and train them.

- Profiles `world_coarse` and `world_fine`, used through `get_config(profile, "diffusion")`.
- The fine denoiser takes 3 input channels: start terrain, upsampled coarse `Δz`, and noisy `Δz`.
- Short step budgets. The models only need to produce sensible output.

Check-in: samples from each level on its own, next to the simulator.

### 6. One level, windowed, on a finite region

Run one level over a region larger than a window with `infinite-tensor`, using the two phases from
"Why T = 2".

- Phase 1: an `InfiniteTensor` whose window function reads the step 2 noise at the window's position
  and runs the Heun schedule from `σ_max` down to the split. It returns the partly denoised window times
  the tent, plus the tent.
- Phase 2: an `InfiniteTensor` that reads the blended phase 1 result and runs the rest of the schedule
  to 0. It returns the result times the tent, plus the tent.
- Query a finite region of phase 2 and divide by the weights.
- This needs a Heun sampler that can start and stop at given noise levels. `diffusion.sample` runs the
  whole schedule, so it gains start and stop arguments.

Check-in: a region one window in size, aligned to the window, equals a plain sample. A 512² fine-only
region (coarse channel set to zero) shows whether seams are visible with T = 1 and with T = 2.

### 7. Two levels

Chain the two levels.

- The two coarse phases, then the two fine phases. Each fine window reads the coarse phase 2 result
  over its footprint through `args`, and upsamples it to fine cells.
- Both levels read the starting terrain from step 1: the coarse level its block means.

Check-in: a 1024² region rendered next to its coarse result and its starting terrain.

### 8. The `World` object

Wrap the pipeline so a user only gives a seed, parameters, `dT`, and a region.

- `World(seed, params, dT, coarse_model, fine_model).get(y0, x0, h, w)` returns eroded heights in m.
- It builds the four `InfiniteTensor` stages once and slices the last one.

Check-in, as tests:

- Querying regions in different orders gives identical results.
- Clearing the cache partway through changes nothing.
- A small query equals the same crop of a larger query.
- Coordinates far from the origin (around 10⁶) work.

### 9. Large-domain data, if needed

Generate simulator runs on large domains, so the coarse level sees well beyond a fine window. Do this
only once steps 1 to 8 work on the existing data.

- World period of 128 cells. The coarse window then covers 512 m, so domains must be larger than that
  for crops away from the edge: 1024² domains with the period fixed near 128 cells, and about 11
  snapshots. About 46 MB per run. Time not yet measured; the full 128² dataset takes about 10 minutes
  on 32 cores.
- Coarse model: 128² crops at 4 m (512 m). Fine model: 256² crops at 1 m. Both away from the domain
  edge.

Check-in: the step 8 renders, before and after.

### 10. Basic evaluation

Measure whether the machinery works, not whether the terrain is good.

- Seams: how much neighboring windows disagree in their overlap before blending, after each phase.
- Order invariance and cache invariance, on large random query sets.
- Time per query, first and later (warm cache).
- For reference, a 512² simulator run on the same starting terrain, next to the world output.

Check-in: read the results together. Decide what to improve first.
