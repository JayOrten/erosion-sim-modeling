# Plan: diffusion surrogate, compared against the U-Net regression baseline

Status: draft.

## Goal

Train a diffusion model that predicts eroded terrain from the same inputs as the U-Net baseline (see
`docs/unet-baseline-plan.md`), then compare the two approaches on the same validation pairs.

A diffusion model learns a distribution and draws samples from it. It starts from pure noise and
removes the noise step by step. The regression U-Net instead learns the mean of the possible answers.
The simulator is deterministic, so a better pixel error is not the expected gain. The expected gain is
sharper channels and better hydrology in places where the regression model averages over possible
channel positions.

After the comparison, the regression code will likely be removed. Until then, both stay working.

## Decisions

- **Formulation: EDM.** EDM is the noise schedule, preconditioning, and loss weighting from Karras et
  al. 2022, "Elucidating the Design Space of Diffusion Models". EDM2, used by Terrain Diffusion, builds
  on it. Sampling uses EDM's Heun solver, a second-order ODE (ordinary differential equation) solver.
- **Architecture: the existing U-Net.** Two changes. It takes a second input channel (the noisy
  target), and the noise level joins the conditioning vector. The EDM2 architecture
  (magnitude-preserving layers) is deferred.
- **Target: the same as regression.** The model generates the normalized change `Δz = z_b - z_a`,
  conditioned on the normalized start heightmap `z_a`. `data.py` and `Normalizer` do not change.
- **Layout.** `Config` gains `objective: "regression" | "diffusion"`. The diffusion loss and sampler
  live in a new `diffusion.py`. `train.py`, `generate.py`, and `evaluate.py` branch on the field.
  Diffusion checkpoints are `artifacts/diffusion-<profile>.pt`.
- **Deferred:** guidance (classifier-free guidance, AutoGuidance) and consistency distillation.

## Steps

### 1. Add the objective to `Config`

Add a field that says which kind of model a run trains, and name checkpoints by it.

- `objective: str = "regression"`, so existing profiles and checkpoints are unchanged.
- `checkpoint_path` uses the prefix `unet-` for regression and `diffusion-` for diffusion.
- `load_checkpoint` reads the field and builds the matching model. Old checkpoints without the field
  load as regression.

Check-in: existing tests pass. An old checkpoint still loads.

### 2. Build the denoiser

Wrap the U-Net so it can denoise a noisy `Δz` at any noise level.

- `UNet` gains an `in_channels` argument (default 1). The diffusion model uses 2: start heightmap
  and noisy `Δz`.
- `Denoiser` in `diffusion.py` applies EDM preconditioning. Given noisy `Δz` with noise level `σ`, it
  scales the input by `c_in`, passes `c_noise = ln(σ) / 4` as an extra conditioning scalar, and
  returns `c_skip · noisy + c_out · network output`. The constants depend on `σ_data`, the standard
  deviation of the normalized target.
- `σ_data` is measured once on the training set when the normalizer is fitted, and stored in the
  checkpoint.
- Edge nodes are always zero change. The denoiser zeroes them in its input and its output.
- `c_noise` is mapped into the conditioning range of the other scalars (about [0, 1]), so the
  existing sinusoidal features handle it.

Check-in: unit tests for output shape, zeroed edges, and the preconditioning at very small and very
large `σ` (it should return the input, and a scaled network output, respectively).

### 3. Train with the EDM loss

Add the diffusion training loss and a validation loss that is stable from one evaluation to the next.

- Training: draw `σ` from EDM's log-normal distribution (`P_mean = -1.2`, `P_std = 1.2`). Add
  `σ · noise` to `Δz` and use the weighted MSE between the denoiser output and the clean `Δz`.
- Validation: the same loss on a fixed grid of `σ` values with fixed noise. The same noise is reused
  at every evaluation, so the numbers can be compared over training. This loss cannot be compared
  with the regression validation loss. Step 7 handles cross-model comparison.
- Early stopping and best-checkpoint selection use this validation loss.

Check-in: the `smoke` diffusion profile trains on CPU in seconds.

### 4. Write the sampler

Turn noise into a predicted terrain change.

- `sample(denoiser, x, cond, steps, seed)` runs EDM's Heun solver from `σ_max` to 0. It uses EDM's
  default time steps (`ρ = 7`) and about 32 steps to start.
- Noise comes from a seed. Following `AGENTS.md`, the stream is derived by hashing a canonical JSON
  record (seed, pair index, sample index) with SHA-256. The same seed always gives the same sample.
- `generate.predict` gains `samples` and `seed` arguments. For diffusion it returns
  `(samples, H, W)`. For regression it is unchanged.
- `rollout` draws one sample per step, from one seed.

Check-in: two calls with the same seed match exactly; different seeds differ. Unit tests for both.

### 5. Add profiles and run the overfit test

Add diffusion profiles that mirror the regression ones, and check that the pipeline can memorize.

- `diffusion_smoke`, `diffusion_overfit`, `diffusion_fixed_dt`. Same data settings as their
  regression twins.
- Run `diffusion_overfit` on 10 pairs. Samples for those pairs should closely match the simulator.

Check-in: look at overfit samples next to the simulator before spending GPU time on step 6.

### 6. Train `diffusion_fixed_dt`

Train on the same data as the regression `fixed_dt` run: one elapsed time (10,000 yr), no uplift.

- Use the same step budget as regression first. Increase it if the validation loss is still falling.
- Pick the number of sampling steps by looking at the final metrics for 8, 16, 32, and 64 steps.

Check-in: training curves and a few samples. Agree on the step count.

### 7. Compare regression and diffusion

Evaluate both models on the same validation pairs with the existing metrics, plus metrics for spread.

- `predict_pairs` takes several checkpoints. It draws the pairs once, so every method sees the same
  pairs. For diffusion it stores N samples per pair (N = 8 to start).
- Methods in the results: `no change`, `regression`, `diffusion (1 sample)`, `diffusion (mean of N)`.
- Existing metrics run on every method: `error_metrics`, `hydrology_metrics` (channel IoU and pit
  counts), `spectrum_comparison`, `error_by_condition`, `error_by_elapsed`.
- New: sample spread. This is the per-pixel standard deviation across samples. Compare where it is
  large with where regression error is large. If they coincide, the diffusion model is uncertain where
  regression is blurry.
- Rollout error and the GIF comparison run both models over the same runs.
- Speed: one diffusion prediction costs `steps` network calls (about 2× that with Heun). Report time
  per prediction for both.
- A section in `notebooks/evaluate.ipynb` runs all of this from two checkpoint paths.

Check-in: read the results together. Decide whether to remove regression, keep it, or change the
diffusion setup.
