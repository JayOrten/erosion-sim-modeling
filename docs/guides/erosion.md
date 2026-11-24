# Reading erosion results (track B)

What the notebooks' tables and plots show, and what good looks like.

**Setup.** Each test pair is a terrain at a start time and the simulator's terrain some years later (`dT`).
Every method predicts the end terrain from the start, and is scored on every cell. The simulator follows the
stream power law with hillslope diffusion: `dz/dt = -K A^m |∇z| + D ∇²z`, where `K` is erodibility, `A`
drainage area (the upstream area draining through a cell), `m` the area exponent, and `D` the diffusion rate.

**Entries (table rows).** Baselines are scored like models: each is its own predictor (`configs/predictors/`), evaluated with its own command in the notebook's Run section.

- `no_change`: the start terrain. The floor every method must beat.
- `analytical`: the stream power law solved exactly (Tzathas et al. 2024). Exact for channel cutting alone;
  misses hillslope smoothing and rivers changing course.
- each model by name; `<name>@<steps>` in a step sweep; `<name> + pit filling` after filling pits;
  `<name> chained <n>x<years>` for `n` calls in a row.

**Metrics.** Channels are the cells with the top 5% of the simulator's drainage area in that pair.

| Metric | What it is | Good |
|---|---|---|
| `rmse` | Root mean square height error (m) | Low |
| `skill` | `1 - MSE / MSE(no change)` over all pairs: share of the true change's error removed | 1 perfect, 0 = no change, negative = worse than doing nothing |
| `median_skill` | Median of per-pair skill. Less swayed by a few big pairs | As above |
| `erosion_ratio` | Mean absolute change over the simulator's | 1. Below is too little change, above too much |
| `iou` | Channel overlap: intersection over union of predicted and true channel cells | 1 perfect, 0 none |
| `iou_1` | Same, allowing one cell of slack | 1. Much higher than `iou` means channels are in the right place but slightly off |
| `log_area_rmse` | RMSE of log10 drainage area. 0.3 is about a factor of 2 | Low |
| `pits_per_km2` | Cells with no lower neighbor, per km². Pits trap water and break drainage | Low; compare with `no_change` |
| `slope_area_error` | Gap between the method's and the simulator's slope-area exponent (how channel slope falls with drainage area) | Low |
| `fine_detail` | Change power at wavelengths of 4 cells or less, over the simulator's | 1. Below is too smooth, above is noise |
| `spread_correlation` | Generative models: correlation of per-cell spread across samples with per-cell error | High: the model is unsure where it is wrong |
| `seconds_per_pair` | Model time per pair | Low |

**Plots.**

- *Breakdowns* (`by_table`, `plot_by`): skill per group of one factor, one line per entry. Factors: `K_ref`
  (erodibility), `D_over_K` (diffusion relative to erodibility), `m`, `dT` (elapsed years), terrain noise
  setup, erodibility map shape and contrast, and the held-out parameter in out-of-distribution tests. A line
  dropping in one group shows where a method fails.
- *Spectrum* (`plot_spectrum`): change power at each wavelength over the simulator's. Fine detail is on the
  right. 1 is matching detail; below 1 is smoother, above is noisier.
- *Examples* (`plot_examples`): per row, the start terrain (shaded relief), then the height change of the
  simulator and each entry. Red is lowered (erosion), blue raised (deposition), on one shared scale per row.
  Compare channel positions and sharpness with the simulator's panel.
- *Step sweep* (`plot_steps`): metrics against network calls per sample. Look for where it stops improving.

**Training plot.** Left: train and validation loss; both should fall and level off. Validation rising while
train keeps falling is overfitting (the best-validation checkpoint is the one scored). Right: validation
skill, `1 - MSE / MSE(no change)`: 1 is perfect, 0 is no better than copying the input. For generative
models it is a one-step estimate, so read its trend, not its value.
