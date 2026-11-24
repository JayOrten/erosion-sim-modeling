# Reading world results (track C)

What the notebooks' tables and plots show, and what good looks like.

**Setup.** A cascade is a stack of level models, coarsest first. Each level predicts the erosion change on
overlapping windows, using the coarser level's output as context, and the windows are blended into
terrain for any region (a world). The cascade is run on whole test runs and scored on the interior, away from the run's fixed edges. The run's own edges affect
drainage everywhere, which the cascade cannot see, so expect an error floor of about 10–15% of the change
(`docs/notes/datagen-measurements.md`).

**Entries (table rows).** Baselines are scored like models: each is its own predictor (`configs/predictors/`), evaluated with its own command in the notebook's Run section.

- each cascade by name: the full pipeline.
- `level:<name>`: one level alone, given the true parent change as context, as in training. Separates a
  level's own error from errors passed down the pipeline.
- `no_change`: the start terrain.
- `analytical`: the stream power law solved exactly on the whole run.
- `analytical_chunks_256`, `erosion_final_seed0_chunks_256` (the single-window model), `simulator_chunks_256`:
  that method on separate closed 256² chunks, stitched.
  Each chunk's edges act as false outlets, so these show the cost of naive chunking.
- `analytical_blended_256`: analytical on 256² chunks at half-chunk stride, blended.

**Metrics.** The same as the erosion notebooks (`rmse`, `skill`, `iou`, and so on: skill 1 is perfect and 0
is no change; IoU 1 is perfect; ratios 1 are right; errors low), plus:

| Metric | What it is | Good |
|---|---|---|
| `chunk_edge_rmse` | RMSE within a few cells of an internal chunk boundary, where chunking errors appear | Low, and close to the entry's overall `rmse` |
| `seconds_per_query` | Time per cascade query | Low |

**Seams.** How much neighboring windows disagree where they overlap, before blending, as a share of the
change there. One number per phase; the last phase is the result. Low means the windows already agree; high
means the blend averages different answers.

**Breakdown by `dT`.** Skill per elapsed time. Longer times should be harder; a sharp drop shows where the
model stops being useful.

**Examples.** Per row, the start terrain, then the height change of the simulator and each entry. Red is
lowered (erosion), blue raised (deposition). Look for visible chunk or window boundaries.

**Training plot.** Left: train and validation loss; both should fall and level off. Validation rising while
train keeps falling is overfitting (the best-validation checkpoint is the one scored). Right: validation
skill, `1 - MSE / MSE(no change)`: 1 is perfect, 0 is no better than copying the input. For generative
models it is a one-step estimate, so read its trend, not its value.
