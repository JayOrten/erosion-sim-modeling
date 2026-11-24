# Reading DLA results (track A)

What the notebooks' tables and plots show, and what good looks like.

**DLA (diffusion-limited aggregation).** Particles arrive one at a time, random-walk, and stick when they
touch the cluster. A walker next to the cluster sticks with **sticking probability `p`**, and otherwise
keeps walking. Low `p` lets walkers slip into gaps before sticking: dense, compact clusters. `p` = 1 is
plain DLA: thin, spread-out branches. Every run grows to 600 particles on a 128 × 128 grid.

**Test conditions.**

- `p=<value>`: grow from the lone seed particle to 600 particles at that `p`. Generation from scratch.
- `start=<id>`: continue a shared 200-particle start cluster to 600. Starts 0–9 were grown at `p` = 0.1,
  10–19 at `p` = 1. Continuation of an existing cluster.

**How scoring works.** DLA is random, so a sample cannot match a simulator run cell for cell. Each metric
gives one number per cluster, and the model's distribution is compared with the simulator's under the same
condition. Good means the model's mean ± std matches the simulator's.

**Metrics.**

| Metric | What it is | Simulator | Good / bad |
|---|---|---|---|
| `particles` | Occupied cells | exactly 600 | Near 600. Lower means too few particles added. |
| `connected_fraction` | Share of cells 4-connected (side neighbors) to the center seed | exactly 1 | 1. Lower means floating fragments; 0 means the seed cell is empty. |
| `start_kept` | Share of the start cluster's cells still occupied | exactly 1 | 1. Lower means the model erased what it was given. In `p=` conditions the start is only the seed, so this is the share of samples that have one. |
| `gyration_radius` | Root mean square distance of cells from their center of mass (cells): how spread out | about 13 at `p` = 0.05, 20 at `p` = 1 | Match mean and std. Large mean and std together usually mean stray far-out cells. |
| `fractal_dimension` | Mass dimension: slope of log(cells within r of the seed) against log r. Denser is higher | about 1.83 at `p` = 0.05, 1.62 at `p` = 1 | Match, and fall as `p` rises. Noisy per cluster. |

The first three are hard checks: the simulator always scores perfectly. The last two check the shape.
`seconds per sample` is the model's sampling time.

**Plots.**

- *Metric plots* (`plot_metric`): one panel per metric, one x position per condition. Black is the
  simulator's mean ± std, color is the model. The dots should overlap, with similar error bars.
- *Examples* (`plot_examples`): per row, the start, 3 simulator clusters, then 3 per model. Look for specks
  apart from the main body, a missing center, and the right density for that `p`.
- *Radial density* (`plot_radial_density`): mean occupancy against distance from the seed. The model's curve
  should lie on the black one. Low near the center is a hollow core; a long tail is stray particles.
- *Step sweep* (`plot_steps`, A4): each metric's 1-Wasserstein distance to the simulator (the mean gap
  between the two distributions' quantiles, in the metric's own units), averaged over conditions, against
  network calls per sample. Lower is better; look for where it stops improving. Last panel: time per sample.
- *Containment* (`plot_containment`, A6): from one start and one noise field, the share of the smaller
  sample's cells missing from the larger sample. Growing further should only add cells, so 0 everywhere is
  perfect agreement.
- *Chaining* (`plot_chain`, A6): the full count reached in 1, 2, or 4 jumps. Gray band is the simulator's
  mean ± std. Dots should stay inside it as the number of jumps rises.

**Training plot.** Left: train and validation loss; both should fall and level off. Validation rising while
train keeps falling is overfitting (the best-validation checkpoint is the one scored). Right: validation
skill, `1 - MSE / MSE(no change)`: 1 is perfect, 0 is no better than copying the input. For generative
models it is a one-step estimate, so read its trend, not its value.
