"""Write one notebook per registered experiment (``erosion_sim.experiments.registry``), from a template per suite.

Run it after adding an experiment: ``uv run python notebooks/experiments/make_notebooks.py``. It rewrites
the notebooks of every experiment, so write conclusions in the notebooks after the results exist, and
regenerate only experiments that have none (``--only B1 B2``).
"""

import argparse
import json
import pathlib

from erosion_sim.experiments import registry

HERE = pathlib.Path(__file__).parent


def nb(cells):
    out = []
    for kind, text in cells:
        cell = {"cell_type": kind, "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        out.append(cell)
    return {"cells": out, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}

SETUP = '''
import json

from erosion_sim.{plots} import plots
from erosion_sim.experiments import registry
from erosion_sim.experiments.results import read_results
from erosion_sim.experiments.runner import model_folder
from erosion_sim.training.train import plot_history

NAME = "{name}"
exp = registry.get(NAME)
results = read_results(NAME)
print(exp.question)
{{"models": {{label: m["folder"] for label, m in results.manifest["models"].items()}},
 **{{k: results.manifest[k] for k in ("code", "time", "slurm_job")}}}}
'''
HISTORY = '''
for label in exp.models:
    folder = model_folder(NAME, label)[0]
    print(label, folder)
    plot_history(json.loads((folder / "history.json").read_text()))
'''
CONCLUSION = "## Conclusion\n\n_Written after reading the results above._"

TRAINING_GUIDE = """
**Training plot.** Left: train and validation loss; both should fall and level off. Validation rising while
train keeps falling is overfitting (the best-validation checkpoint is the one scored). Right: validation
skill, `1 - MSE / MSE(no change)`: 1 is perfect, 0 is no better than copying the input. For generative
models it is a one-step estimate, so read its trend, not its value.
"""

DLA_GUIDE = """
## How to read this

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
""" + TRAINING_GUIDE

EROSION_GUIDE = """
## How to read this

**Setup.** Each test pair is a terrain at a start time and the simulator's terrain some years later (`dT`).
Every method predicts the end terrain from the start, and is scored on every cell. The simulator follows the
stream power law with hillslope diffusion: `dz/dt = -K A^m |∇z| + D ∇²z`, where `K` is erodibility, `A`
drainage area (the upstream area draining through a cell), `m` the area exponent, and `D` the diffusion rate.

**Entries (table rows).**

- `no change`: the start terrain. The floor every method must beat.
- `analytical`: the stream power law solved exactly (Tzathas et al. 2024). Exact for channel cutting alone;
  misses hillslope smoothing and rivers changing course.
- each model by label; `<label>@<steps>` in a step sweep; `<label> + pit filling` after filling pits;
  `<label> chained <n>x<years>` for `n` calls in a row.

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
| `pits_per_km2` | Cells with no lower neighbor, per km². Pits trap water and break drainage | Low; compare with `no change` |
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
""" + TRAINING_GUIDE

WORLD_GUIDE = """
## How to read this

**Setup.** A world is a stack of levels, coarsest first. Each level predicts the erosion change on
overlapping windows, using its parent's output as context, and the windows are blended. The world is run on
whole test runs and scored on the interior, away from the run's fixed edges. The run's own edges affect
drainage everywhere, which the world cannot see, so expect an error floor of about 10–15% of the change
(`docs/notes/datagen-measurements.md`).

**Entries (table rows).**

- each world by name: the full pipeline.
- `level:<label>`: one level alone, given the true parent change as context, as in training. Separates a
  level's own error from errors passed down the pipeline.
- `analytical`: the stream power law solved exactly on the whole run.
- `analytical_chunks`, `model_chunks`, `simulator_chunks`: that method on separate closed chunks, stitched.
  Each chunk's edges act as false outlets, so these show the cost of naive chunking.
- `analytical_blended`: analytical on chunks at half-chunk stride, blended.

**Metrics.** The same as the erosion notebooks (`rmse`, `skill`, `iou`, and so on: skill 1 is perfect and 0
is no change; IoU 1 is perfect; ratios 1 are right; errors low), plus:

| Metric | What it is | Good |
|---|---|---|
| `chunk_edge_rmse` | RMSE within a few cells of an internal chunk boundary, where chunking errors appear | Low, and close to the entry's overall `rmse` |
| `seconds_per_query` | Time per world query | Low |

**Seams.** How much neighboring windows disagree where they overlap, before blending, as a share of the
change there. One number per phase; the last phase is the result. Low means the windows already agree; high
means the blend averages different answers.

**Breakdown by `dT`.** Skill per elapsed time. Longer times should be harder; a sharp drop shows where the
model stops being useful.

**Examples.** Per row, the start terrain, then the height change of the simulator and each entry. Red is
lowered (erosion), blue raised (deposition). Look for visible chunk or window boundaries.
""" + TRAINING_GUIDE

def _data_command(profile: str) -> str:
    """The command that makes dataset ``profile``: DLA locally, erosion groups as a CPU job."""
    if profile.startswith("dla"):
        return f"uv run erosion-sim dla-generate {profile}"
    return f"sbatch slurm/datagen.sbatch {profile.split('/', 1)[1]}"


def run_section(name: str) -> str:
    """Markdown listing what to run for experiment ``name``, read from its registry entry."""
    exp = registry.get(name)
    data, train, borrowed, needs = [], [], [], []
    for label in exp.models:
        owner, owner_label, cfg = registry.resolve(name, label)
        data.append(cfg.data_profile)
        if owner == name:
            train.append(f"uv run erosion-sim train {name} {label}")
        else:
            borrowed.append(f"- `{label}` is `{owner}/{owner_label}`, trained in {owner}. If it is not trained yet, "
                            f"`uv run erosion-sim train {name} {label}` trains it there.")
        for kind, other in (("residual_of", cfg.residual_of), ("teacher", cfg.teacher)):
            if other:
                needs.append(f"- `{label}` needs `{other}` trained first ({kind}).")
    settings = exp.settings
    test = getattr(settings, "data_profile", None)
    if test:
        data.append(test)
    data = list(dict.fromkeys(data))
    lines = ["## Run", "", "Run these from the repo root, in order. `uv run erosion-sim list` shows what is already "
             "trained.", "", "**1. Data.** Skip any that already exist.", "", "```sh"]
    lines += [_data_command(d) for d in data]
    if "simulator_chunks" in getattr(settings, "baselines", ()):
        group = (test or data[0]).split("/", 1)[1]
        lines.append(f"(cd datagen && uv run erosion-datagen chunked {group} test --chunk {settings.chunk})"
                     "   # in a CPU job")
    lines += ["```", ""]
    if train:
        lines += ["**2. Train.** One command per model; each resumes if stopped.", "", "```sh", *train, "```", ""]
    else:
        lines += ["**2. Train.** Nothing new: every model comes from another experiment.", ""]
    if borrowed or needs:
        lines += [*borrowed, *needs, ""]
    lines += ["**3. Evaluate.**", "", "```sh", f"uv run erosion-sim evaluate {name}", "```", "",
              f"On the cluster, `uv run erosion-sim submit {name}` does steps 2 and 3 as Slurm jobs.", "",
              "Then run the cells below."]
    return "\n".join(lines)


def header(name: str, note: str = "") -> str:
    exp = registry.get(name)
    note = f" {note}" if note else ""
    return (f"# {name}. {exp.title}\n\n{exp.question}\n\nSee `docs/plans/experiments-plan.md`, entry {name}.{note} "
            f"Results are read from `artifacts/results/{name}/`. Nothing here trains or samples.\n\n{run_section(name)}")


notebooks = {
    "A1": nb([
        ("markdown", header("A1")),
        ("markdown", DLA_GUIDE),
        ("code", SETUP.format(name="A1", plots="dla")),
        ("markdown", "## Training"),
        ("code", HISTORY),
        ("markdown", "## Every metric, model beside simulator"),
        ("code", "plots.table(results)"),
        ("code", "for metric in plots.METRIC_LABELS:\n    plots.plot_metric(results, metric)"),
        ("markdown", "## Examples"),
        ("code", "plots.plot_examples(results)\nshared = [c for c in results.manifest[\"conditions\"] if c.startswith(\"start=\")]\nplots.plot_examples(results, conditions=shared[:3] + shared[-3:])"),
        ("code", "plots.plot_radial_density(results)"),
        ("markdown", CONCLUSION),
    ]),
    "A2": nb([
        ("markdown", header("A2", "The EDM arm is A1's model; only flow matching trains.")),
        ("markdown", DLA_GUIDE),
        ("code", SETUP.format(name="A2", plots="dla")),
        ("markdown", "## Training"),
        ("code", HISTORY),
        ("markdown", "## Every metric, at 32 sampler steps (EDM: 63 network calls, flow matching: 32)"),
        ("code", "plots.table(results)"),
        ("code", "for metric in plots.METRIC_LABELS:\n    plots.plot_metric(results, metric)"),
        ("code", "plots.plot_examples(results)\nplots.plot_radial_density(results)"),
        ("markdown", CONCLUSION + "\n\nThen set `DLA_BEST` and A3's models in `experiments/registry.py` to the winning objective."),
    ]),
    "A3": nb([
        ("markdown", header("A3")),
        ("markdown", DLA_GUIDE),
        ("code", SETUP.format(name="A3", plots="dla")),
        ("markdown", "## Training\n\nCompare time per step too: the history's `seconds` over its last step."),
        ("code", HISTORY + "\nfor label in exp.models:\n    done = json.loads((model_folder(NAME, label)[0] / \"done.json\").read_text())\n    print(label, f\"{done['seconds'] / done['step'] * 1000:.1f} ms per step\")"),
        ("code", "plots.table(results)"),
        ("code", "for metric in plots.METRIC_LABELS:\n    plots.plot_metric(results, metric)"),
        ("code", "plots.plot_examples(results)"),
        ("markdown", CONCLUSION),
    ]),
    "A4": nb([
        ("markdown", header("A4", "No training: A2's two models at 4, 8, 16, 32, and 64 steps.")),
        ("markdown", DLA_GUIDE),
        ("code", SETUP.format(name="A4", plots="dla")),
        ("markdown", "## Distance to the simulator against network calls\n\nEach point is a metric's 1-Wasserstein distance to the simulator, averaged over the test conditions. Lower is better."),
        ("code", "plots.plot_steps(results)"),
        ("code", "plots.table(results, [\"particles\", \"connected_fraction\", \"fractal_dimension\"])"),
        ("markdown", CONCLUSION + "\n\nThen set the step count for A5 and A6 in `experiments/registry.py`."),
    ]),
    "A5": nb([
        ("markdown", header("A5")),
        ("markdown", DLA_GUIDE),
        ("code", SETUP.format(name="A5", plots="dla")),
        ("code", "plots.table(results, [\"gyration_radius\", \"fractal_dimension\", \"connected_fraction\"])"),
        ("code", "for metric in (\"fractal_dimension\", \"gyration_radius\"):\n    plots.plot_metric(results, metric)"),
        ("code", "plots.plot_examples(results)"),
        ("markdown", CONCLUSION),
    ]),
    "A6": nb([
        ("markdown", header("A6")),
        ("markdown", DLA_GUIDE),
        ("code", SETUP.format(name="A6", plots="dla")),
        ("markdown", "## Containment\n\nFraction of the smaller sample's cells missing from the larger one, for every pair of jumps from one start and one noise field. 0 is perfect agreement."),
        ("code", "plots.plot_containment(results)\nresults.scalars[next(iter(exp.models))][\"containment\"]"),
        ("markdown", "## Chaining\n\nThe full count reached in 1, 2, or 4 equal jumps, against the simulator's continuations from the same shared starts."),
        ("code", "for metric in plots.METRIC_LABELS:\n    plots.plot_chain(results, metric)"),
        ("markdown", CONCLUSION),
    ]),
    "summary": nb([
        ("markdown", "# Results summary\n\nEvery finished results folder, from `artifacts/results/`."),
        ("code", "from erosion_sim.experiments.registry import EXPERIMENTS\nfrom erosion_sim.experiments.results import list_results\n\nrows = list_results()\nfor r in rows:\n    title = EXPERIMENTS[r[\"name\"]].title if r[\"name\"] in EXPERIMENTS else \"\"\n    print(f\"{r['name']:<20} {title:<26} {r['suite']:<14} {r['time']}\")\n    print(f\"    {r['models']}\")"),
    ]),
}


def erosion_notebook(exp) -> dict:
    name = exp.name
    sweep = exp.settings is not None and exp.settings.step_sweep
    chains = exp.settings is not None and exp.settings.chains
    cells = [
        ("markdown", header(name)),
        ("markdown", EROSION_GUIDE),
        ("code", SETUP.format(name=name, plots="erosion")),
        ("markdown", "## Training"),
        ("code", HISTORY),
        ("markdown", "## Every metric, over all test pairs"),
        ("code", "plots.table(results)"),
    ]
    if sweep:
        cells += [("markdown", "## Against network calls"), ("code", "plots.plot_steps(results)")]
    if chains:
        cells += [("markdown", "## One call against chained calls"),
                  ("code", 'plots.table(results)  # rows "<model> chained <n>x<years>" are scored on the pairs at n × years')]
    cells += [
        ("markdown", "## Breakdowns\n\nEach factor with more than one group among the test pairs."),
        ("code", 'for factor in results.scalars["by"]:\n    plots.by_table(results, factor, "skill")\n    print()'),
        ("code", 'for factor in results.scalars["by"]:\n    plots.plot_by(results, factor, "skill")'),
        ("markdown", "## Detail and examples"),
        ("code", "plots.plot_spectrum(results)\nplots.plot_examples(results)"),
        ("markdown", CONCLUSION),
    ]
    return nb(cells)


def world_notebook(exp) -> dict:
    name = exp.name
    return nb([
        ("markdown", header(name)),
        ("markdown", WORLD_GUIDE),
        ("code", SETUP.format(name=name, plots="erosion")),
        ("markdown", "## Training"),
        ("code", HISTORY),
        ("markdown", "## Worlds, levels alone, and baselines, on the runs' interiors"),
        ("code", "plots.table(results, plots.TABLE + ('chunk_edge_rmse', 'seconds_per_query'))"),
        ("code", 'if "dT" in results.scalars["by"]:\n    plots.by_table(results, "dT", "skill", kinds=("world", "baseline"))'),
        ("markdown", "## Seams\n\nHow much neighboring windows disagree before blending, per level and phase, as a share of the change."),
        ("code", 'for world, levels in results.scalars["seams"].items():\n    for level, phases in levels.items():\n        print(world, level, [f"{p[\'relative\']:.1%}" for p in phases])'),
        ("markdown", "## Examples"),
        ("code", 'plots.plot_examples(results, kinds=("world", "baseline"))'),
        ("markdown", CONCLUSION),
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--only", nargs="*", help="experiment names; default every registered experiment")
    args = parser.parse_args()
    books = dict(notebooks)
    for name, exp in registry.EXPERIMENTS.items():
        if name.startswith("smoke") or name in books:
            continue
        books[name] = erosion_notebook(exp) if exp.suite == "erosion" else world_notebook(exp)
    for name, book in books.items():
        if args.only and name not in args.only:
            continue
        path = HERE / f"{name}.ipynb"
        path.write_text(json.dumps(book, indent=1) + "\n")
        print(path)


if __name__ == "__main__":
    main()
