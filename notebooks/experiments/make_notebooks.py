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
{{k: results.manifest[k] for k in ("models", "time", "slurm_job")}}
'''
HISTORY = '''
for label in exp.models:
    folder = model_folder(NAME, label)[0]
    print(label, folder)
    plot_history(json.loads((folder / "history.json").read_text()))
'''
CONCLUSION = "## Conclusion\n\n_Written after reading the results above._"

def header(name, title, run_note):
    return f"# {name}. {title}\n\nSee `docs/plans/experiments-plan.md`, entry {name}. {run_note}\n\nResults are read from `artifacts/results/{name}/` (or `$EROSION_ARTIFACTS/results/{name}/`). Nothing here trains or samples."

RUN = "Run it from the login node with `erosion-sim submit {name}`."
notebooks = {
    "A1": nb([
        ("markdown", header("A1", "DLA base", RUN.format(name="A1"))),
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
        ("markdown", header("A2", "Objective", RUN.format(name="A2") + " The EDM arm is A1's model; only flow matching trains.")),
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
        ("markdown", header("A3", "Backbone", RUN.format(name="A3"))),
        ("code", SETUP.format(name="A3", plots="dla")),
        ("markdown", "## Training\n\nCompare time per step too: the history's `seconds` over its last step."),
        ("code", HISTORY + "\nfor label in exp.models:\n    done = json.loads((model_folder(NAME, label)[0] / \"done.json\").read_text())\n    print(label, f\"{done['seconds'] / done['step'] * 1000:.1f} ms per step\")"),
        ("code", "plots.table(results)"),
        ("code", "for metric in plots.METRIC_LABELS:\n    plots.plot_metric(results, metric)"),
        ("code", "plots.plot_examples(results)"),
        ("markdown", CONCLUSION),
    ]),
    "A4": nb([
        ("markdown", header("A4", "Sampler steps", RUN.format(name="A4") + " No training: A2's two models at 4, 8, 16, 32, and 64 steps.")),
        ("code", SETUP.format(name="A4", plots="dla")),
        ("markdown", "## Distance to the simulator against network calls\n\nEach point is a metric's 1-Wasserstein distance to the simulator, averaged over the test conditions. Lower is better."),
        ("code", "plots.plot_steps(results)"),
        ("code", "plots.table(results, [\"particles\", \"connected_fraction\", \"fractal_dimension\"])"),
        ("markdown", CONCLUSION + "\n\nThen set the step count for A5 and A6 in `experiments/registry.py`."),
    ]),
    "A5": nb([
        ("markdown", header("A5", "Sticking probability", RUN.format(name="A5"))),
        ("code", SETUP.format(name="A5", plots="dla")),
        ("code", "plots.table(results, [\"gyration_radius\", \"fractal_dimension\", \"connected_fraction\"])"),
        ("code", "for metric in (\"fractal_dimension\", \"gyration_radius\"):\n    plots.plot_metric(results, metric)"),
        ("code", "plots.plot_examples(results)"),
        ("markdown", CONCLUSION),
    ]),
    "A6": nb([
        ("markdown", header("A6", "Agreement across steps", RUN.format(name="A6"))),
        ("code", SETUP.format(name="A6", plots="dla")),
        ("markdown", "## Containment\n\nFraction of the smaller sample's cells missing from the larger one, for every pair of jumps from one start and one noise field. 0 is perfect agreement."),
        ("code", "plots.plot_containment(results)\nresults.scalars[next(iter(exp.models))][\"containment\"]"),
        ("markdown", "## Chaining\n\nThe full count reached in 1, 2, or 4 equal jumps, against the simulator's continuations from the same shared starts."),
        ("code", "for metric in plots.METRIC_LABELS:\n    plots.plot_chain(results, metric)"),
        ("markdown", CONCLUSION),
    ]),
    "summary": nb([
        ("markdown", "# Results summary\n\nEvery finished results folder, from `artifacts/results/` (or `$EROSION_ARTIFACTS/results/`)."),
        ("code", "from erosion_sim.experiments.registry import EXPERIMENTS\nfrom erosion_sim.experiments.results import list_results\n\nrows = list_results()\nfor r in rows:\n    title = EXPERIMENTS[r[\"name\"]].title if r[\"name\"] in EXPERIMENTS else \"\"\n    print(f\"{r['name']:<20} {title:<26} {r['suite']:<14} {r['time']}\")\n    print(f\"    {r['models']}\")"),
    ]),
}


def erosion_notebook(exp) -> dict:
    name = exp.name
    sweep = exp.settings is not None and exp.settings.step_sweep
    chains = exp.settings is not None and exp.settings.chains
    cells = [
        ("markdown", header(name, exp.title, RUN.format(name=name))),
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
        ("markdown", header(name, exp.title, RUN.format(name=name))),
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
