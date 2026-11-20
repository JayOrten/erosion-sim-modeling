"""Command-line entry point, ``erosion-sim``. Each subcommand runs one stage; the Slurm scripts call them.

- ``erosion-sim list``: the registered experiments (``experiments/registry.py``) and their models.
- ``erosion-sim submit <experiment> [--dry-run] [--force-evaluate]``: from the login node, submit the
  experiment's training jobs and its evaluation job (``runner.submit``).
- ``erosion-sim train <experiment> <model>``: train one model of an experiment, resuming if it was
  interrupted. Does nothing if it is finished.
- ``erosion-sim evaluate <experiment> [--force]``: run the experiment's suite, writing
  ``artifacts/results/<experiment>/``. ``--force`` replaces existing results.
- ``erosion-sim dla-generate <profile> [split ...]``: simulate DLA splits (``dla/data.py``). Runs
  locally; the DLA data is small and fast.
"""

import argparse


def _list(args) -> None:
    from .experiments.registry import EXPERIMENTS, resolve
    from .experiments.results import results_dir
    from .paths import is_finished, model_dir

    for name, exp in EXPERIMENTS.items():
        done = (results_dir(name) / "manifest.json").exists()
        print(f"{name:<20} {exp.title}  [{exp.suite}]{'  results' if done else ''}")
        for label in exp.models:
            owner, owner_label, _ = resolve(name, label)
            folder = model_dir(owner, owner_label)
            state = "trained" if is_finished(folder) else ("started" if folder.exists() else "")
            print(f"    {label:<16} {owner + '/' + owner_label:<32} {state}")


def _submit(args) -> None:
    from .experiments.runner import submit

    submit(args.experiment, dry_run=args.dry_run, force_evaluate=args.force_evaluate)


def _train(args) -> None:
    from .experiments.runner import train_model

    print(f"finished: {train_model(args.experiment, args.model)}", flush=True)


def _evaluate(args) -> None:
    from .experiments.runner import evaluate

    print(f"wrote {evaluate(args.experiment, force=args.force)}", flush=True)


def _dla_generate(args) -> None:
    from .dla.config import PROFILES, SPLITS
    from .dla.data import make_split

    cfg = PROFILES[args.profile]
    for split in args.splits or SPLITS:
        print(f"{cfg.profile}/{split}: {cfg.runs(split):,} runs", flush=True)
        path = make_split(cfg, split, progress_every=args.progress_every)
        print(f"wrote {path}", flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="erosion-sim")
    commands = parser.add_subparsers(dest="command", required=True)

    listing = commands.add_parser("list", help="registered experiments and the state of their models")
    listing.set_defaults(run=_list)

    submit = commands.add_parser("submit", help="submit an experiment's training and evaluation jobs")
    submit.add_argument("experiment")
    submit.add_argument("--dry-run", action="store_true", help="print the sbatch commands without running them")
    submit.add_argument("--force-evaluate", action="store_true", help="redo the evaluation if results exist")
    submit.set_defaults(run=_submit)

    train = commands.add_parser("train", help="train one model of an experiment, resuming if interrupted")
    train.add_argument("experiment")
    train.add_argument("model", help="the model's label in the experiment")
    train.set_defaults(run=_train)

    evaluate = commands.add_parser("evaluate", help="run an experiment's evaluation suite")
    evaluate.add_argument("experiment")
    evaluate.add_argument("--force", action="store_true", help="replace existing results")
    evaluate.set_defaults(run=_evaluate)

    dla = commands.add_parser("dla-generate", help="simulate DLA dataset splits")
    dla.add_argument("profile", help="a profile in erosion_sim.dla.config.PROFILES, e.g. dla_master")
    dla.add_argument("splits", nargs="*", help="train, val, test; none means all three")
    dla.add_argument("--progress-every", type=int, default=2_000)
    dla.set_defaults(run=_dla_generate)

    args = parser.parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    main()
