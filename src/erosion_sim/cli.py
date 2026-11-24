"""Command-line entry point, ``erosion-sim``. Each subcommand runs one stage; the Slurm scripts call them.

- ``erosion-sim list``: every model, world, and evaluation in ``configs/``, with what is trained and
  evaluated.
- ``erosion-sim train <model>``: train ``configs/models/<model>.yaml`` into ``artifacts/models/<model>/``,
  resuming if it was interrupted. Does nothing if it is finished.
- ``erosion-sim evaluate <evaluation> <predictor> [--force]``: score a model or world under
  ``configs/evaluations/<evaluation>.yaml``, writing ``artifacts/results/<evaluation>/<predictor>/``.
  ``--force`` replaces existing results.
- ``erosion-sim dla-generate <profile> [split ...]``: simulate DLA splits (``dla/data.py``). Runs
  locally; the DLA data is small and fast.
"""

import argparse


def _list(args) -> None:
    from .config import config_names
    from .evaluation.evaluate import world_config
    from .paths import RESULTS_DIR, is_finished, model_dir

    def state(name: str) -> str:
        folder = model_dir(name)
        return "trained" if is_finished(folder) else ("started" if folder.exists() else "")

    print("models")
    for name in config_names("models"):
        print(f"    {name:<32} {state(name)}")
    print("worlds")
    for name in config_names("worlds"):
        print(f"    {name:<32} {', '.join(world_config(name)['levels'])}")
    print("evaluations (predictors with results)")
    for name in config_names("evaluations"):
        done = sorted(m.parent.name for m in (RESULTS_DIR / name).glob("*/manifest.json"))
        print(f"    {name:<32} {', '.join(done)}")


def _train(args) -> None:
    from .training.run import train_model

    print(f"finished: {train_model(args.model)}", flush=True)


def _evaluate(args) -> None:
    from .evaluation.evaluate import evaluate

    print(f"wrote {evaluate(args.evaluation, args.predictor, force=args.force)}", flush=True)


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

    listing = commands.add_parser("list", help="models, worlds, and evaluations, with what is done")
    listing.set_defaults(run=_list)

    train = commands.add_parser("train", help="train a model, resuming if interrupted")
    train.add_argument("model", help="a file name in configs/models/, without .yaml")
    train.set_defaults(run=_train)

    evaluate = commands.add_parser("evaluate", help="score a model or world under an evaluation")
    evaluate.add_argument("evaluation", help="a file name in configs/evaluations/, without .yaml")
    evaluate.add_argument("predictor", help="a model (configs/models/) or world (configs/worlds/)")
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
