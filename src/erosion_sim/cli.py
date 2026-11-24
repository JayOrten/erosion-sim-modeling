"""Command-line entry point, ``erosion-sim``. Each subcommand runs one stage; the Slurm scripts call them.

- ``erosion-sim list``: every dataset, model, predictor, and evaluation in ``configs/``, with what is
  generated, trained, and evaluated.
- ``erosion-sim train <model>``: train ``configs/models/<model>.yaml`` into ``artifacts/models/<model>/``,
  resuming if it was interrupted. Does nothing if it is finished.
- ``erosion-sim evaluate <evaluation> <predictor> [--force]``: score a model, cascade, or baseline under
  ``configs/evaluations/<evaluation>.yaml``, writing ``artifacts/results/<evaluation>/<predictor>/``.
  ``--force`` replaces existing results.
- ``erosion-sim dla-generate <dataset> [split ...]``: simulate the splits of the DLA dataset in
  ``configs/data/<dataset>.yaml`` (``dla/data.py``). Skips finished splits.
"""

import argparse


def _list(args) -> None:
    from .config import config_names, data_names
    from .evaluation.evaluate import predictor_config
    from .paths import DATA_DIR, RESULTS_DIR, is_finished, model_dir

    def state(name: str) -> str:
        folder = model_dir(name)
        return "trained" if is_finished(folder) else ("started" if folder.exists() else "")

    print("datasets (finished splits)")
    for name in data_names():
        done = [s for s in ("train", "val", "test") if (DATA_DIR / name / s / "meta.json").exists()]
        print(f"    {name:<32} {', '.join(done)}")
    print("models")
    for name in config_names("models"):
        print(f"    {name:<32} {state(name)}")
    print("predictors (besides models)")
    for name in config_names("predictors"):
        spec = predictor_config(name)
        what = ", ".join(spec["levels"]) if spec["kind"] == "cascade" else spec["method"]
        print(f"    {name:<32} {spec['kind']:<9} {what}")
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
    from .dla.config import SPLITS, data_config
    from .dla.data import make_split

    cfg = data_config(args.profile)
    for split in args.splits or SPLITS:
        print(f"{cfg.profile}/{split}: {cfg.runs(split):,} runs", flush=True)
        path = make_split(cfg, split, progress_every=args.progress_every, processes=args.processes)
        print(f"done: {path}", flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="erosion-sim")
    commands = parser.add_subparsers(dest="command", required=True)

    listing = commands.add_parser("list", help="datasets, models, predictors, and evaluations, with what is done")
    listing.set_defaults(run=_list)

    train = commands.add_parser("train", help="train a model, resuming if interrupted")
    train.add_argument("model", help="a file name in configs/models/, without .yaml")
    train.set_defaults(run=_train)

    evaluate = commands.add_parser("evaluate", help="score a model, cascade, or baseline under an evaluation")
    evaluate.add_argument("evaluation", help="a file name in configs/evaluations/, without .yaml")
    evaluate.add_argument("predictor", help="a model (configs/models/), or a cascade or baseline (configs/predictors/)")
    evaluate.add_argument("--force", action="store_true", help="replace existing results")
    evaluate.set_defaults(run=_evaluate)

    dla = commands.add_parser("dla-generate", help="simulate DLA dataset splits")
    dla.add_argument("profile", help="a DLA dataset file in configs/data/, without .yaml, e.g. dla_master")
    dla.add_argument("splits", nargs="*", help="train, val, test; none means all three")
    dla.add_argument("--progress-every", type=int, default=2_000)
    dla.add_argument("--processes", type=int, help="default: every core the process may use")
    dla.set_defaults(run=_dla_generate)

    args = parser.parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    main()
