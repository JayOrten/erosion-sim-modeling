"""Command-line entry point, ``erosion-sim``. Each subcommand runs one stage, for use from Slurm jobs.

- ``erosion-sim dla-generate <profile> [split ...]``: simulate DLA splits (``dla/data.py``). Runs
  locally; the DLA data is small and fast.

Training, evaluation, and submission are added in later steps (docs/plans/implementation-plan.md).
"""

import argparse


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

    dla = commands.add_parser("dla-generate", help="simulate DLA dataset splits")
    dla.add_argument("profile", help="a profile in erosion_sim.dla.config.PROFILES, e.g. dla_master")
    dla.add_argument("splits", nargs="*", help="train, val, test; none means all three")
    dla.add_argument("--progress-every", type=int, default=2_000)
    dla.set_defaults(run=_dla_generate)

    args = parser.parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    main()
