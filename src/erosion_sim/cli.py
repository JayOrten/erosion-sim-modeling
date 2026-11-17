"""Command-line entry point, ``erosion-sim``. Each subcommand runs one stage, for use from Slurm jobs.

- ``erosion-sim dla-generate <profile> [split ...]``: simulate DLA splits (``dla/data.py``). Runs
  locally; the DLA data is small and fast.
- ``erosion-sim train <profile> [--set field=value ...]``: train one model in its model folder
  (``identity.model_dir``), resuming if it was interrupted. Does nothing if the model is finished.

Evaluation and submission are added in later steps (docs/plans/implementation-plan.md).
"""

import argparse
import ast
import os
import sys
from dataclasses import replace


def _dla_generate(args) -> None:
    from .dla.config import PROFILES, SPLITS
    from .dla.data import make_split

    cfg = PROFILES[args.profile]
    for split in args.splits or SPLITS:
        print(f"{cfg.profile}/{split}: {cfg.runs(split):,} runs", flush=True)
        path = make_split(cfg, split, progress_every=args.progress_every)
        print(f"wrote {path}", flush=True)


def check_commit() -> None:
    """Inside a submitted job, fail unless the repo is still at the commit the job was submitted with
    (``EROSION_COMMIT``) and ``src/`` has no uncommitted changes. Does nothing when the variable is unset."""
    from .paths import git_state

    expected = os.environ.get("EROSION_COMMIT")
    if not expected:
        return
    state = git_state()
    if state["commit"] != expected or state["dirty"]:
        sys.exit(f"the job was submitted at commit {expected}, but the repo is at {state['commit']}"
                 f"{' with uncommitted changes in src/' if state['dirty'] else ''}. Commit and resubmit.")


def parse_overrides(items: list[str]) -> dict:
    """``["steps=500", "lr=1e-3", "data=(('fractal', 'fbm'),)"]`` as a dict. Values are Python literals;
    anything else is kept as a string."""
    overrides = {}
    for item in items:
        name, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"--set takes field=value, got {item!r}")
        try:
            overrides[name] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            overrides[name] = value
    return overrides


def _train(args) -> None:
    from .config import PROFILES
    from .identity import is_finished, model_dir
    from .train import train

    check_commit()
    cfg = replace(PROFILES[args.profile], **parse_overrides(args.set))
    folder = model_dir(cfg)
    if is_finished(folder):
        print(f"{folder} is already trained", flush=True)
        return
    train(cfg, out=folder)
    print(f"finished: {folder}", flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="erosion-sim")
    commands = parser.add_subparsers(dest="command", required=True)

    dla = commands.add_parser("dla-generate", help="simulate DLA dataset splits")
    dla.add_argument("profile", help="a profile in erosion_sim.dla.config.PROFILES, e.g. dla_master")
    dla.add_argument("splits", nargs="*", help="train, val, test; none means all three")
    dla.add_argument("--progress-every", type=int, default=2_000)
    dla.set_defaults(run=_dla_generate)

    train = commands.add_parser("train", help="train one model, resuming if interrupted")
    train.add_argument("profile", help="a profile in erosion_sim.config.PROFILES")
    train.add_argument("--set", action="append", default=[], metavar="FIELD=VALUE",
                       help="replace a config field, e.g. --set steps=500; repeatable")
    train.set_defaults(run=_train)

    args = parser.parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    main()
