"""Command-line entry point, ``erosion-datagen``. The Slurm script ``slurm/datagen.sbatch`` calls it.

A group is named by its file under the repo's ``configs/data/``, such as ``erosion_master/main_256``.

- ``erosion-datagen list``: every group, its runs per split, snapshots, and expected disk use.
- ``erosion-datagen generate <group> [split ...]``: simulate the splits (all by default) into
  ``artifacts/data/<group>/<split>/``. Resumes an interrupted split; skips a finished one.
- ``erosion-datagen chunked <group> <split> --chunk 256``: the same runs as separate closed chunks,
  stitched, for experiment C8's "independent chunks" baseline (``chunked_<chunk>.npy`` in the split).

Both run on every core the process may use (on Slurm, the job's), or ``--processes``.
"""

import argparse


def _list(args) -> None:
    from .config import group_names, load_group

    print(f"{'group':<26} {'size':>5} {'train':>6} {'val':>5} {'test':>5} {'snapshots':>9} {'disk (GB)':>9}")
    for name in group_names():
        g = load_group(name)
        runs = g.train_runs + g.val_runs + g.test_runs
        gb = runs * g.snapshots * g.size**2 * 4 / 1e9
        print(f"{name:<26} {g.size:>5} {g.train_runs:>6} {g.val_runs:>5} {g.test_runs:>5} {g.snapshots:>9} {gb:>9.1f}")


def _generate(args) -> None:
    from .config import load_group
    from .simulate import cores, dataset_dir, make_dataset

    group = load_group(args.group)
    for split in args.splits or group.splits:
        print(f"{group.profile}/{split}: {group.runs(split):,} runs of {group.size}² on "
              f"{args.processes or cores()} processes", flush=True)
        make_dataset(group, split, progress_every=args.progress_every, processes=args.processes)
        print(f"done: {dataset_dir(group, split)}", flush=True)


def _chunked(args) -> None:
    from .config import load_group
    from .simulate import make_chunked

    group = load_group(args.group)
    path = make_chunked(group, args.split, args.chunk, progress_every=args.progress_every, processes=args.processes)
    print(f"wrote {path}", flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="erosion-datagen")
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list", help="every group, with run counts and disk use")
    listing.set_defaults(run=_list)
    generate = commands.add_parser("generate", help="simulate a group's splits")
    generate.add_argument("group", help="a file under configs/data/, without .yaml, e.g. erosion_master/main_256")
    generate.add_argument("splits", nargs="*", help="train, val, test; none means every split the group has")
    generate.add_argument("--progress-every", type=int, default=50)
    generate.add_argument("--processes", type=int, help="default: every core the process may use")
    generate.set_defaults(run=_generate)
    chunked = commands.add_parser("chunked", help="re-simulate a split's runs as separate closed chunks")
    chunked.add_argument("group")
    chunked.add_argument("split")
    chunked.add_argument("--chunk", type=int, default=256)
    chunked.add_argument("--progress-every", type=int, default=100)
    chunked.add_argument("--processes", type=int, help="default: every core the process may use")
    chunked.set_defaults(run=_chunked)
    args = parser.parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    main()
