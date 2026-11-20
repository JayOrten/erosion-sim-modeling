# erosion-datagen

Generates the erosion datasets that the main project trains on. It has its own environment because it
uses `richdem` for pit filling, and `richdem` 0.3.4 (its last release) needs Python 3.10 or older and
numpy 1. See `../docs/notes/simulator-timing.md`.

## Setup

```sh
cd datagen
uv sync
```

`richdem` builds from source. `pyproject.toml` passes the compiler flag it needs (`-include cstdint`).

## Use

```sh
uv run erosion-datagen list                      # every group: grid, runs per split, snapshots, disk
uv run erosion-datagen generate main_256         # every split of a group; resumes, skips finished splits
uv run erosion-datagen chunked main_512 test --chunk 256   # the runs again as closed chunks (C8)
```

On the cluster, `sbatch slurm/datagen.sbatch <group> [split ...]` from the repo root runs `generate` on
one CPU node, with a process per core.

Groups are in `src/erosion_datagen/config.py` (``GROUPS``), following the experiments plan's "Erosion
data" section: `main_256` to `main_2048` (one per scale band), `ood_*`, `long_*`, and two `smoke` groups.
Every column of a run is drawn independently from the run's own random stream, seeded by SHA-256 of
(dataset, group, split, index), so a run does not change if its split grows. `real_dem` groups are
deferred (they need elevation tiles downloaded first).

Each run's starting terrain is an `erosion_terrain.Terrain` and its erodibility map an
`erosion_terrain.ErodibilityMap` (in `../terrain/`, shared with the main package), so the world can
compute both anywhere. The simulator uses `K × map` per cell.

Output goes to `../artifacts/data/<dataset>/<group>/<split>/`, or under the folder set in `../local.json`
(see the main README): `snapshots.npy` (float32, `(runs, snapshots, rows, cols)`), `runs.npz`, and `meta.json`.
`runs.npz` is the run table, one array per column: `group`, `split`, `index`, `run_seed`, every field of
`Terrain` (`seed` is the noise seed), `K_ref`, `K`, `D`, `m`, `U` (always 0), `ood_param`, and the map
columns. `meta.json` holds the group settings and the snapshot `times`, and is written last. The main
project opens a split with `erosion_sim.data.splits.Split.open("erosion_master/main_256", "train")`.

`measure.py` has the measurements behind the group settings (feature size, edge reach, speed;
`../docs/notes/datagen-measurements.md`). `timing.py` times the simulator; the main package runs it as a
separate process to compare the simulator's speed with the models'. `render.py` makes 3D renders.

## Tests

```sh
uv run pytest
```
