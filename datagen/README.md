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

```python
from erosion_datagen.config import PROFILES
from erosion_datagen.simulate import make_dataset

for split in ["train", "val", "test"]:
    make_dataset(PROFILES["world"], split, progress_every=500)
```

Profiles are in `src/erosion_datagen/config.py`. `world` is the dataset for the infinite world
(`../docs/plans/infinite-plan.md`): the period is fixed at 32 cells.

Each run's starting terrain is an `erosion_terrain.Terrain` (in `../terrain/`, shared with the main
package), drawn per run from the profile's ranges. Datasets made before the terrain fields were stored
under `Terrain`'s names (with `terrain_seed`, and noise types as indices) do not open with `Split`;
regenerate them.

Output goes to `../artifacts/data/<profile>/<split>/`: `snapshots.npy` (float32, shape
`(runs, snapshots, rows, cols)`), `params.npz`, and `meta.json`. `params.npz` holds one array per run
for each erosion parameter (`K`, `K_ref`, `D`, `m`, `U`) and each field of `erosion_terrain.Terrain`,
plus the snapshot `times`. The main project opens a split with `erosion_sim.data.Split`.

`src/erosion_datagen/timing.py` times the simulator. The main package runs it as a separate process
to compare the simulator's speed with the models'.

`../notebooks/datagen/demo.ipynb` walks through one run and a small dataset. The notebooks in
`../notebooks/datagen/` use `.venv` in this folder as their kernel.

## Tests

```sh
uv run pytest
```
