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

The starting terrain is Perlin fBm from FastNoiseLite (`src/erosion_datagen/terrain.py`). The
`smoke`, `small`, and `full` datasets on disk were made with an earlier hand-written Perlin generator,
rescaled to exactly 64 m of relief per run. That code is at commit `15c6de9`; the current code no
longer reproduces those datasets.

Output goes to `../artifacts/data/<profile>/<split>/`: `snapshots.npy` (float32, shape
`(runs, snapshots, rows, cols)`), `params.npz`, and `meta.json`. The main project reads these files
with plain numpy.

`notebooks/demo.ipynb` walks through one run and a small dataset. Use `.venv` in this folder as its
kernel.

## Tests

```sh
uv run pytest
```
