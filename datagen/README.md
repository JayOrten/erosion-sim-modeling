# erosion-datagen

Generates the erosion datasets that the main project trains on. It has its own environment because it
uses `richdem` for pit filling, and `richdem` 0.3.4 (its last release) needs Python 3.10 or older and
numpy 1. See `../docs/simulator-timing.md`.

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
    make_dataset(PROFILES["full"], split, progress_every=500)
```

Output goes to `../artifacts/data/<profile>/<split>/`: `snapshots.npy` (float32, shape
`(runs, snapshots, rows, cols)`), `params.npz`, and `meta.json`. The main project reads these files
with plain numpy.

`notebooks/demo.ipynb` walks through one run and a small dataset. Use `.venv` in this folder as its
kernel.

## Tests

```sh
uv run pytest
```
