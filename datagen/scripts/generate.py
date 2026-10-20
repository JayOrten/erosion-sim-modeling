"""Generate every split of one profile.

Run from datagen/ with: uv run python scripts/generate.py small
"""

import sys

from erosion_datagen.config import PROFILES
from erosion_datagen.simulate import make_dataset

if __name__ == "__main__":
    cfg = PROFILES[sys.argv[1]]
    for split in ["train", "val", "test"]:
        out = make_dataset(cfg, split, progress_every=max(1, cfg.runs(split) // 10))
        print(f"wrote {out}", flush=True)
