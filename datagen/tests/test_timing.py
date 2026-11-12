import json
import subprocess
import sys

import numpy as np

from erosion_datagen.timing import time_simulator
from erosion_terrain import fbm

PARAMS = dict(K=3e-4, D=3e-4, m=0.4, U=0.0, dt=100.0, steps=5)


def test_time_simulator():
    assert time_simulator(64 * fbm((16, 16), period=8.0), **PARAMS, repeats=2) > 0


def test_time_simulator_as_process(tmp_path):
    request = tmp_path / "request.npz"
    np.savez(request, z0=64 * fbm((16, 16), period=8.0), **PARAMS)
    result = subprocess.run([sys.executable, "-m", "erosion_datagen.timing", str(request)],
                            capture_output=True, text=True, check=True)
    assert json.loads(result.stdout.strip().splitlines()[-1])["seconds"] > 0
