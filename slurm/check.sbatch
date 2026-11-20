#!/bin/bash
# Check that the environments run on a GPU compute node with no internet access.
# Submit from the repo root: sbatch slurm/check.sbatch
#SBATCH --job-name=check
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export UV_OFFLINE=1  # fail at once instead of trying to download

echo "node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

uv run --no-sync python - <<'PY'
import torch

print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0))
# torch.compile builds kernels with Triton, which needs a working compiler on the node.
f = torch.compile(lambda x: torch.nn.functional.gelu(x @ x.T))
print("compiled matmul ok:", f(torch.randn(256, 256, device="cuda")).shape)
from erosion_sim.paths import ARTIFACTS_DIR
print("erosion_sim artifacts dir:", ARTIFACTS_DIR)
PY

echo "== main tests"
uv run --no-sync pytest -q

echo "== datagen tests"
(cd datagen && uv run --no-sync pytest -q)
