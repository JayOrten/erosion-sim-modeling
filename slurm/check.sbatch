#!/bin/bash
# Check that the environments run on a GPU compute node with no internet access.
# Submit from the repo root: sbatch slurm/check.sbatch
#SBATCH --job-name=check
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
# QOS dw87 (dw partition, A100s) by default; override with `SBATCH_QOS=cs` (cs, cs2, cs3) or `SBATCH_QOS=gpu`.
#SBATCH --qos=dw87
# torch cu130 wheels need compute capability >= 7.5 and x86: this excludes P100 (pascal) and GH200 (ARM).
# Node tags: zen3 = dw/cs A100, sapphirerapids = cs2 H100, graniterapids = cs3 B200, hopper = H200, lovelace = L40S.
#SBATCH --constraint=zen3|sapphirerapids|graniterapids|hopper|lovelace
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
