#!/bin/bash
# Train one model on one GPU. Submit from the repo root:
#   erosion-sim submit <experiment>      (submits every job an experiment needs), or by hand:
#   sbatch slurm/train.sbatch <experiment> <model>
# If a job dies, submitting it again resumes from the model's last.pt.
#SBATCH --job-name=train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
# QOS dw87 (dw partition, A100s) by default; override with `SBATCH_QOS=cs` (cs, cs2, cs3) or `SBATCH_QOS=gpu`.
#SBATCH --qos=dw87
# torch cu130 wheels need compute capability >= 7.5 and x86: this excludes P100 (pascal) and GH200 (ARM).
# Node tags: zen3 = dw/cs A100, sapphirerapids = cs2 H100, graniterapids = cs3 B200, hopper = H200, lovelace = L40S.
#SBATCH --constraint=zen3|sapphirerapids|graniterapids|hopper|lovelace
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export UV_OFFLINE=1
echo "== start $(date -Is) on $(hostname), job ${SLURM_JOB_ID:-local}"
uv run --no-sync erosion-sim train "$@"
