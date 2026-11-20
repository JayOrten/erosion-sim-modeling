#!/bin/bash
# Run one experiment's evaluation suite on one GPU. Submit from the repo root:
#   erosion-sim submit <experiment>      (submits every job an experiment needs), or by hand:
#   sbatch slurm/evaluate.sbatch <experiment>
# Results go to artifacts/results/<experiment>/
#SBATCH --job-name=evaluate
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export UV_OFFLINE=1
echo "== start $(date -Is) on $(hostname), job ${SLURM_JOB_ID:-local}"
uv run --no-sync erosion-sim evaluate "$@"
