#!/bin/bash
# Score one model or world under one evaluation, on one GPU. Submit from the repo root:
#   sbatch slurm/evaluate.sbatch <evaluation> <predictor> [--force]
# Results go to artifacts/results/<evaluation>/<predictor>/, with everything it prints in evaluate.log.
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
