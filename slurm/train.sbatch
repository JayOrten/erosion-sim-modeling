#!/bin/bash
# Train one model on one GPU. Submit from the repo root:
#   sbatch slurm/train.sbatch <model>
# Everything it prints also goes to artifacts/models/<model>/train.log.
# If a job dies, submitting it again resumes from the model's last.pt.
#SBATCH --job-name=train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export UV_OFFLINE=1
echo "== start $(date -Is) on $(hostname), job ${SLURM_JOB_ID:-local}"
uv run --no-sync erosion-sim train "$@"
