#!/bin/bash
# Simulate a DLA dataset profile on one CPU node. Submit from the repo root:
#   sbatch slurm/dla_generate.sbatch <dataset> [split ...]      e.g. dla_master
# Datasets are the files in configs/data/ with generator: dla. No splits means train, val, and test.
# Writes artifacts/data/<dataset>/<split>/; finished splits are skipped. One process per core of the job.
#SBATCH --job-name=dla_generate
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export UV_OFFLINE=1
echo "== start $(date -Is) on $(hostname), job ${SLURM_JOB_ID:-local}, cores $SLURM_CPUS_PER_TASK"
uv run --no-sync erosion-sim dla-generate "$@"
