#!/bin/bash
# Simulate one erosion group on one CPU node. Submit from the repo root:
#   sbatch slurm/datagen.sbatch <dataset>/<group> [split ...]      e.g. erosion_master/main_256
# Groups are the files in configs/data/<dataset>/ (`datagen/.venv/bin/erosion-datagen list`). Uses every core
# of the job. If the job hits its time limit, submit it again: each split resumes from the runs on disk, and
# finished splits are skipped.
#SBATCH --job-name=datagen
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR/datagen"
export UV_OFFLINE=1
echo "== start $(date -Is) on $(hostname), job ${SLURM_JOB_ID:-local}, cores $SLURM_CPUS_PER_TASK"
uv run --no-sync erosion-datagen generate "$@"
