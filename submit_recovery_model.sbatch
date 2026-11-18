#!/bin/bash
#SBATCH --job-name=recovery_model
#SBATCH --account=brics.b5ac
#SBATCH --partition=workq
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/lus/lfs1aip2/scratch/b5ac/freddieyu.b5ac/joint_graph_work/logs/%x-%j.out
#SBATCH --error=/lus/lfs1aip2/scratch/b5ac/freddieyu.b5ac/joint_graph_work/logs/%x-%j.err

set -euo pipefail

source /home/b5ac/freddieyu.b5ac/miniforge3/etc/profile.d/conda.sh
conda activate hf-hpc

cd /home/b5ac/freddieyu.b5ac/project_2
export PYTHONPATH=/home/b5ac/freddieyu.b5ac/project_2
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

SCRATCH=/lus/lfs1aip2/scratch/b5ac/freddieyu.b5ac/joint_graph_work
mkdir -p "$SCRATCH/logs" "$RUN_DIR/logs"

set +e
python -m src.run_recovery_model --run-dir "$RUN_DIR" --method "$METHOD" 2>&1 | tee "$RUN_DIR/logs/model-${METHOD}.log"
status=${PIPESTATUS[0]}
set -e

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  cp "$SCRATCH/logs/recovery_model-${SLURM_JOB_ID}.out" "$RUN_DIR/logs/slurm-${METHOD}-${SLURM_JOB_ID}.out" || true
  cp "$SCRATCH/logs/recovery_model-${SLURM_JOB_ID}.err" "$RUN_DIR/logs/slurm-${METHOD}-${SLURM_JOB_ID}.err" || true
fi
exit "$status"
