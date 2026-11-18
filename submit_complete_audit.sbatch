#!/bin/bash
#SBATCH --job-name=path_audit_complete
#SBATCH --account=brics.b5ac
#SBATCH --partition=workq
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/lus/lfs1aip2/scratch/b5ac/freddieyu.b5ac/joint_graph_work/logs/%x-%j.out
#SBATCH --error=/lus/lfs1aip2/scratch/b5ac/freddieyu.b5ac/joint_graph_work/logs/%x-%j.err

# SCOPE=dry runs the exact totals and a small concentration sample.
# SCOPE=formal writes results/path_evidence_audit_completion_<UTC>/.

set -euo pipefail

source /home/b5ac/freddieyu.b5ac/miniforge3/etc/profile.d/conda.sh
conda activate hf-hpc

cd /home/b5ac/freddieyu.b5ac/project_2
export PYTHONPATH=/home/b5ac/freddieyu.b5ac/project_2
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

SCRATCH=/lus/lfs1aip2/scratch/b5ac/freddieyu.b5ac/joint_graph_work
mkdir -p "$SCRATCH/logs"

python -m unittest tests.test_complete_audit tests.test_path_evidence

SCOPE=${SCOPE:-dry}
if [[ "$SCOPE" == "formal" ]]; then
  STAMP=$(date -u +%Y%m%dT%H%MZ)
  RUN_DIR=/home/b5ac/freddieyu.b5ac/projects/project_2/results/path_evidence_audit_completion_${STAMP}
  SAMPLE_QUERIES=0
else
  RUN_DIR=$SCRATCH/path_audit_complete_dry_${SLURM_JOB_ID:-manual}
  SAMPLE_QUERIES=${SAMPLE_QUERIES:-3}
fi

mkdir -p "$RUN_DIR/logs" "$RUN_DIR/analysis" "$RUN_DIR/figures"
printf 'scope=%s\nrun_dir=%s\njob_id=%s\nsample_queries=%s\n' \
  "$SCOPE" "$RUN_DIR" "${SLURM_JOB_ID:-}" "$SAMPLE_QUERIES" \
  | tee "$RUN_DIR/logs/invocation.txt"

python -m src.run_complete_audit \
  --run-dir "$RUN_DIR" \
  --sample-queries "$SAMPLE_QUERIES" \
  2>&1 | tee "$RUN_DIR/logs/run.log"

echo "completed $RUN_DIR"
