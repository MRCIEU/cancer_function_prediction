#!/bin/bash
#SBATCH --job-name=path_evidence
#SBATCH --account=brics.b5ac
#SBATCH --partition=workq
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/lus/lfs1aip2/scratch/b5ac/freddieyu.b5ac/joint_graph_work/logs/%x-%j.out
#SBATCH --error=/lus/lfs1aip2/scratch/b5ac/freddieyu.b5ac/joint_graph_work/logs/%x-%j.err

# SCOPE=dry profiles PMID overlap on high-degree cancers and writes to scratch.
# SCOPE=formal writes the complete audit under the project results directory.
# The wall time below is the dry-run request. Formal submission overrides --time and --mem.

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

python -m unittest tests.test_path_evidence tests.test_graph_audit

SCOPE=${SCOPE:-dry}
WORKERS=${SLURM_CPUS_PER_TASK:-4}
if [[ "$SCOPE" == "formal" ]]; then
  STAMP=$(date -u +%Y%m%dT%H%MZ)
  RUN_DIR=/home/b5ac/freddieyu.b5ac/projects/project_2/results/path_evidence_audit_${STAMP}
  CANCER_LIMIT=0
else
  RUN_DIR=$SCRATCH/path_evidence_dry_${SLURM_JOB_ID:-manual}
  CANCER_LIMIT=${CANCER_LIMIT:-5}
fi

mkdir -p "$RUN_DIR/logs" "$RUN_DIR/analysis"
printf 'scope=%s\nrun_dir=%s\njob_id=%s\ncancer_limit=%s\nworkers=%s\n' \
  "$SCOPE" "$RUN_DIR" "${SLURM_JOB_ID:-}" "$CANCER_LIMIT" "$WORKERS" \
  | tee "$RUN_DIR/logs/invocation.txt"

python -m src.run_path_evidence \
  --run-dir "$RUN_DIR" \
  --cancer-limit "$CANCER_LIMIT" \
  --workers "$WORKERS" \
  2>&1 | tee "$RUN_DIR/logs/run.log"

echo "completed $RUN_DIR"
