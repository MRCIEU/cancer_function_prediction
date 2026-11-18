#!/bin/bash
#SBATCH --job-name=citation_audit
#SBATCH --account=brics.b5ac
#SBATCH --partition=workq
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --output=/lus/lfs1aip2/scratch/b5ac/freddieyu.b5ac/joint_graph_work/logs/%x-%j.out
#SBATCH --error=/lus/lfs1aip2/scratch/b5ac/freddieyu.b5ac/joint_graph_work/logs/%x-%j.err

# SCOPE=dry profiles hub groups after checking the published path totals.
# SCOPE=formal writes results/citation_path_evidence_audit_<UTC>/.

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

python -m unittest tests.test_citation_features tests.test_path_evidence

SCOPE=${SCOPE:-dry}
if [[ "$SCOPE" == "formal" ]]; then
  STAMP=$(date -u +%Y%m%dT%H%MZ)
  RUN_DIR=/home/b5ac/freddieyu.b5ac/projects/project_2/results/citation_path_evidence_audit_${STAMP}
  WITNESS=all
  LARGE=--write-large
  WORKERS=4
else
  RUN_DIR=$SCRATCH/citation_audit_dry_${SLURM_JOB_ID:-manual}
  WITNESS=hubs
  LARGE=
  WORKERS=1
fi

mkdir -p "$RUN_DIR/logs" "$RUN_DIR/analysis"
printf 'scope=%s\nrun_dir=%s\njob_id=%s\nwitness=%s\n' \
  "$SCOPE" "$RUN_DIR" "${SLURM_JOB_ID:-}" "$WITNESS" \
  | tee "$RUN_DIR/logs/invocation.txt"

set +e
python -m src.run_citation_audit \
  --run-dir "$RUN_DIR" \
  --witness "$WITNESS" \
  --workers "$WORKERS" \
  $LARGE \
  2>&1 | tee "$RUN_DIR/logs/run.log"
status=${PIPESTATUS[0]}
set -e

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  cp "$SCRATCH/logs/citation_audit-${SLURM_JOB_ID}.out" "$RUN_DIR/logs/slurm-${SLURM_JOB_ID}.out" || true
  cp "$SCRATCH/logs/citation_audit-${SLURM_JOB_ID}.err" "$RUN_DIR/logs/slurm-${SLURM_JOB_ID}.err" || true
fi

echo "completed $RUN_DIR status=$status"
exit "$status"
