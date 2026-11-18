#!/bin/bash
#SBATCH --job-name=joint_graph_audit
#SBATCH --account=brics.b5ac
#SBATCH --partition=workq
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/lus/lfs1aip2/scratch/b5ac/freddieyu.b5ac/joint_graph_work/logs/%x-%j.out
#SBATCH --error=/lus/lfs1aip2/scratch/b5ac/freddieyu.b5ac/joint_graph_work/logs/%x-%j.err

# MODE=dry parses a capped SemMedDB prefix after the full download.
# MODE=formal parses the complete pinned inputs and publishes the run directory.
# Do not put credentials in this script. UMLS_API_KEY is taken from the environment.

set -euo pipefail

source /home/b5ac/freddieyu.b5ac/miniforge3/etc/profile.d/conda.sh
conda activate hf-hpc

cd /home/b5ac/freddieyu.b5ac/project_2
export PYTHONPATH=/home/b5ac/freddieyu.b5ac/project_2
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

SCRATCH=/lus/lfs1aip2/scratch/b5ac/freddieyu.b5ac/joint_graph_work
DATA_ROOT=/home/b5ac/freddieyu.b5ac/projects/project_2/data
mkdir -p "$SCRATCH/logs" "$SCRATCH/prepared" "$SCRATCH/alignment"

python -m unittest tests.test_graph_audit

MODE=${MODE:-dry}
if [[ "$MODE" == "formal" ]]; then
  STAMP=$(date -u +%Y%m%dT%H%MZ)
  RUN_DIR=/home/b5ac/freddieyu.b5ac/projects/project_2/results/joint_graph_and_evidence_audit_${STAMP}
  MAX_PRED=0
  MAX_ENT=0
else
  RUN_DIR=$SCRATCH/dry_${SLURM_JOB_ID:-manual}
  MAX_PRED=${MAX_PREDICATION_ROWS:-2000000}
  MAX_ENT=${MAX_ENTITY_ROWS:-2000000}
fi

mkdir -p "$RUN_DIR/logs" "$RUN_DIR/analysis" "$RUN_DIR/figures" "$RUN_DIR/graph"
cp config/study_design.yaml "$RUN_DIR/study_design.yaml"
cp config/data_sources.yaml "$RUN_DIR/data_sources.yaml"
printf 'mode=%s\nrun_dir=%s\njob_id=%s\n' "$MODE" "$RUN_DIR" "${SLURM_JOB_ID:-}" | tee "$RUN_DIR/logs/invocation.txt"

python -m src.prepare_data \
  --stage all \
  --sources config/data_sources.yaml \
  --data-root "$DATA_ROOT" \
  --scratch "$SCRATCH/prepared" \
  --max-predication-rows "$MAX_PRED" \
  --max-entity-rows "$MAX_ENT"

python -m src.align_entities --scratch "$SCRATCH/prepared" --out "$SCRATCH/alignment"
python -m src.build_joint_graph \
  --scratch "$SCRATCH/prepared" \
  --alignment "$SCRATCH/alignment" \
  --design "$RUN_DIR/study_design.yaml" \
  --run-dir "$RUN_DIR"
python -m src.extract_evidence_paths \
  --run-dir "$RUN_DIR" \
  --scratch "$SCRATCH/prepared" \
  --alignment "$SCRATCH/alignment"
python -m src.audit_evidence \
  --run-dir "$RUN_DIR" \
  --scratch "$SCRATCH/prepared" \
  --alignment "$SCRATCH/alignment" \
  --data-root "$DATA_ROOT"

echo "completed $RUN_DIR"
