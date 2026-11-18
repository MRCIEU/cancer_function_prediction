#!/bin/bash
#SBATCH --job-name=recovery_merge
#SBATCH --account=brics.b5ac
#SBATCH --partition=workq
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/lus/lfs1aip2/scratch/b5ac/freddieyu.b5ac/joint_graph_work/logs/%x-%j.out
#SBATCH --error=/lus/lfs1aip2/scratch/b5ac/freddieyu.b5ac/joint_graph_work/logs/%x-%j.err

set -euo pipefail

source /home/b5ac/freddieyu.b5ac/miniforge3/etc/profile.d/conda.sh
conda activate hf-hpc

cd /home/b5ac/freddieyu.b5ac/project_2
export PYTHONPATH=/home/b5ac/freddieyu.b5ac/project_2

mkdir -p "$RUN_DIR/logs"
python -m src.merge_recovery_report --run-dir "$RUN_DIR" 2>&1 | tee "$RUN_DIR/logs/merge.log"
