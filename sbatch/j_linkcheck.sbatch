#!/bin/bash
#SBATCH --job-name=da-linkcheck
#SBATCH --account=brics.b5ac
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/projects/b5ac/project_2/results/03_integrated_analysis/data_analysis_linkcheck_round1/logs/linkcheck_%j.out
#SBATCH --error=/projects/b5ac/project_2/results/03_integrated_analysis/data_analysis_linkcheck_round1/logs/linkcheck_%j.err
#SBATCH --chdir=/home/b5ac/freddieyu.b5ac/project_2

set -euo pipefail
echo "START_SHELL job=${SLURM_JOB_ID} host=$(hostname) utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) cpus=${SLURM_CPUS_PER_TASK}"
export PYTHONPATH=/home/b5ac/freddieyu.b5ac/project_2
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
exec /projects/b5ac/project_2/envs/core_semmed_py312/bin/python -u /home/b5ac/freddieyu.b5ac/project_2/03_integrated_analysis/report_review/run_linkcheck.py
