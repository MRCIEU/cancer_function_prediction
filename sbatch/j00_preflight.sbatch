#!/bin/bash
#SBATCH --job-name=da-j00-preflight
#SBATCH --account=brics.b5ac
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/projects/b5ac/project_2/results/03_integrated_analysis/data_analysis_round1/logs/j00_%j.out
#SBATCH --error=/projects/b5ac/project_2/results/03_integrated_analysis/data_analysis_round1/logs/j00_%j.err
#SBATCH --chdir=/home/b5ac/freddieyu.b5ac/project_2

set -euo pipefail
echo "START_SHELL job=${SLURM_JOB_ID} host=$(hostname) utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) cpus=${SLURM_CPUS_PER_TASK}"
export PYTHONPATH=/home/b5ac/freddieyu.b5ac/project_2
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MPLBACKEND=Agg
exec /home/b5ac/freddieyu.b5ac/miniforge3/bin/python -u /home/b5ac/freddieyu.b5ac/project_2/03_integrated_analysis/run_preflight.py
